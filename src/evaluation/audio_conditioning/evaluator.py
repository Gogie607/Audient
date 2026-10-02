"""Controlled evaluation of provider information versus Qwen prior."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch

from runweaver_ml.dataset_management import build_datasets, create_loader
from runweaver_ml.phase_control.execution import clear_trainable, system_state

from ...modules import QwenLanguageCore, TrainingModelWrapper
from ...training.audio_prefix import AudioPrefixForward
from .metrics import ConditionMetrics


SUPPORTED_CONDITIONS = (
    "correct",
    "shuffled_prefix",
    "zero_prefix",
    "no_prefix",
    "shuffled_frames",
)


class AudioConditioningEvaluator:
    def __init__(
        self,
        *,
        model,
        language_core,
        loader,
        conditions,
        max_batches: int | None,
        prompt: str = "",
        amp_enabled: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        shuffle_seed: int = 1729,
    ) -> None:
        unknown = set(conditions) - set(SUPPORTED_CONDITIONS)
        if unknown:
            raise ValueError(f"unsupported evaluation conditions: {sorted(unknown)}")
        if "correct" not in conditions:
            raise ValueError("evaluation conditions must include 'correct'")
        self.model = model
        self.language_core = language_core
        self.loader = loader
        self.conditions = tuple(conditions)
        self.max_batches = max_batches
        self.amp_enabled = amp_enabled and torch.cuda.is_available()
        self.amp_dtype = amp_dtype
        self.shuffle_generator = torch.Generator(device="cpu")
        self.shuffle_generator.manual_seed(shuffle_seed)
        self.adapter = AudioPrefixForward(language_core, prompt=prompt)

    @torch.no_grad()
    def run(self) -> tuple[dict, list[dict]]:
        provider = self.model.get(self.adapter.provider_name)
        metrics = {name: ConditionMetrics() for name in self.conditions}
        sample_records: list[dict] = []
        eval_mode = clear_trainable(self.model.get_mode())
        with system_state(self.model, eval_mode):
            for batch_index, batch in enumerate(self.loader):
                if self.max_batches is not None and batch_index >= self.max_batches:
                    break
                traits = batch.get("speech_traits", {})
                if batch.get("duration_s") is None and not (
                    isinstance(traits, Mapping) and traits.get("duration_s") is not None
                ):
                    raise KeyError(
                        "audio-conditioning evaluation requires duration_s "
                        "directly or inside speech_traits"
                    )
                features, _ = self.adapter.prepare_features(batch, provider)
                texts = self.adapter.text_values(batch, features.values.shape[0])
                with torch.amp.autocast(
                    "cuda", enabled=self.amp_enabled, dtype=self.amp_dtype
                ):
                    correct = provider(features).semantic
                    condition_streams = self._condition_streams(
                        provider, features, correct
                    )
                    batch_results = {}
                    for name in self.conditions:
                        values, padding_mask = condition_streams[name]
                        inputs = self.adapter.build_language_inputs(
                            values, padding_mask, texts
                        )
                        output = self.language_core(
                            inputs_embeds=inputs.inputs_embeds,
                            attention_mask=inputs.attention_mask,
                        )
                        batch_results[name] = metrics[name].update(
                            output.logits, inputs.labels
                        )
                        del output
                sample_records.extend(
                    self._sample_records(batch, texts, batch_results)
                )

        summary = {
            "conditions": {
                name: metric.summarize()
                for name, metric in metrics.items()
            },
            "paired_gains": self._paired_gains(sample_records),
            "sample_count": len(sample_records),
        }
        return summary, sample_records

    def _condition_streams(self, provider, features, correct):
        values = correct.values
        mask = correct.padding_mask
        batch_size = values.shape[0]
        streams = {"correct": (values, mask)}
        if "shuffled_prefix" in self.conditions:
            if batch_size < 2:
                raise ValueError("shuffled_prefix requires evaluation batch_size >= 2")
            order = torch.roll(torch.arange(batch_size, device=values.device), 1)
            streams["shuffled_prefix"] = (
                values.index_select(0, order),
                None if mask is None else mask.index_select(0, order.to(mask.device)),
            )
        if "zero_prefix" in self.conditions:
            streams["zero_prefix"] = (torch.zeros_like(values), mask)
        if "no_prefix" in self.conditions:
            token_id = getattr(self.language_core.tokenizer, "bos_token_id", None)
            if token_id is None:
                token_id = self.language_core.tokenizer.eos_token_id
            if token_id is None:
                raise ValueError("no_prefix condition requires a BOS or EOS token")
            token_ids = torch.full(
                (batch_size, 1),
                int(token_id),
                dtype=torch.long,
                device=self.language_core.device,
            )
            prior_prefix = self.language_core.embed_tokens(token_ids)
            streams["no_prefix"] = (
                prior_prefix,
                torch.zeros(
                    (batch_size, 1), dtype=torch.bool, device=prior_prefix.device
                ),
            )
        if "shuffled_frames" in self.conditions:
            shuffled_values = features.values.clone()
            for row in range(batch_size):
                valid_count = (
                    features.values.shape[1]
                    if features.padding_mask is None
                    else int((~features.padding_mask[row]).sum().item())
                )
                order = torch.randperm(
                    valid_count, generator=self.shuffle_generator
                ).to(shuffled_values.device)
                shuffled_values[row, :valid_count] = features.values[row, order]
            shuffled_features = replace(features, values=shuffled_values)
            shuffled = provider(shuffled_features).semantic
            streams["shuffled_frames"] = (
                shuffled.values, shuffled.padding_mask
            )
        return streams

    @staticmethod
    def _sample_records(batch, texts, batch_results):
        records = []
        sample_ids = batch.get("sample_id", [None] * len(texts))
        domains = batch.get("domain", [None] * len(texts))
        for row, text in enumerate(texts):
            condition_values = {
                name: values[row]
                for name, values in batch_results.items()
            }
            records.append({
                "sample_id": sample_ids[row],
                "domain": domains[row],
                "target_text": text,
                "conditions": condition_values,
            })
        return records

    @staticmethod
    def _paired_gains(records):
        gains = {}
        comparisons = {
            "utterance_specific": "shuffled_prefix",
            "zero_prefix": "zero_prefix",
            "qwen_prior": "no_prefix",
            "temporal_order": "shuffled_frames",
        }
        for label, condition in comparisons.items():
            values = [
                record["conditions"][condition]["loss"]
                - record["conditions"]["correct"]["loss"]
                for record in records
                if condition in record["conditions"]
            ]
            if values:
                gains[label] = {
                    "mean_loss_gain": sum(values) / len(values),
                    "fraction_correct_better": sum(value > 0 for value in values)
                    / len(values),
                }
        return gains


def run_audio_conditioning_evaluation(config: Mapping[str, Any]):
    device = torch.device(config["system"]["device"])
    model = TrainingModelWrapper.load(
        config["models"]["checkpoint"], map_location="cpu", device=device
    )
    qwen_config = config["models"]["language_core"]
    language_core = QwenLanguageCore.from_config(
        qwen_config,
        device=None if qwen_config.get("device_map") is not None else device,
    )
    dataset = build_datasets(config["dataset"])
    loader = create_loader(
        dataset,
        required_modalities=config["dataset"]["requires"],
        batch_size=config["evaluation"]["batch_size"],
    )
    amp_name = config["evaluation"].get("amp_dtype", "bfloat16")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[amp_name]
    evaluator = AudioConditioningEvaluator(
        model=model,
        language_core=language_core,
        loader=loader,
        conditions=config["evaluation"]["conditions"],
        max_batches=config["evaluation"].get("max_batches"),
        prompt=config["evaluation"].get("prompt", ""),
        amp_enabled=bool(config["evaluation"].get("amp_enabled", True)),
        amp_dtype=amp_dtype,
        shuffle_seed=int(config["evaluation"].get("shuffle_seed", 1729)),
    )
    summary, records = evaluator.run()
    return _write_results(config["output"], summary, records)


def _write_results(config, summary, records):
    output_dir = Path(config["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = config.get("prefix", "audio_conditioning")
    summary_path = output_dir / f"{prefix}_{run_id}_summary.json"
    samples_path = output_dir / f"{prefix}_{run_id}_samples.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with samples_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return summary_path, samples_path, summary
