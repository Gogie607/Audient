"""One shared forward pass for semantic and speech-trait objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from runweaver_ml.phase_control import TrainingContext

from ...modules import AudioBackboneFeatures, AudioTiming, QwenLanguageCore
from .payloads import AudioPrefixPayload, TeacherForcedInputs


class AudioPrefixForward:
    """Adapt a decoded DSM batch to the provider and frozen language core.

    Dataset routing stays outside this object. The batch is expected to contain
    pregenerated ``audio_encoding`` values, target ``text`` (or ``txt``), and
    optional ``duration_s`` and ``speech_traits`` values.
    """

    def __init__(
        self,
        language_core: QwenLanguageCore,
        *,
        provider_name: str = "audio_provider",
        prompt: str = "",
        sample_rate: int = 16_000,
        temporal_trait_groups: tuple[str, ...] = ("timing", "energy", "pitch"),
        include_shuffled_semantic: bool = False,
    ) -> None:
        self.language_core = language_core
        self.provider_name = provider_name
        self.prompt = prompt
        self.sample_rate = sample_rate
        self.temporal_trait_groups = temporal_trait_groups
        self.include_shuffled_semantic = include_shuffled_semantic

    def __call__(self, context: TrainingContext, model) -> TrainingContext:
        batch = context.batch
        if not isinstance(batch, Mapping):
            raise TypeError("audio-prefix training expects a mapping batch")

        provider = model.get(self.provider_name)
        features, trait_payload = self.prepare_features(batch, provider)
        encodings = features.values
        representation = provider(features)

        texts = self.text_values(batch, encodings.shape[0])
        language_inputs = self.build_language_inputs(
            representation.semantic.values,
            representation.semantic.padding_mask,
            texts,
        )
        output = self.language_core(
            inputs_embeds=language_inputs.inputs_embeds,
            attention_mask=language_inputs.attention_mask,
        )

        shuffled_logits = None
        if self.include_shuffled_semantic:
            if encodings.shape[0] < 2:
                raise ValueError(
                    "matched-vs-shuffled semantic training requires batch_size >= 2"
                )
            order = torch.roll(
                torch.arange(encodings.shape[0], device=representation.semantic.values.device),
                1,
            )
            shuffled_values = representation.semantic.values.detach().index_select(
                0, order
            )
            shuffled_mask = representation.semantic.padding_mask
            if shuffled_mask is not None:
                shuffled_mask = shuffled_mask.detach().index_select(
                    0, order.to(shuffled_mask.device)
                )
            shuffled_inputs = self.build_language_inputs(
                shuffled_values, shuffled_mask, texts
            )
            if not torch.equal(shuffled_inputs.labels, language_inputs.labels):
                raise RuntimeError("shuffled branch changed transcript labels")
            with torch.no_grad():
                shuffled_output = self.language_core(
                    inputs_embeds=shuffled_inputs.inputs_embeds,
                    attention_mask=shuffled_inputs.attention_mask,
                )
            shuffled_logits = shuffled_output.logits

        payload = AudioPrefixPayload(
            batch_size=encodings.shape[0],
            sample_ids=self._metadata_values(batch, "sample_id", encodings.shape[0]),
            domains=self._metadata_values(batch, "domain", encodings.shape[0]),
            target_text=tuple(texts),
        )
        context.put("representation", representation)
        context.put("logits", output.logits)
        context.put("labels", language_inputs.labels)
        if shuffled_logits is not None:
            context.put("shuffled_logits", shuffled_logits)
        context.put(
            "speech_trait_targets",
            self._normalize_trait_targets(trait_payload),
        )
        context.put("payload", payload)
        return context

    def prepare_features(self, batch: Mapping[str, Any], provider):
        """Create the provider-boundary payload from one decoded DSM batch."""
        encodings = self._required_tensor(batch, "audio_encoding")
        if encodings.ndim != 3:
            raise ValueError("audio_encoding must have shape [batch, frames, dim]")
        encodings = encodings.to(provider.device)

        trait_payload = batch.get("speech_traits", {})
        duration_value = batch.get("duration_s")
        if duration_value is None and isinstance(trait_payload, Mapping):
            duration_value = trait_payload.get("duration_s")
        padding_mask = self._padding_mask(
            duration_value,
            batch_size=encodings.shape[0],
            frame_count=encodings.shape[1],
            frame_rate_hz=provider.backbone_spec.frame_rate_hz,
            device=encodings.device,
        )
        duration_s = self._duration_tensor(
            duration_value, encodings.shape[0], encodings.device
        )
        input_samples = int(math.ceil(float(duration_s.max().item()) * self.sample_rate))
        features = AudioBackboneFeatures(
            values=encodings,
            backbone=provider.backbone_spec,
            padding_mask=padding_mask,
            timing=AudioTiming(
                sample_rate=self.sample_rate,
                input_sample_offset=0,
                input_sample_count=input_samples,
                output_frame_offset=0,
                output_frame_count=encodings.shape[1],
            ),
        )
        return features, trait_payload

    def text_values(self, batch: Mapping[str, Any], batch_size: int) -> list[str]:
        return self._text_values(batch, batch_size)

    def build_language_inputs(
        self,
        semantic_values: Tensor,
        semantic_padding_mask: Tensor | None,
        texts: list[str],
    ) -> TeacherForcedInputs:
        target_ids, target_mask = self._tokenize_targets(texts)
        target_ids = target_ids.to(self.language_core.device)
        target_mask = target_mask.to(self.language_core.device)

        semantic = semantic_values.to(
            device=self.language_core.device,
            dtype=self.language_core.dtype,
        )
        semantic_mask = semantic_padding_mask
        if semantic_mask is None:
            semantic_valid = torch.ones(
                semantic.shape[:2], dtype=torch.bool, device=semantic.device
            )
        else:
            semantic_valid = ~semantic_mask.to(semantic.device)

        embeddings = [semantic]
        masks = [semantic_valid]
        label_parts = [
            torch.full(
                semantic.shape[:2], -100, dtype=torch.long, device=semantic.device
            )
        ]
        if self.prompt:
            prompt_ids, prompt_mask = self._tokenize_prompt(
                self.prompt, semantic.shape[0]
            )
            prompt_ids = prompt_ids.to(self.language_core.device)
            prompt_mask = prompt_mask.to(self.language_core.device).bool()
            embeddings.insert(0, self.language_core.embed_tokens(prompt_ids))
            masks.insert(0, prompt_mask)
            label_parts.insert(0, torch.full_like(prompt_ids, -100))

        embeddings.append(self.language_core.embed_tokens(target_ids))
        masks.append(target_mask.bool())
        label_parts.append(target_ids.masked_fill(~target_mask.bool(), -100))
        inputs_embeds = torch.cat(embeddings, dim=1)
        attention_mask = torch.cat(masks, dim=1).long()
        labels = torch.cat(label_parts, dim=1)
        return TeacherForcedInputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _tokenize_targets(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        sequences: list[list[int]] = []
        eos_id = self.language_core.tokenizer.eos_token_id
        pad_id = self.language_core.tokenizer.pad_token_id
        if eos_id is None:
            raise ValueError("language-core tokenizer must define eos_token_id")
        if pad_id is None:
            pad_id = eos_id
        for text in texts:
            encoded = self.language_core.tokenize(text, add_special_tokens=True)
            ids = encoded["input_ids"]
            if isinstance(ids, Tensor):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            sequence = list(ids)
            if not sequence or sequence[-1] != eos_id:
                sequence.append(eos_id)
            sequences.append(sequence)
        width = max(len(sequence) for sequence in sequences)
        ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            mask[row, : len(sequence)] = True
        return ids, mask

    def _tokenize_prompt(self, prompt: str, batch_size: int) -> tuple[Tensor, Tensor]:
        encoded = self.language_core.tokenize(
            [prompt] * batch_size,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    @staticmethod
    def _required_tensor(batch: Mapping[str, Any], name: str) -> Tensor:
        value = batch.get(name)
        if not isinstance(value, Tensor):
            raise TypeError(f"batch[{name!r}] must be a tensor")
        return value

    @staticmethod
    def _text_values(batch: Mapping[str, Any], batch_size: int) -> list[str]:
        value = batch.get("text", batch.get("txt"))
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, Sequence):
            values = [str(item) for item in value]
        else:
            raise TypeError("batch must contain text or txt strings")
        if len(values) != batch_size:
            raise ValueError("text target count must match audio batch size")
        return values

    @staticmethod
    def _duration_tensor(value: Any, batch_size: int, device) -> Tensor:
        if value is None:
            return torch.full((batch_size,), 30.0, device=device)
        duration = torch.as_tensor(value, dtype=torch.float32, device=device).flatten()
        if duration.numel() != batch_size:
            raise ValueError("duration_s must contain one value per sample")
        if torch.any(duration <= 0):
            raise ValueError("duration_s values must be positive")
        return duration

    @classmethod
    def _padding_mask(
        cls, value, *, batch_size, frame_count, frame_rate_hz, device
    ) -> Tensor:
        duration = cls._duration_tensor(value, batch_size, device)
        valid_frames = torch.ceil(duration * frame_rate_hz).long().clamp(1, frame_count)
        frames = torch.arange(frame_count, device=device).unsqueeze(0)
        return frames >= valid_frames.unsqueeze(1)

    @staticmethod
    def _metadata_values(batch, name: str, batch_size: int) -> tuple[str, ...]:
        value = batch.get(name)
        if value is None:
            return ()
        if isinstance(value, str):
            values = (value,)
        else:
            values = tuple(str(item) for item in value)
        if len(values) != batch_size:
            raise ValueError(f"{name} count must match batch size")
        return values

    def _normalize_trait_targets(self, traits: Any) -> dict[str, Any]:
        """Convert the persisted speech-trait schema to objective groups."""
        if not isinstance(traits, Mapping):
            return traits

        normalized: dict[str, Any] = {}
        for name, group in traits.items():
            if not isinstance(group, Mapping):
                if name not in {
                    "version", "sample_rate", "duration_s", "speaker_id", "language_id"
                }:
                    normalized[name] = group
                continue
            if "scalars" in group:
                normalized[name] = {
                    "values": group["scalars"],
                    "valid_mask": group.get("scalar_valid"),
                }
        temporal_values = []
        temporal_valid = []
        for name in self.temporal_trait_groups:
            group = traits.get(name)
            if isinstance(group, Mapping) and "temporal" in group:
                temporal_value = torch.as_tensor(group["temporal"])
                temporal_values.append(temporal_value)
                valid = group.get("temporal_valid")
                if valid is None:
                    valid = torch.ones(
                        temporal_value.shape[:-1], dtype=torch.bool
                    )
                temporal_valid.append(torch.as_tensor(valid, dtype=torch.bool))

        if temporal_values:
            normalized["temporal"] = {
                "values": torch.cat(temporal_values, dim=1),
                "valid_mask": torch.cat(temporal_valid, dim=1),
            }
        return normalized
