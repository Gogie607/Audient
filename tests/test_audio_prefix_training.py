"""Tests for the first composable V2 training recipe."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.modules import QwenLanguageCore, TrainingModelWrapper, WhisperProvider
from src.training import (
    AudioPrefixSnapshotCollector,
    create_audio_prefix_training_module,
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        del kwargs
        values = text if isinstance(text, list) else [text]
        sequences = [[3 + (len(value) % 5), 4] for value in values]
        if isinstance(text, list):
            return {
                "input_ids": torch.tensor(sequences, dtype=torch.long),
                "attention_mask": torch.ones(len(sequences), 2, dtype=torch.long),
            }
        return {"input_ids": sequences[0], "attention_mask": [1, 1]}


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(16, 8)
        self.output = nn.Linear(8, 16, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        return SimpleNamespace(logits=self.output(inputs_embeds))


class DummyParams:
    def __init__(self, values=None):
        self.values = (
            {
            "semantic_weight": 1.0,
            "trait_smooth_l1_beta": 1.0,
            "timing_weight": 1.0,
            "energy_weight": 1.0,
            "pitch_weight": 1.0,
            "voice_weight": 1.0,
            "temporal_weight": 1.0,
            }
            if values is None
            else values
        )

    def step(self, step):
        del step

    def has(self, name):
        return name in self.values

    def get(self, name):
        return self.values.get(name)


def make_system():
    provider = WhisperProvider(
        {
            "identifier": "test-whisper",
            "revision": None,
            "output_layer": -1,
            "feature_dimension": 8,
            "frame_rate_hz": 2.0,
        },
        model_dim=8,
        semantic_dim=8,
        temporal_reduction=2,
        semantic_blocks=1,
        num_heads=2,
        ff_hidden_dim=16,
        trait_group_dims={"timing": 2},
        temporal_trait_channels=1,
        temporal_trait_bins=3,
        trait_hidden_dim=8,
    )
    wrapper = TrainingModelWrapper({"audio_provider": provider})
    core = QwenLanguageCore(
        model=FakeQwen(),
        tokenizer=FakeTokenizer(),
        config={"model_name": "test/qwen", "expected_hidden_size": 8},
    )
    return wrapper, core


class AudioPrefixTrainingTests(unittest.TestCase):
    def test_composes_semantic_and_trait_losses_with_one_forward(self):
        wrapper, core = make_system()
        train_mode = wrapper.get_mode()
        eval_mode = wrapper.get_mode()
        train_mode["audio_provider"] = {
            name: True for name in train_mode["audio_provider"]
        }
        module = create_audio_prefix_training_module(
            model=wrapper,
            language_core=core,
            params=DummyParams(),
            train_mode=train_mode,
            eval_mode=eval_mode,
            trait_weight_parameters={
                "timing": "timing_weight",
                "temporal": "temporal_weight",
            },
            collector=AudioPrefixSnapshotCollector(max_samples=2),
        )
        wrapper.set_trainable(
            {"audio_provider": {name: True for name in train_mode["audio_provider"]}}
        )
        batch = {
            "audio_encoding": torch.randn(2, 8, 8),
            "text": ["hello", "world"],
            "sample_id": ["a", "b"],
            "domain": ["read", "conversation"],
            "speech_traits": {
                "duration_s": torch.tensor([3.0, 4.0]),
                "timing": {
                    "scalars": torch.randn(2, 2),
                    "scalar_valid": torch.ones(2, 2, dtype=torch.bool),
                    "temporal": torch.randn(2, 1, 3),
                    "temporal_valid": torch.ones(2, 1, dtype=torch.bool),
                },
            },
        }

        loss, payload = module.compute(batch)
        loss.backward()

        self.assertEqual(set(payload.objective_losses), {
            "semantic_token",
            "speech_trait/timing",
            "speech_trait/temporal",
            "total",
        })
        self.assertIn("semantic/token_loss", payload.metrics)
        self.assertIn("traits/timing_mse", payload.metrics)
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in wrapper.trainable_parameters()
        ))
        self.assertTrue(all(parameter.grad is None for parameter in core.parameters()))

        module.on_validation_begin(step=4)
        module.on_validation_payload(payload)
        report = module.on_validation_end()
        self.assertIn("semantic/perplexity", report)
        self.assertEqual(len(module.collector.snapshots), 2)

    def test_semantic_only_recipe_does_not_require_trait_targets(self):
        wrapper, core = make_system()
        module = create_audio_prefix_training_module(
            model=wrapper,
            language_core=core,
            params=DummyParams(),
            train_mode=wrapper.get_mode(),
            eval_mode=wrapper.get_mode(),
            include_traits=False,
        )
        loss, payload = module.compute({
            "audio_encoding": torch.randn(1, 6, 8),
            "duration_s": [3.0],
            "txt": ["test"],
        })
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(any(
            name.startswith("speech_trait/")
            for name in payload.objective_losses
        ))

    def test_rejects_missing_trait_group(self):
        wrapper, core = make_system()
        module = create_audio_prefix_training_module(
            model=wrapper,
            language_core=core,
            params=DummyParams(),
            train_mode=wrapper.get_mode(),
            eval_mode=wrapper.get_mode(),
            trait_weight_parameters={
                "timing": "timing_weight",
                "temporal": "temporal_weight",
            },
        )
        with self.assertRaisesRegex(KeyError, "temporal"):
            module.compute({
                "audio_encoding": torch.randn(1, 6, 8),
                "duration_s": [3.0],
                "text": ["test"],
                "speech_traits": {"timing": torch.randn(1, 2)},
            })

    def test_objective_weights_come_from_parameter_wrapper(self):
        wrapper, core = make_system()
        params = DummyParams({"semantic_weight": 0.0})
        module = create_audio_prefix_training_module(
            model=wrapper,
            language_core=core,
            params=params,
            train_mode=wrapper.get_mode(),
            eval_mode=wrapper.get_mode(),
            include_traits=False,
        )
        loss, payload = module.compute({
            "audio_encoding": torch.randn(1, 6, 8),
            "duration_s": [3.0],
            "text": ["test"],
        })
        self.assertEqual(loss.item(), 0.0)
        self.assertGreater(payload.objective_losses["semantic_token"], 0.0)

    def test_missing_objective_weight_fails_loudly(self):
        wrapper, core = make_system()
        module = create_audio_prefix_training_module(
            model=wrapper,
            language_core=core,
            params=DummyParams({}),
            train_mode=wrapper.get_mode(),
            eval_mode=wrapper.get_mode(),
            include_traits=False,
        )
        with self.assertRaisesRegex(KeyError, "semantic_weight"):
            module.compute({
                "audio_encoding": torch.randn(1, 6, 8),
                "duration_s": [3.0],
                "text": ["test"],
            })

    def test_position_weighted_contrast_reports_region_metrics(self):
        wrapper, core = make_system()
        params = DummyParams({
            "semantic_weight": 1.0,
            "semantic_contrast_weight": 0.25,
            "semantic_contrast_margin": 0.5,
            "semantic_first_weight": 4.0,
            "semantic_early_weight": 2.0,
            "semantic_middle_weight": 1.0,
            "semantic_remaining_weight": 1.0,
            "contrast_first_weight": 4.0,
            "contrast_early_weight": 2.0,
            "contrast_middle_weight": 1.0,
        })
        module = create_audio_prefix_training_module(
            model=wrapper,
            language_core=core,
            params=params,
            train_mode=wrapper.get_mode(),
            eval_mode=wrapper.get_mode(),
            include_traits=False,
            semantic_objective="position_weighted",
            enable_audio_contrast=True,
        )

        loss, payload = module.compute({
            "audio_encoding": torch.randn(2, 6, 8),
            "duration_s": [3.0, 2.5],
            "text": ["first", "different"],
        })

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(payload.objective_losses), {
            "semantic_position_weighted",
            "semantic_audio_contrast",
            "total",
        })
        self.assertIn("semantic/first_loss", payload.metrics)
        self.assertIn("semantic/early_accuracy", payload.metrics)
        self.assertIn(
            "semantic/contrast_first_logp_advantage", payload.metrics
        )
        self.assertIn("semantic/token_loss", payload.metrics)

    def test_contrast_rejects_single_sample_batch(self):
        wrapper, core = make_system()
        params = DummyParams({
            "semantic_weight": 1.0,
            "semantic_contrast_weight": 0.25,
            "semantic_contrast_margin": 0.5,
            "semantic_first_weight": 4.0,
            "semantic_early_weight": 2.0,
            "semantic_middle_weight": 1.0,
            "semantic_remaining_weight": 1.0,
            "contrast_first_weight": 4.0,
            "contrast_early_weight": 2.0,
            "contrast_middle_weight": 1.0,
        })
        module = create_audio_prefix_training_module(
            model=wrapper,
            language_core=core,
            params=params,
            train_mode=wrapper.get_mode(),
            eval_mode=wrapper.get_mode(),
            include_traits=False,
            semantic_objective="position_weighted",
            enable_audio_contrast=True,
        )

        with self.assertRaisesRegex(ValueError, "batch_size >= 2"):
            module.compute({
                "audio_encoding": torch.randn(1, 6, 8),
                "duration_s": [3.0],
                "text": ["test"],
            })


if __name__ == "__main__":
    unittest.main()
