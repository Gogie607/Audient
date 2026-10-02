"""Tests for the composite Whisper semantic and speech-trait provider."""

from __future__ import annotations

import unittest

import torch

from src.modules import WhisperProvider
from src.modules.audio import AudioBackboneFeatures, AudioBackboneSpec, AudioTiming
from src.modules.components.modules import MaskedTemporalReducer


WHISPER_TEST_SPEC = AudioBackboneSpec(
    identifier="test/whisper",
    revision="local-test",
    output_layer=-1,
    feature_dimension=8,
    frame_rate_hz=50.0,
)


def make_provider() -> WhisperProvider:
    return WhisperProvider(
        WHISPER_TEST_SPEC,
        model_dim=8,
        semantic_dim=16,
        temporal_reduction=4,
        semantic_blocks=2,
        num_heads=2,
        ff_hidden_dim=24,
        trait_group_dims={"timing": 3, "pitch": 2},
        temporal_trait_channels=2,
        temporal_trait_bins=4,
        trait_hidden_dim=12,
    )


def make_features() -> AudioBackboneFeatures:
    padding_mask = torch.tensor(
        [
            [False, False, False, False, False, False, False, False, False],
            [False, False, False, False, False, True, True, True, True],
        ]
    )
    return AudioBackboneFeatures(
        values=torch.randn(2, 9, 8),
        backbone=WHISPER_TEST_SPEC,
        timing=AudioTiming(
            sample_rate=16_000,
            input_sample_offset=0,
            input_sample_count=2_880,
            output_frame_offset=0,
            output_frame_count=9,
        ),
        padding_mask=padding_mask,
    )


class WhisperProviderTests(unittest.TestCase):
    def test_provider_initializes_frozen_and_in_eval_mode(self) -> None:
        provider = make_provider()

        self.assertFalse(provider.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in provider.parameters()))
        self.assertTrue(all(not state for state in provider.get_mode().values()))
        self.assertTrue(all(not state for state in provider.get_trainable().values()))

    def test_composite_outputs_semantics_and_traits(self) -> None:
        output = make_provider()(make_features())

        self.assertEqual(output.semantic.values.shape, (2, 3, 16))
        self.assertEqual(output.semantic.frame_rate_hz, 12.5)
        self.assertEqual(
            output.semantic.padding_mask.tolist(),
            [[False, False, False], [False, False, True]],
        )
        self.assertEqual(output.trait_conditioning.values.shape, (2, 1, 16))
        self.assertEqual(output.trait_predictions["timing"].shape, (2, 3))
        self.assertEqual(output.trait_predictions["pitch"].shape, (2, 2))
        self.assertEqual(output.trait_predictions["temporal"].shape, (2, 2, 4))

    def test_provider_checkpoint_contains_only_composite_modules(self) -> None:
        provider = make_provider()
        state_keys = tuple(provider.state_dict())

        self.assertTrue(state_keys)
        self.assertTrue(all(key.startswith("provider_modules.") for key in state_keys))
        self.assertTrue(all("backbone" not in key for key in state_keys))

        restored = WhisperProvider.from_serialized(provider.serialize())
        self.assertEqual(restored.backbone_spec, WHISPER_TEST_SPEC)
        self.assertEqual(set(restored.state_dict()), set(provider.state_dict()))

    def test_padded_feature_values_do_not_change_outputs(self) -> None:
        provider = make_provider().eval()
        features = make_features()
        changed_values = features.values.clone()
        changed_values[features.padding_mask] = 10_000.0
        changed_features = AudioBackboneFeatures(
            values=changed_values,
            backbone=features.backbone,
            timing=features.timing,
            padding_mask=features.padding_mask,
        )

        with torch.no_grad():
            original = provider(features)
            changed = provider(changed_features)

        torch.testing.assert_close(
            original.semantic.values,
            changed.semantic.values,
        )
        torch.testing.assert_close(
            original.trait_conditioning.values,
            changed.trait_conditioning.values,
        )

    def test_reducer_rejects_fully_padded_sample(self) -> None:
        reducer = MaskedTemporalReducer(8, reduction=4)
        with self.assertRaises(ValueError):
            reducer(
                torch.randn(1, 8, 8),
                padding_mask=torch.ones(1, 8, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
