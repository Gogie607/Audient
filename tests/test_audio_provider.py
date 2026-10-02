"""Tests for the architecture-level audio-provider contract."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.modules.audio import (
    AudioBackbone,
    AudioBackboneFeatures,
    AudioBackboneSpec,
    AudioProvider,
    AudioProviderCapabilities,
    AudioRepresentation,
    AudioStream,
    AudioTiming,
    RepresentationKind,
)


EXAMPLE_BACKBONE_SPEC = AudioBackboneSpec(
    identifier="example/audio-backbone",
    revision="test-revision",
    output_layer=-1,
    feature_dimension=4,
    frame_rate_hz=50.0,
)


class ExampleAudioBackbone(AudioBackbone):
    def __init__(self) -> None:
        super().__init__(nn.Linear(1, 4), EXAMPLE_BACKBONE_SPEC)

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        sample_rate: int,
        padding_mask: torch.Tensor | None = None,
    ) -> AudioBackboneFeatures:
        return AudioBackboneFeatures(
            values=self.model(waveform.unsqueeze(-1)),
            backbone=self.spec,
            timing=AudioTiming(
                sample_rate=sample_rate,
                input_sample_offset=0,
                input_sample_count=waveform.shape[1],
                output_frame_offset=0,
                output_frame_count=waveform.shape[1],
            ),
            padding_mask=padding_mask,
        )


class ExampleAudioProvider(AudioProvider):
    component_type = "test_audio_provider"

    def __init__(self, backbone_spec: dict[str, object] | None = None) -> None:
        resolved_spec = (
            AudioBackboneSpec.from_config(backbone_spec)
            if backbone_spec is not None
            else EXAMPLE_BACKBONE_SPEC
        )
        super().__init__(
            resolved_spec,
            trainable_modules={"semantic_projection": nn.Linear(4, 8)},
        )
        self.initialize_frozen()

    @property
    def capabilities(self) -> AudioProviderCapabilities:
        return AudioProviderCapabilities()

    def forward_features(
        self,
        features: AudioBackboneFeatures,
    ) -> AudioRepresentation:
        values = self.provider_modules["semantic_projection"](features.values)
        frames = values.shape[1]
        return AudioRepresentation(
            semantic=AudioStream(
                values=values,
                kind=RepresentationKind.CONTINUOUS,
                frame_rate_hz=features.backbone.frame_rate_hz,
                padding_mask=features.padding_mask,
            ),
            timing=AudioTiming(
                sample_rate=features.timing.sample_rate,
                input_sample_offset=features.timing.input_sample_offset,
                input_sample_count=features.timing.input_sample_count,
                output_frame_offset=features.timing.output_frame_offset,
                output_frame_count=frames,
            ),
        )

    def get_config(self) -> dict[str, object]:
        return {"backbone_spec": self.backbone_spec.to_config()}


class AudioProviderTests(unittest.TestCase):
    def test_external_backbone_remains_frozen(self) -> None:
        backbone = ExampleAudioBackbone()
        backbone.train()

        self.assertFalse(backbone.training)
        self.assertFalse(backbone.model.training)
        self.assertTrue(all(not p.requires_grad for p in backbone.parameters()))

    def test_provider_state_contains_no_backbone_parameters(self) -> None:
        provider = ExampleAudioProvider()
        self.assertEqual(
            set(provider.state_dict()),
            {"provider_modules.semantic_projection.weight", "provider_modules.semantic_projection.bias"},
        )
        serialized = provider.serialize()
        self.assertEqual(
            serialized["config"]["backbone_spec"],
            EXAMPLE_BACKBONE_SPEC.to_config(),
        )

    def test_trainability_controls_only_provider_modules(self) -> None:
        provider = ExampleAudioProvider()
        self.assertEqual(
            provider.get_trainable(),
            {"semantic_projection": False},
        )
        self.assertEqual(list(provider.trainable_parameters()), [])

        provider.set_trainable({"semantic_projection": True})
        self.assertEqual(
            provider.get_trainable(),
            {"semantic_projection": True},
        )
        self.assertTrue(
            all(
                p.requires_grad
                for p in provider.provider_modules["semantic_projection"].parameters()
            )
        )

    def test_composite_mode_state_round_trips(self) -> None:
        provider = ExampleAudioProvider()
        provider.set_mode({"semantic_projection": False})
        state = provider.get_mode()

        provider.set_mode(True)
        provider.set_mode(state)

        self.assertEqual(
            provider.get_mode(),
            {"semantic_projection": False},
        )

    def test_call_runs_complete_provider_prefix(self) -> None:
        provider = ExampleAudioProvider()
        backbone = ExampleAudioBackbone()
        waveform = torch.randn(2, 16)
        padding_mask = torch.zeros(2, 16, dtype=torch.bool)

        features = backbone(
            waveform,
            sample_rate=16_000,
            padding_mask=padding_mask,
        )
        representation = provider(features)

        self.assertEqual(representation.semantic.values.shape, (2, 16, 8))
        self.assertEqual(representation.semantic.kind, RepresentationKind.CONTINUOUS)
        self.assertEqual(representation.timing.output_frame_count, 16)

    def test_training_can_enter_at_backbone_feature_boundary(self) -> None:
        provider = ExampleAudioProvider()
        cached_features = AudioBackboneFeatures(
            values=torch.randn(2, 10, 4),
            backbone=EXAMPLE_BACKBONE_SPEC,
            timing=AudioTiming(
                sample_rate=16_000,
                input_sample_offset=0,
                input_sample_count=3_200,
                output_frame_offset=0,
                output_frame_count=10,
            ),
            padding_mask=torch.zeros(2, 10, dtype=torch.bool),
        )

        representation = provider(cached_features)

        self.assertEqual(representation.semantic.values.shape, (2, 10, 8))
        self.assertEqual(representation.semantic.frame_rate_hz, 50.0)

    def test_provider_rejects_features_from_another_backbone(self) -> None:
        provider = ExampleAudioProvider()
        incompatible_spec = AudioBackboneSpec(
            identifier="other/backbone",
            revision=None,
            output_layer=None,
            feature_dimension=4,
            frame_rate_hz=50.0,
        )
        features = AudioBackboneFeatures(
            values=torch.randn(1, 2, 4),
            backbone=incompatible_spec,
            timing=AudioTiming(
                sample_rate=16_000,
                input_sample_offset=0,
                input_sample_count=640,
                output_frame_offset=0,
                output_frame_count=2,
            ),
        )

        with self.assertRaises(ValueError):
            provider(features)

    def test_non_streaming_provider_rejects_stream_creation(self) -> None:
        with self.assertRaises(RuntimeError):
            ExampleAudioProvider().create_stream_state()

    def test_representation_rejects_inconsistent_frame_count(self) -> None:
        stream = AudioStream(
            values=torch.randn(1, 3, 4),
            kind=RepresentationKind.CONTINUOUS,
            frame_rate_hz=50.0,
        )
        with self.assertRaises(ValueError):
            AudioRepresentation(
                semantic=stream,
                timing=AudioTiming(
                    sample_rate=16_000,
                    input_sample_offset=0,
                    input_sample_count=960,
                    output_frame_offset=0,
                    output_frame_count=2,
                ),
            )


if __name__ == "__main__":
    unittest.main()
