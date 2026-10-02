"""Whisper-feature provider with semantic and speech-trait heads."""

from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn

from ...audio import (
    AudioBackboneFeatures,
    AudioBackboneSpec,
    AudioProvider,
    AudioProviderCapabilities,
    AudioRepresentation,
    AudioStream,
    AudioTiming,
    RepresentationKind,
)
from ..heads import SemanticPrefixHead, SpeechTraitHeads


class WhisperInputStem(nn.Module):
    """Normalize Whisper states and optionally change their working width."""

    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = (
            nn.Identity()
            if input_dim == model_dim
            else nn.Linear(input_dim, model_dim)
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, values: Tensor, *, padding_mask: Tensor | None) -> Tensor:
        values = self.output_norm(self.input_projection(self.input_norm(values)))
        if padding_mask is not None:
            values = values.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return values


class WhisperProvider(AudioProvider):
    """Checkpointed heads consuming frozen Whisper encoder features."""

    component_type = "whisper_provider"

    def __init__(
        self,
        backbone_spec: AudioBackboneSpec | Mapping[str, Any],
        *,
        model_dim: int = 1024,
        semantic_dim: int = 2048,
        temporal_reduction: int = 4,
        semantic_blocks: int = 2,
        num_heads: int = 8,
        ff_hidden_dim: int | None = None,
        trait_group_dims: Mapping[str, int] | None = None,
        temporal_trait_channels: int = 0,
        temporal_trait_bins: int = 64,
        trait_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        resolved_spec = (
            backbone_spec
            if isinstance(backbone_spec, AudioBackboneSpec)
            else AudioBackboneSpec.from_config(backbone_spec)
        )
        resolved_trait_dims = dict(trait_group_dims or {})
        self._config = {
            "backbone_spec": resolved_spec.to_config(),
            "model_dim": model_dim,
            "semantic_dim": semantic_dim,
            "temporal_reduction": temporal_reduction,
            "semantic_blocks": semantic_blocks,
            "num_heads": num_heads,
            "ff_hidden_dim": ff_hidden_dim,
            "trait_group_dims": resolved_trait_dims,
            "temporal_trait_channels": temporal_trait_channels,
            "temporal_trait_bins": temporal_trait_bins,
            "trait_hidden_dim": trait_hidden_dim,
            "dropout": dropout,
        }

        modules = {
            "input_stem": WhisperInputStem(resolved_spec.feature_dimension, model_dim),
            "semantic_head": SemanticPrefixHead(
                model_dim,
                semantic_dim,
                reduction=temporal_reduction,
                num_blocks=semantic_blocks,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
            ),
            "speech_trait_heads": SpeechTraitHeads(
                model_dim,
                semantic_dim,
                scalar_group_dims=resolved_trait_dims,
                temporal_channels=temporal_trait_channels,
                temporal_bins=temporal_trait_bins,
                num_heads=num_heads,
                hidden_dim=trait_hidden_dim,
                dropout=dropout,
            ),
        }
        super().__init__(resolved_spec, trainable_modules=modules)
        self.temporal_reduction = temporal_reduction
        self.initialize_frozen()

    @property
    def capabilities(self) -> AudioProviderCapabilities:
        trait_names = list(self._config["trait_group_dims"])
        if self._config["temporal_trait_channels"]:
            trait_names.append("temporal")
        return AudioProviderCapabilities(
            supports_streaming=False,
            emitted_trait_names=tuple(trait_names),
            semantic_kind=RepresentationKind.CONTINUOUS,
        )

    def forward_features(
        self,
        features: AudioBackboneFeatures,
    ) -> AudioRepresentation:
        stem = self.provider_modules["input_stem"](
            features.values,
            padding_mask=features.padding_mask,
        )
        semantic_values, semantic_padding_mask = self.provider_modules[
            "semantic_head"
        ](
            stem,
            padding_mask=features.padding_mask,
        )
        trait_output = self.provider_modules["speech_trait_heads"](
            stem,
            padding_mask=features.padding_mask,
        )

        output_frames = semantic_values.shape[1]
        semantic_frame_rate = (
            features.backbone.frame_rate_hz / self.temporal_reduction
        )
        return AudioRepresentation(
            semantic=AudioStream(
                values=semantic_values,
                kind=RepresentationKind.CONTINUOUS,
                frame_rate_hz=semantic_frame_rate,
                padding_mask=semantic_padding_mask,
            ),
            timing=AudioTiming(
                sample_rate=features.timing.sample_rate,
                input_sample_offset=features.timing.input_sample_offset,
                input_sample_count=features.timing.input_sample_count,
                output_frame_offset=(
                    features.timing.output_frame_offset // self.temporal_reduction
                ),
                output_frame_count=output_frames,
            ),
            trait_conditioning=AudioStream(
                values=trait_output.conditioning,
                kind=RepresentationKind.TRAITS,
                frame_rate_hz=1.0,
            ),
            trait_predictions=trait_output.predictions,
        )

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)
