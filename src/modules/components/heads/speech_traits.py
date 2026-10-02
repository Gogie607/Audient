"""Configurable speech-trait heads operating directly on audio features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor, nn

from ..modules import AttentionPool, Float32RMSNorm, LearnedQueryPool


@dataclass(frozen=True)
class SpeechTraitHeadOutput:
    """Trait conditioning representation plus supervised predictions."""

    conditioning: Tensor
    predictions: Mapping[str, Tensor]


class RegressionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)


class SpeechTraitHeads(nn.Module):
    """Predict scalar groups and normalized temporal trait trajectories."""

    def __init__(
        self,
        model_dim: int,
        conditioning_dim: int,
        *,
        scalar_group_dims: Mapping[str, int],
        temporal_channels: int,
        temporal_bins: int = 64,
        num_heads: int = 8,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        invalid_groups = {
            name: size for name, size in scalar_group_dims.items() if size <= 0
        }
        if invalid_groups:
            raise ValueError(f"trait group dimensions must be positive: {invalid_groups}")
        if temporal_channels < 0:
            raise ValueError("temporal_channels must be non-negative")

        resolved_hidden_dim = hidden_dim or model_dim
        self.global_pool = AttentionPool(model_dim, num_heads=num_heads)
        self.scalar_heads = nn.ModuleDict()
        for name, output_dim in scalar_group_dims.items():
            self.scalar_heads[name] = RegressionHead(
                model_dim,
                output_dim,
                hidden_dim=resolved_hidden_dim,
                dropout=dropout,
            )

        self.conditioning_head = nn.Sequential(
            nn.Linear(model_dim, conditioning_dim),
            Float32RMSNorm(conditioning_dim),
        )

        self.temporal_pool: LearnedQueryPool | None = None
        self.temporal_output: nn.Linear | None = None
        if temporal_channels:
            self.temporal_pool = LearnedQueryPool(
                model_dim,
                output_tokens=temporal_bins,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.temporal_output = nn.Linear(model_dim, temporal_channels)

    def forward(
        self,
        values: Tensor,
        *,
        padding_mask: Tensor | None = None,
    ) -> SpeechTraitHeadOutput:
        summary = self.global_pool(values, padding_mask=padding_mask)
        predictions: dict[str, Tensor] = {}
        for name in self.scalar_heads:
            predictions[name] = self.scalar_heads[name](summary)

        if self.temporal_pool is not None and self.temporal_output is not None:
            temporal = self.temporal_pool(values, padding_mask=padding_mask)
            predictions["temporal"] = self.temporal_output(temporal).transpose(1, 2)

        conditioning = self.conditioning_head(summary).unsqueeze(1)
        return SpeechTraitHeadOutput(
            conditioning=conditioning,
            predictions=predictions,
        )
