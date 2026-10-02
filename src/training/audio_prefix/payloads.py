"""Data exchanged by the first audio-prefix training recipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from torch import Tensor


@dataclass(frozen=True)
class MetricValue:
    """An additive metric numerator and its aggregation weight."""

    total: float
    weight: int


@dataclass
class AudioPrefixPayload:
    """Small, detached result returned to RunWeaver handlers."""

    batch_size: int
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    objective_losses: dict[str, float] = field(default_factory=dict)
    sample_ids: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    target_text: tuple[str, ...] = ()
    predicted_token_ids: Tensor | None = None
    target_token_ids: Tensor | None = None
    target_token_mask: Tensor | None = None
    snapshot_values: Mapping[str, Any] = field(default_factory=dict)

    def report(self) -> str:
        fields = [
            f"{name}={value:.6f}"
            for name, value in self.objective_losses.items()
        ]
        return " | ".join(fields)


@dataclass(frozen=True)
class TeacherForcedInputs:
    inputs_embeds: Tensor
    attention_mask: Tensor
    labels: Tensor
