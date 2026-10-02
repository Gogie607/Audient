"""Independent validation metrics for audio-prefix training."""

from __future__ import annotations

import math

from runweaver_ml.phase_control import MetricHandlerBase


class AudioPrefixMetrics(MetricHandlerBase):
    def reset(self):
        self._totals: dict[str, float] = {}
        self._weights: dict[str, int] = {}

    def update(self, payload):
        for name, value in payload.metrics.items():
            self._totals[name] = self._totals.get(name, 0.0) + value.total
            self._weights[name] = self._weights.get(name, 0) + value.weight

    def summarize(self) -> dict[str, float]:
        summary = {
            name: total / self._weights[name]
            for name, total in self._totals.items()
            if self._weights[name]
        }
        token_loss = summary.get("semantic/token_loss")
        if token_loss is not None:
            summary["semantic/perplexity"] = math.exp(min(token_loss, 80.0))
        return summary

    def report(self, summary=None) -> str:
        values = summary if summary is not None else self.summarize()
        return " | ".join(f"{name}={value:.6f}" for name, value in values.items())

