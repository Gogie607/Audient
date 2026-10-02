"""Paired token metrics for audio-conditioning attribution."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


WINDOWS = (1, 4, 8, 16)


class ConditionMetrics:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, logits, labels) -> list[dict[str, float]]:
        shifted_logits = logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:]
        records = []
        for row in range(shifted_labels.shape[0]):
            valid = shifted_labels[row].ne(-100)
            row_logits = shifted_logits[row][valid]
            row_labels = shifted_labels[row][valid]
            if not row_labels.numel():
                raise ValueError("evaluation sample contains no target tokens")
            losses = functional.cross_entropy(
                row_logits, row_labels, reduction="none"
            )
            predictions = row_logits.argmax(dim=-1)
            top5 = row_logits.topk(min(5, row_logits.shape[-1]), dim=-1).indices
            correct = predictions.eq(row_labels)
            top5_correct = top5.eq(row_labels.unsqueeze(-1)).any(dim=-1)
            record = {
                "token_count": int(row_labels.numel()),
                "loss": float(losses.mean().item()),
                "accuracy": float(correct.float().mean().item()),
                "top5_accuracy": float(top5_correct.float().mean().item()),
            }
            self._add("loss", float(losses.sum().item()), row_labels.numel())
            self._add("accuracy", float(correct.sum().item()), row_labels.numel())
            self._add(
                "top5_accuracy", float(top5_correct.sum().item()), row_labels.numel()
            )
            for width in WINDOWS:
                count = min(width, row_labels.numel())
                value = float(losses[:count].sum().item())
                self._add(f"first_{width}_loss", value, count)
                record[f"first_{width}_loss"] = value / count
            records.append(record)
        return records

    def summarize(self) -> dict[str, float]:
        summary = {
            name: total / self.counts[name]
            for name, total in self.totals.items()
        }
        summary["perplexity"] = math.exp(min(summary["loss"], 80.0))
        return summary

    def _add(self, name: str, total: float, count: int) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + total
        self.counts[name] = self.counts.get(name, 0) + int(count)

