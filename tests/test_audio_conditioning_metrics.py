"""Tests for paired audio-conditioning token metrics."""

from __future__ import annotations

import unittest

import torch

from src.evaluation.audio_conditioning import ConditionMetrics


class ConditionMetricsTests(unittest.TestCase):
    def test_reports_full_and_early_token_windows(self):
        labels = torch.tensor([[-100, -100, 1, 2, 3]])
        logits = torch.zeros(1, 5, 6)
        logits[0, 1, 1] = 8.0
        logits[0, 2, 2] = 8.0
        logits[0, 3, 3] = 8.0
        metric = ConditionMetrics()

        records = metric.update(logits, labels)
        summary = metric.summarize()

        self.assertEqual(records[0]["token_count"], 3)
        self.assertAlmostEqual(summary["accuracy"], 1.0)
        self.assertAlmostEqual(summary["top5_accuracy"], 1.0)
        self.assertIn("first_1_loss", summary)
        self.assertIn("first_16_loss", summary)

    def test_aggregation_is_weighted_by_token_count(self):
        metric = ConditionMetrics()
        logits = torch.zeros(2, 4, 4)
        labels = torch.tensor([
            [-100, 1, -100, -100],
            [-100, 1, 2, 3],
        ])

        metric.update(logits, labels)
        summary = metric.summarize()

        self.assertAlmostEqual(summary["loss"], torch.log(torch.tensor(4.0)).item())


if __name__ == "__main__":
    unittest.main()
