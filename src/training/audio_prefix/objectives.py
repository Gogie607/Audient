"""Composable losses for the first Whisper-to-Qwen experiment."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor

from runweaver_ml.phase_control import TrainingObjective

from .payloads import MetricValue


class WeightedObjective(TrainingObjective):
    def __init__(self, name: str, *, weight_parameter: str):
        if not weight_parameter:
            raise ValueError("weight_parameter must be a non-empty string")
        self.name = name
        self.weight_parameter = weight_parameter
        self.last_loss = 0.0

    def resolved_weight(self, params) -> float:
        if params is None or not params.has(self.weight_parameter):
            raise KeyError(
                f"missing required objective weight {self.weight_parameter!r}"
            )
        weight = float(params.get(self.weight_parameter))
        if weight < 0:
            raise ValueError(
                f"objective weight {self.weight_parameter!r} must be non-negative"
            )
        return weight

    def report(self):
        return f"{self.name}={self.last_loss:.6f}"

    def reset_metrics(self):
        pass

    def update_metrics(self, payload):
        del payload

    def report_metrics(self):
        return ""


class SemanticTokenObjective(WeightedObjective):
    """Teacher-forced causal token loss applied only to target tokens."""

    def __init__(self, *, weight_parameter: str = "semantic_weight"):
        super().__init__("semantic_token", weight_parameter=weight_parameter)

    def compute(self, context, params):
        context.require("logits", "labels", "payload")
        logits = context.get("logits")[:, :-1, :]
        labels = context.get("labels")[:, 1:]
        valid = labels.ne(-100)
        token_count = int(valid.sum().item())
        if token_count == 0:
            raise ValueError("semantic objective received no target tokens")
        loss_sum = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        loss = loss_sum / token_count
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(labels).logical_and(valid).sum()
        top_k = min(5, logits.shape[-1])
        top5_correct = logits.topk(top_k, dim=-1).indices.eq(
            labels.clamp_min(0).unsqueeze(-1)
        ).any(dim=-1).logical_and(valid).sum()

        payload = context.get("payload")
        payload.metrics["semantic/token_loss"] = MetricValue(
            float(loss_sum.detach().item()), token_count
        )
        payload.metrics["semantic/token_accuracy"] = MetricValue(
            float(correct.detach().item()), token_count
        )
        payload.metrics["semantic/top5_accuracy"] = MetricValue(
            float(top5_correct.detach().item()), token_count
        )
        payload.predicted_token_ids = predictions.detach().cpu()
        payload.target_token_ids = labels.detach().cpu()
        payload.target_token_mask = valid.detach().cpu()
        self.last_loss = float(loss.detach().item())
        payload.objective_losses[self.name] = self.last_loss
        return loss * self.resolved_weight(params)


class SpeechTraitObjective(WeightedObjective):
    """Scale-robust masked regression for one speech-trait group."""

    def __init__(
        self,
        group: str,
        *,
        weight_parameter: str,
        beta_parameter: str = "trait_smooth_l1_beta",
    ):
        if not group:
            raise ValueError("speech-trait group must be non-empty")
        super().__init__(
            f"speech_trait/{group}",
            weight_parameter=weight_parameter,
        )
        self.group = group
        self.beta_parameter = beta_parameter

    def compute(self, context, params):
        context.require("representation", "speech_trait_targets", "payload")
        predictions = context.get("representation").trait_predictions
        targets = context.get("speech_trait_targets")
        if not isinstance(targets, Mapping):
            raise TypeError("speech_traits must be a mapping keyed by trait group")
        if self.group not in predictions:
            raise KeyError(f"provider has no speech-trait prediction {self.group!r}")
        if self.group not in targets:
            raise KeyError(f"missing speech-trait target group {self.group!r}")

        prediction = predictions[self.group]
        target, valid = self._target_and_mask(targets[self.group], prediction)
        valid_values = valid.expand_as(prediction)
        count = int(valid_values.sum().item())
        if count == 0:
            raise ValueError(
                f"speech-trait group {self.group!r} has no valid targets"
            )
        beta = self._resolved_beta(params)
        element_loss = functional.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
            beta=beta,
        )
        loss = element_loss.masked_select(valid_values).mean()
        squared_error = (prediction - target).square().masked_select(valid_values)
        payload = context.get("payload")
        payload.metrics[f"traits/{self.group}_mse"] = MetricValue(
            float(squared_error.detach().sum().item()), count
        )
        payload.metrics[f"traits/{self.group}_smooth_l1"] = MetricValue(
            float(element_loss.masked_select(valid_values).detach().sum().item()),
            count,
        )
        self.last_loss = float(loss.detach().item())
        payload.objective_losses[self.name] = self.last_loss
        return loss * self.resolved_weight(params)

    def _resolved_beta(self, params) -> float:
        if params is None or not params.has(self.beta_parameter):
            raise KeyError(
                f"missing required SmoothL1 beta {self.beta_parameter!r}"
            )
        beta = float(params.get(self.beta_parameter))
        if beta <= 0:
            raise ValueError(f"{self.beta_parameter!r} must be positive")
        return beta

    @staticmethod
    def _target_and_mask(value, prediction: Tensor) -> tuple[Tensor, Tensor]:
        if isinstance(value, Mapping):
            raw_target = value.get("values")
            raw_valid = value.get("valid_mask")
        else:
            raw_target = value
            raw_valid = None
        target = torch.as_tensor(
            raw_target, dtype=prediction.dtype, device=prediction.device
        )
        if target.shape != prediction.shape:
            raise ValueError(
                f"trait target shape {tuple(target.shape)} does not match "
                f"prediction shape {tuple(prediction.shape)}"
            )
        if raw_valid is None:
            valid = torch.isfinite(target)
        else:
            valid = torch.as_tensor(raw_valid, dtype=torch.bool, device=prediction.device)
            while valid.ndim < prediction.ndim:
                valid = valid.unsqueeze(-1)
            try:
                valid = valid.expand_as(prediction)
            except RuntimeError as error:
                raise ValueError("trait valid_mask is not broadcastable to target") from error
            valid = valid.logical_and(torch.isfinite(target))
        return target.nan_to_num(), valid
