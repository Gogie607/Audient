"""Composable losses for the first Whisper-to-Qwen experiment."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor

from runweaver_ml.phase_control import TrainingObjective

from .payloads import MetricValue


POSITION_REGIONS = {
    "first": (1, 1),
    "early": (2, 4),
    "middle": (5, 8),
    "remaining": (9, None),
}

DEFAULT_POSITION_WEIGHT_PARAMETERS = {
    "first": "semantic_first_weight",
    "early": "semantic_early_weight",
    "middle": "semantic_middle_weight",
    "remaining": "semantic_remaining_weight",
}

DEFAULT_CONTRAST_POSITION_WEIGHT_PARAMETERS = {
    "first": "contrast_first_weight",
    "early": "contrast_early_weight",
    "middle": "contrast_middle_weight",
}


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


def _aligned_tokens(context):
    logits = context.get("logits")[:, :-1, :]
    labels = context.get("labels")[:, 1:]
    valid = labels.ne(-100)
    positions = valid.long().cumsum(dim=1)
    return logits, labels, valid, positions


def _region_mask(valid: Tensor, positions: Tensor, region: str) -> Tensor:
    start, end = POSITION_REGIONS[region]
    mask = valid.logical_and(positions >= start)
    if end is not None:
        mask = mask.logical_and(positions <= end)
    return mask


def _unreduced_token_nll(logits: Tensor, labels: Tensor) -> Tensor:
    return functional.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=-100,
        reduction="none",
    )


class PositionWeightedSemanticObjective(WeightedObjective):
    """Teacher-forced CE formed from independently normalized position groups."""

    def __init__(
        self,
        *,
        weight_parameter: str = "semantic_weight",
        position_weight_parameters=None,
    ):
        super().__init__("semantic_position_weighted", weight_parameter=weight_parameter)
        self.position_weight_parameters = dict(
            position_weight_parameters or DEFAULT_POSITION_WEIGHT_PARAMETERS
        )
        if set(self.position_weight_parameters) != set(POSITION_REGIONS):
            raise ValueError(
                "position weights must define first, early, middle, and remaining"
            )

    def compute(self, context, params):
        context.require("logits", "labels", "payload")
        logits, labels, valid, positions = _aligned_tokens(context)
        token_count = int(valid.sum().item())
        if token_count == 0:
            raise ValueError("semantic objective received no target tokens")
        token_nll = _unreduced_token_nll(logits, labels)
        weighted_loss = logits.new_zeros(())
        active_weight = 0.0
        payload = context.get("payload")
        predictions = logits.argmax(dim=-1)

        for region, parameter_name in self.position_weight_parameters.items():
            mask = _region_mask(valid, positions, region)
            count = int(mask.sum().item())
            if count == 0:
                continue
            weight = self._resolved_region_weight(params, parameter_name)
            region_loss = token_nll.masked_select(mask).mean()
            weighted_loss = weighted_loss + weight * region_loss
            active_weight += weight
            correct = predictions.eq(labels).logical_and(mask).sum()
            payload.metrics[f"semantic/{region}_loss"] = MetricValue(
                float(token_nll.masked_select(mask).detach().sum().item()), count
            )
            payload.metrics[f"semantic/{region}_accuracy"] = MetricValue(
                float(correct.detach().item()), count
            )
        if active_weight <= 0:
            raise ValueError("semantic position weights have no positive active weight")
        loss = weighted_loss / active_weight
        self._record_legacy_metrics(
            payload, logits, labels, valid, token_nll, predictions
        )
        self.last_loss = float(loss.detach().item())
        payload.objective_losses[self.name] = self.last_loss
        return loss * self.resolved_weight(params)

    @staticmethod
    def _resolved_region_weight(params, name: str) -> float:
        if params is None or not params.has(name):
            raise KeyError(f"missing required position weight {name!r}")
        weight = float(params.get(name))
        if weight < 0:
            raise ValueError(f"position weight {name!r} must be non-negative")
        return weight

    @staticmethod
    def _record_legacy_metrics(
        payload, logits, labels, valid, token_nll, predictions
    ) -> None:
        token_count = int(valid.sum().item())
        correct = predictions.eq(labels).logical_and(valid).sum()
        top_k = min(5, logits.shape[-1])
        top5_correct = logits.topk(top_k, dim=-1).indices.eq(
            labels.clamp_min(0).unsqueeze(-1)
        ).any(dim=-1).logical_and(valid).sum()
        payload.metrics["semantic/token_loss"] = MetricValue(
            float(token_nll.masked_select(valid).detach().sum().item()), token_count
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


class SemanticAudioContrastObjective(WeightedObjective):
    """Require correct audio to support early targets more than shuffled audio."""

    def __init__(
        self,
        *,
        weight_parameter: str = "semantic_contrast_weight",
        margin_parameter: str = "semantic_contrast_margin",
        position_weight_parameters=None,
    ):
        super().__init__("semantic_audio_contrast", weight_parameter=weight_parameter)
        self.margin_parameter = margin_parameter
        self.position_weight_parameters = dict(
            position_weight_parameters
            or DEFAULT_CONTRAST_POSITION_WEIGHT_PARAMETERS
        )
        expected = {"first", "early", "middle"}
        if set(self.position_weight_parameters) != expected:
            raise ValueError(
                "contrast position weights must define first, early, and middle"
            )

    def compute(self, context, params):
        context.require("logits", "shuffled_logits", "labels", "payload")
        correct_logits, labels, valid, positions = _aligned_tokens(context)
        shuffled_logits = context.get("shuffled_logits")[:, :-1, :]
        if shuffled_logits.shape != correct_logits.shape:
            raise ValueError("correct and shuffled logits must have matching shapes")
        correct_nll = _unreduced_token_nll(correct_logits, labels)
        shuffled_nll = _unreduced_token_nll(shuffled_logits, labels)
        logp_advantage = shuffled_nll - correct_nll
        margin = self._resolved_margin(params)
        hinge = functional.relu(margin - logp_advantage)
        weighted_loss = correct_logits.new_zeros(())
        active_weight = 0.0
        payload = context.get("payload")

        for region, parameter_name in self.position_weight_parameters.items():
            mask = _region_mask(valid, positions, region)
            count = int(mask.sum().item())
            if count == 0:
                continue
            weight = PositionWeightedSemanticObjective._resolved_region_weight(
                params, parameter_name
            )
            region_loss = hinge.masked_select(mask).mean()
            weighted_loss = weighted_loss + weight * region_loss
            active_weight += weight
            advantage_values = logp_advantage.masked_select(mask)
            satisfied = advantage_values.ge(margin).sum()
            payload.metrics[f"semantic/contrast_{region}_logp_advantage"] = MetricValue(
                float(advantage_values.detach().sum().item()), count
            )
            payload.metrics[f"semantic/contrast_{region}_margin_satisfaction"] = MetricValue(
                float(satisfied.detach().item()), count
            )
        if active_weight <= 0:
            raise ValueError("contrast position weights have no positive active weight")
        loss = weighted_loss / active_weight
        self.last_loss = float(loss.detach().item())
        payload.objective_losses[self.name] = self.last_loss
        return loss * self.resolved_weight(params)

    def _resolved_margin(self, params) -> float:
        if params is None or not params.has(self.margin_parameter):
            raise KeyError(f"missing required contrast margin {self.margin_parameter!r}")
        margin = float(params.get(self.margin_parameter))
        if margin <= 0:
            raise ValueError("semantic contrast margin must be positive")
        return margin


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
