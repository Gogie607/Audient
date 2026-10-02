"""Small assembly boundary for the first complete training module."""

from __future__ import annotations

from runweaver_ml.phase_control import TrainingModule

from .collector import AudioPrefixSnapshotCollector
from .composer import AudioObjectiveComposer
from .forward import AudioPrefixForward
from .metrics import AudioPrefixMetrics
from .objectives import SemanticTokenObjective, SpeechTraitObjective


DEFAULT_TRAIT_WEIGHT_PARAMETERS = {
    "timing": "timing_weight",
    "energy": "energy_weight",
    "pitch": "pitch_weight",
    "voice": "voice_weight",
    "temporal": "temporal_weight",
}


def create_audio_prefix_training_module(
    *,
    model,
    language_core,
    params,
    train_mode,
    eval_mode,
    semantic_weight_parameter: str = "semantic_weight",
    trait_weight_parameters=None,
    trait_beta_parameter: str = "trait_smooth_l1_beta",
    include_traits: bool = True,
    prompt: str = "",
    temporal_trait_groups: tuple[str, ...] = ("timing", "energy", "pitch"),
    collector=None,
    checkpoint_handler=None,
):
    """Assemble objectives and handlers without constructing models or data."""
    objectives = [
        SemanticTokenObjective(weight_parameter=semantic_weight_parameter)
    ]
    if include_traits:
        weight_parameters = dict(
            trait_weight_parameters or DEFAULT_TRAIT_WEIGHT_PARAMETERS
        )
        objectives.extend(
            SpeechTraitObjective(
                group,
                weight_parameter=weight_parameter,
                beta_parameter=trait_beta_parameter,
            )
            for group, weight_parameter in weight_parameters.items()
        )
    trainer = AudioObjectiveComposer(
        model=model,
        forward=AudioPrefixForward(
            language_core,
            prompt=prompt,
            temporal_trait_groups=temporal_trait_groups,
        ),
        objectives=objectives,
        train_mode=train_mode,
        eval_mode=eval_mode,
    )
    return TrainingModule(
        trainer=trainer,
        params=params,
        metrics=AudioPrefixMetrics(),
        collector=(
            collector
            if collector is not None
            else AudioPrefixSnapshotCollector(max_samples=0)
        ),
        validation_owner="handlers",
        checkpoint_handler=checkpoint_handler,
    )
