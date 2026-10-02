"""Small assembly boundary for the first complete training module."""

from __future__ import annotations

from runweaver_ml.phase_control import TrainingModule

from .collector import AudioPrefixSnapshotCollector
from .composer import AudioObjectiveComposer
from .forward import AudioPrefixForward
from .metrics import AudioPrefixMetrics
from .objectives import (
    PositionWeightedSemanticObjective,
    SemanticAudioContrastObjective,
    SemanticTokenObjective,
    SpeechTraitObjective,
)


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
    semantic_objective: str = "standard",
    position_weight_parameters=None,
    enable_audio_contrast: bool = False,
    contrast_weight_parameter: str = "semantic_contrast_weight",
    contrast_margin_parameter: str = "semantic_contrast_margin",
    contrast_position_weight_parameters=None,
    trait_weight_parameters=None,
    trait_beta_parameter: str = "trait_smooth_l1_beta",
    include_traits: bool = True,
    prompt: str = "",
    temporal_trait_groups: tuple[str, ...] = ("timing", "energy", "pitch"),
    collector=None,
    checkpoint_handler=None,
):
    """Assemble objectives and handlers without constructing models or data."""
    semantic_objectives = {
        "standard": lambda: SemanticTokenObjective(
            weight_parameter=semantic_weight_parameter
        ),
        "position_weighted": lambda: PositionWeightedSemanticObjective(
            weight_parameter=semantic_weight_parameter,
            position_weight_parameters=position_weight_parameters,
        ),
    }
    try:
        objectives = [semantic_objectives[semantic_objective]()]
    except KeyError as error:
        raise ValueError(
            f"unsupported semantic_objective {semantic_objective!r}"
        ) from error
    if enable_audio_contrast:
        if semantic_objective != "position_weighted":
            raise ValueError("audio contrast requires position_weighted semantics")
        objectives.append(SemanticAudioContrastObjective(
            weight_parameter=contrast_weight_parameter,
            margin_parameter=contrast_margin_parameter,
            position_weight_parameters=contrast_position_weight_parameters,
        ))
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
            include_shuffled_semantic=enable_audio_contrast,
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
