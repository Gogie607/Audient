"""Composable Whisper-provider to frozen-Qwen training recipe."""

from .collector import AudioPrefixSnapshotCollector, JsonSnapshotWriter
from .composer import AudioObjectiveComposer
from .factory import create_audio_prefix_training_module
from .forward import AudioPrefixForward
from .metrics import AudioPrefixMetrics
from .objectives import (
    PositionWeightedSemanticObjective,
    SemanticAudioContrastObjective,
    SemanticTokenObjective,
    SpeechTraitObjective,
)
from .payloads import AudioPrefixPayload, MetricValue, TeacherForcedInputs

__all__ = [
    "AudioObjectiveComposer",
    "AudioPrefixForward",
    "AudioPrefixMetrics",
    "AudioPrefixPayload",
    "AudioPrefixSnapshotCollector",
    "JsonSnapshotWriter",
    "MetricValue",
    "SemanticTokenObjective",
    "PositionWeightedSemanticObjective",
    "SemanticAudioContrastObjective",
    "SpeechTraitObjective",
    "TeacherForcedInputs",
    "create_audio_prefix_training_module",
]
