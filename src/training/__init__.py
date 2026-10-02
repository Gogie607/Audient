"""Research-specific training assembly."""

from .audio_prefix import (
    AudioObjectiveComposer,
    AudioPrefixForward,
    AudioPrefixMetrics,
    AudioPrefixPayload,
    AudioPrefixSnapshotCollector,
    JsonSnapshotWriter,
    SemanticTokenObjective,
    SpeechTraitObjective,
    create_audio_prefix_training_module,
)
from .session import RunLogger, SessionFactories, TrainingSession, run_training

__all__ = [
    "AudioObjectiveComposer",
    "AudioPrefixForward",
    "AudioPrefixMetrics",
    "AudioPrefixPayload",
    "AudioPrefixSnapshotCollector",
    "JsonSnapshotWriter",
    "SemanticTokenObjective",
    "SpeechTraitObjective",
    "RunLogger",
    "SessionFactories",
    "TrainingSession",
    "create_audio_prefix_training_module",
    "run_training",
]
