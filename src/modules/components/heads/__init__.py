"""Reusable output heads for model components."""

from .semantic_prefix import SemanticPrefixHead
from .speech_traits import SpeechTraitHeadOutput, SpeechTraitHeads

__all__ = [
    "SemanticPrefixHead",
    "SpeechTraitHeadOutput",
    "SpeechTraitHeads",
]
