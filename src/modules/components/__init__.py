"""Reusable model components and concrete provider definitions."""

from .heads import SemanticPrefixHead, SpeechTraitHeadOutput, SpeechTraitHeads
from .modules import (
    AttentionPool,
    CrossAttentionBlock,
    FeedForward,
    LearnedQueryPool,
    MaskedTemporalReducer,
    SelfAttentionBlock,
)
from .providers import WhisperProvider

__all__ = [
    "AttentionPool",
    "CrossAttentionBlock",
    "FeedForward",
    "LearnedQueryPool",
    "MaskedTemporalReducer",
    "SelfAttentionBlock",
    "SemanticPrefixHead",
    "SpeechTraitHeadOutput",
    "SpeechTraitHeads",
    "WhisperProvider",
]
