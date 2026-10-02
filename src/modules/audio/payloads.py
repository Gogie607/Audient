"""Data-only contracts exchanged by audio providers and downstream modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from torch import Tensor


class RepresentationKind(str, Enum):
    """The numerical form of one provider output stream."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    RVQ = "rvq"
    TRAITS = "traits"


@dataclass(frozen=True)
class AudioBackboneSpec:
    """Persistent identity of the non-checkpointed feature producer."""

    identifier: str
    revision: str | None
    output_layer: str | int | None
    feature_dimension: int
    frame_rate_hz: float

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("backbone identifier must not be empty")
        if self.feature_dimension <= 0:
            raise ValueError("feature_dimension must be positive")
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")

    def to_config(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "output_layer": self.output_layer,
            "feature_dimension": self.feature_dimension,
            "frame_rate_hz": self.frame_rate_hz,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> AudioBackboneSpec:
        return cls(
            identifier=str(config["identifier"]),
            revision=config.get("revision"),
            output_layer=config.get("output_layer"),
            feature_dimension=int(config["feature_dimension"]),
            frame_rate_hz=float(config["frame_rate_hz"]),
        )


@dataclass(frozen=True)
class AudioTiming:
    """Timing information for a provider output chunk."""

    sample_rate: int
    input_sample_offset: int
    input_sample_count: int
    output_frame_offset: int
    output_frame_count: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        for name in (
            "input_sample_offset",
            "input_sample_count",
            "output_frame_offset",
            "output_frame_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class AudioStream:
    """One aligned representation stream emitted by an audio provider."""

    values: Tensor
    kind: RepresentationKind
    frame_rate_hz: float
    padding_mask: Tensor | None = None
    frame_times_seconds: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim < 2:
            raise ValueError("values must begin with [batch, frames]")
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")

        expected_shape = self.values.shape[:2]
        if self.padding_mask is not None and self.padding_mask.shape != expected_shape:
            raise ValueError("padding_mask must have shape [batch, frames]")
        if (
            self.frame_times_seconds is not None
            and self.frame_times_seconds.shape != expected_shape
        ):
            raise ValueError("frame_times_seconds must have shape [batch, frames]")


@dataclass(frozen=True)
class AudioBackboneFeatures:
    """Prepared output at the boundary after an audio backbone.

    Dataset management may supply compatible pregenerated instances during
    training. Live inference obtains the same payload from the provider's
    backbone hook.
    """

    values: Tensor
    backbone: AudioBackboneSpec
    timing: AudioTiming
    padding_mask: Tensor | None = None
    frame_times_seconds: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim < 2:
            raise ValueError("values must begin with [batch, frames]")
        if self.values.shape[-1] != self.backbone.feature_dimension:
            raise ValueError(
                "last feature dimension must match backbone.feature_dimension"
            )

        expected_shape = self.values.shape[:2]
        if self.values.shape[1] != self.timing.output_frame_count:
            raise ValueError(
                "feature frame count must match timing.output_frame_count"
            )
        if self.padding_mask is not None and self.padding_mask.shape != expected_shape:
            raise ValueError("padding_mask must have shape [batch, frames]")
        if (
            self.frame_times_seconds is not None
            and self.frame_times_seconds.shape != expected_shape
        ):
            raise ValueError("frame_times_seconds must have shape [batch, frames]")


@dataclass(frozen=True)
class AudioRepresentation:
    """Topology-preserving output of an audio provider."""

    semantic: AudioStream
    timing: AudioTiming
    acoustic: AudioStream | None = None
    trait_conditioning: AudioStream | None = None
    trait_predictions: Mapping[str, Tensor] = field(default_factory=dict)
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        semantic_frames = self.semantic.values.shape[1]
        if semantic_frames != self.timing.output_frame_count:
            raise ValueError(
                "semantic frame count must match timing.output_frame_count"
            )


@dataclass(frozen=True)
class AudioProviderCapabilities:
    """Capabilities exposed by a concrete provider implementation."""

    supports_streaming: bool = False
    emits_acoustic_stream: bool = False
    emitted_trait_names: tuple[str, ...] = ()
    semantic_kind: RepresentationKind = RepresentationKind.CONTINUOUS
    acoustic_kind: RepresentationKind | None = None


@dataclass
class AudioProviderState:
    """Provider-owned state carried between streamed chunks."""

    input_sample_offset: int = 0
    output_frame_offset: int = 0
    provider_state: Any = None

    def __post_init__(self) -> None:
        if self.input_sample_offset < 0 or self.output_frame_offset < 0:
            raise ValueError("stream offsets must be non-negative")
