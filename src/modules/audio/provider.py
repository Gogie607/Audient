"""Base contract for composite audio providers."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from torch import nn

from ..base import ModelComponent
from .payloads import (
    AudioBackboneFeatures,
    AudioBackboneSpec,
    AudioProviderCapabilities,
    AudioProviderState,
    AudioRepresentation,
)


class AudioProvider(ModelComponent):
    """Trainable composite that converts backbone features into an LLM prefix.

    The required frozen backbone is an external runtime preprocessor. Only its
    specification is retained here; its weights are not part of provider state.
    """

    component_type = None

    def __init__(
        self,
        backbone_spec: AudioBackboneSpec,
        *,
        trainable_modules: Mapping[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.backbone_spec = backbone_spec
        self.provider_modules = nn.ModuleDict(dict(trainable_modules or {}))

    @property
    @abstractmethod
    def capabilities(self) -> AudioProviderCapabilities:
        """Describe output forms and optional provider features."""

    def forward(
        self,
        features: AudioBackboneFeatures,
    ) -> AudioRepresentation:
        """Run the complete provider prefix using normal PyTorch semantics."""
        self.validate_backbone_features(features)
        return self.forward_features(features)

    @abstractmethod
    def forward_features(
        self,
        features: AudioBackboneFeatures,
    ) -> AudioRepresentation:
        """Transform backbone-boundary features into the provider output."""

    def validate_backbone_features(self, features: AudioBackboneFeatures) -> None:
        if features.backbone != self.backbone_spec:
            raise ValueError(
                "backbone features are incompatible with this provider: "
                f"expected {self.backbone_spec!r}, received {features.backbone!r}"
            )

    def serialize(self) -> dict:
        payload = super().serialize()
        expected_spec = self.backbone_spec.to_config()
        if payload["config"].get("backbone_spec") != expected_spec:
            raise RuntimeError(
                "AudioProvider.get_config() must include its exact "
                "backbone_spec so the runtime preprocessor can be recreated"
            )
        return payload

    def create_stream_state(self) -> AudioProviderState:
        if not self.capabilities.supports_streaming:
            raise RuntimeError(f"{type(self).__name__} does not support streaming")
        return AudioProviderState()

    def encode_chunk(
        self,
        features: AudioBackboneFeatures,
        *,
        state: AudioProviderState,
        is_final: bool = False,
    ) -> tuple[AudioRepresentation, AudioProviderState]:
        del features, state, is_final
        raise RuntimeError(f"{type(self).__name__} does not support streaming")

    def train(self, mode: bool = True) -> AudioProvider:
        super().train(mode)
        return self

    def get_mode(self) -> dict[str, bool]:
        return {
            **{
                name: module.training
                for name, module in self.provider_modules.items()
            },
        }

    def set_mode(self, mode: bool | Mapping[str, bool]) -> AudioProvider:
        if isinstance(mode, bool):
            self.train(mode)
            return self

        unknown = set(mode) - set(self.provider_modules.keys())
        if unknown:
            raise KeyError(f"unknown provider modules: {sorted(unknown)}")
        for name, module_mode in mode.items():
            self.provider_modules[name].train(module_mode)
        return self

    def get_trainable(self) -> dict[str, bool]:
        return {
            **{
                name: all(parameter.requires_grad for parameter in module.parameters())
                for name, module in self.provider_modules.items()
            },
        }

    def set_trainable(
        self,
        trainable: bool | Mapping[str, bool],
    ) -> AudioProvider:
        if isinstance(trainable, bool):
            for parameter in self.provider_modules.parameters():
                parameter.requires_grad = trainable
            return self

        unknown = set(trainable) - set(self.provider_modules.keys())
        if unknown:
            raise KeyError(f"unknown provider modules: {sorted(unknown)}")
        for name, module_trainable in trainable.items():
            for parameter in self.provider_modules[name].parameters():
                parameter.requires_grad = module_trainable
        return self
