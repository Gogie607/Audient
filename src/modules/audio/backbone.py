"""Runtime-only audio backbone contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from .payloads import AudioBackboneFeatures, AudioBackboneSpec


class AudioBackbone(nn.Module, ABC):
    """Frozen, externally reconstructed audio feature preprocessor.

    Audio backbones are deliberately not ModelComponents. They are not owned or
    serialized by an AudioProvider checkpoint.
    """

    def __init__(self, model: nn.Module, spec: AudioBackboneSpec) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self._freeze()

    @abstractmethod
    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
        padding_mask: Tensor | None = None,
    ) -> AudioBackboneFeatures:
        """Produce the provider-boundary features from a prepared batch."""

    def train(self, mode: bool = True) -> AudioBackbone:
        del mode
        super().train(False)
        self._freeze()
        return self

    def _freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
