"""Optional decoding contract kept separate from audio representation extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from .payloads import AudioRepresentation


class AudioDecoder(nn.Module, ABC):
    """Convert an audio representation back into waveform samples."""

    @abstractmethod
    def decode(self, representation: AudioRepresentation) -> Tensor:
        """Decode a complete representation batch."""


class AudioDecodeStream(ABC):
    """Stateful incremental decoder created by a compatible AudioDecoder."""

    @abstractmethod
    def decode_chunk(
        self,
        representation: AudioRepresentation,
        *,
        is_final: bool = False,
    ) -> Tensor:
        """Decode one representation chunk and retain synthesis state."""
