"""Runtime state control for model components."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class Controllable(ABC):
    """Contract for objects with restorable mode and trainability state."""

    @abstractmethod
    def get_mode(self):
        """Return state sufficient to restore the current execution mode."""

    @abstractmethod
    def set_mode(self, state):
        """Apply a state previously accepted or returned by this object."""

    @abstractmethod
    def get_trainable(self):
        """Return state sufficient to restore parameter trainability."""

    @abstractmethod
    def set_trainable(self, state):
        """Apply trainability state and return this object."""


class ControllableNode(nn.Module, Controllable):
    """Boolean mode/trainability control for a leaf PyTorch component."""

    def __init__(self) -> None:
        super().__init__()
        self._trainable = True

    def get_mode(self) -> bool:
        return self.training

    def set_mode(self, state: bool):
        if not isinstance(state, bool):
            raise TypeError("leaf model mode must be a boolean")
        nn.Module.train(self, state)
        return self

    def get_trainable(self) -> bool:
        return self._trainable

    def set_trainable(self, state: bool):
        if not isinstance(state, bool):
            raise TypeError("leaf trainability state must be a boolean")

        self._trainable = state
        for parameter in self.parameters():
            parameter.requires_grad_(state)
        return self

    def trainable_parameters(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def device(self) -> torch.device:
        parameter = next(self.parameters(), None)
        if parameter is not None:
            return parameter.device

        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.device

        return torch.device("cpu")

