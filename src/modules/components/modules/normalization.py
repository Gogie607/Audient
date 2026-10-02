"""Normalization layers with explicit mixed-precision behavior."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class Float32RMSNorm(nn.Module):
    """RMSNorm with FP32 computation and activation-dtype output."""

    def __init__(self, normalized_shape: int | tuple[int, ...], eps=None) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, values: Tensor) -> Tensor:
        input_dtype = values.dtype
        eps = self.eps
        if eps is None:
            eps = torch.finfo(input_dtype).eps
        normalized = functional.rms_norm(
            values.float(),
            self.normalized_shape,
            self.weight.float(),
            eps,
        )
        return normalized.to(dtype=input_dtype)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"

