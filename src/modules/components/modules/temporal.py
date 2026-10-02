"""Reusable temporal sequence transformations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MaskedTemporalReducer(nn.Module):
    """Locally reduce a padded sequence without using fixed latent queries."""

    def __init__(self, model_dim: int, *, reduction: int = 4) -> None:
        super().__init__()
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        self.reduction = reduction
        self.depthwise = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=reduction,
            stride=reduction,
            groups=model_dim,
        )
        self.gated_projection = nn.Conv1d(model_dim, model_dim * 2, kernel_size=1)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        values: Tensor,
        *,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, frames, dimension]")
        batch_size, frame_count, _ = values.shape
        if padding_mask is not None:
            if padding_mask.shape != (batch_size, frame_count):
                raise ValueError("padding_mask must have shape [batch, frames]")
            if padding_mask.dtype is not torch.bool:
                raise TypeError("padding_mask must be boolean")
            if padding_mask.all(dim=1).any():
                raise ValueError("every sample must contain at least one valid frame")
            values = values.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        target_frames = math.ceil(frame_count / self.reduction)
        padded_frames = target_frames * self.reduction
        trailing_padding = padded_frames - frame_count
        channels_first = values.transpose(1, 2)
        if trailing_padding:
            channels_first = F.pad(channels_first, (0, trailing_padding))
        reduced = self.depthwise(channels_first)
        reduced = F.glu(self.gated_projection(reduced), dim=1)
        reduced = self.output_norm(reduced.transpose(1, 2))

        if padding_mask is None:
            return reduced, None

        valid = (~padding_mask).to(dtype=values.dtype).unsqueeze(1)
        if trailing_padding:
            valid = F.pad(valid, (0, trailing_padding))
        valid = F.max_pool1d(
            valid,
            kernel_size=self.reduction,
            stride=self.reduction,
        )
        return reduced, ~valid[:, 0].bool()
