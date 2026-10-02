"""Reusable transformer feed-forward blocks."""

from __future__ import annotations

from torch import Tensor, nn


class FeedForward(nn.Module):
    """GELU transformer feed-forward network preserving the final dimension."""

    def __init__(
        self,
        model_dim: int,
        *,
        hidden_dim: int | None = None,
        expansion: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        resolved_hidden_dim = hidden_dim or int(model_dim * expansion)
        if model_dim <= 0 or resolved_hidden_dim <= 0:
            raise ValueError("feed-forward dimensions must be positive")
        self.net = nn.Sequential(
            nn.Linear(model_dim, resolved_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(resolved_hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)
