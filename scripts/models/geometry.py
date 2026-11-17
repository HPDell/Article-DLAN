"""Multidimensional rotary adapter used in the original comparisons."""
import torch
from torch import Tensor, nn

class RotaryEmbeddingND(nn.Module):
    """Block-diagonal rotary embedding for an arbitrary coordinate dimension.

    The constructor matches the notebook's ``RotaryEmbedding2D`` interface so
    the reviewed GeoAggregator implementation can use this class without other
    architectural changes. Rotation blocks are distributed across coordinate
    axes; repeated blocks on an axis receive progressively lower frequencies.
    """

    def __init__(
        self,
        d: int = 4,
        base: int = 10_000,
        scale: float = 1.0,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if d < 2 or d % 2:
            raise ValueError("rotary dimension must be a positive even number")
        self.d = d
        self.base = base
        self.scale = scale
        self.n_heads = n_heads

    def embed(self, spa_feat: Tensor) -> Tensor:
        """Return rotation matrices with shape ``[batch, sequence, d, d]``."""

        if spa_feat.ndim != 3:
            raise ValueError("spatial features must have shape [batch, sequence, dim]")
        batch_size, sequence_length, coordinate_dim = spa_feat.shape
        block_count = self.d // 2
        if coordinate_dim > block_count:
            raise ValueError(
                "coordinate dimension exceeds available rotary rotation blocks"
            )
        block_axis = torch.arange(block_count, device=spa_feat.device) % coordinate_dim
        block_order = torch.div(
            torch.arange(block_count, device=spa_feat.device),
            coordinate_dim,
            rounding_mode="floor",
        )
        max_order = max(int(block_order.max().item()) + 1, 1)
        frequency = 1.0 / (
            self.base ** (block_order.to(spa_feat.dtype) / max_order)
        )
        angle = (
            spa_feat[..., block_axis]
            * frequency.view(1, 1, -1)
            * self.scale
        )
        sine, cosine = torch.sin(angle), torch.cos(angle)
        rotation = torch.zeros(
            batch_size,
            sequence_length,
            self.d,
            self.d,
            dtype=spa_feat.dtype,
            device=spa_feat.device,
        )
        for block in range(block_count):
            lower = 2 * block
            rotation[..., lower, lower] = cosine[..., block]
            rotation[..., lower, lower + 1] = -sine[..., block]
            rotation[..., lower + 1, lower] = sine[..., block]
            rotation[..., lower + 1, lower + 1] = cosine[..., block]
        return rotation

    def forward(self, spa_feat: Tensor) -> Tensor:
        return self.embed(spa_feat)
