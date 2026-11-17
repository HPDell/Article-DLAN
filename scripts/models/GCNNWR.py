"""Parameter-matched geographically convolutional neural weighted regression.

The model follows the core GCNNWR construction: a regular geographic grid is
converted into a distance image for each query, a convolutional network emits
location-specific multipliers, and those multipliers modulate global OLS
coefficients.  The grid size changes spatial resolution but not parameter
count because the encoder ends with adaptive pooling.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def regular_spatial_grid(u: Tensor, shape: tuple[int, int]) -> Tensor:
    """Create a row-major regular grid over the training coordinate bounds."""

    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError("u must have shape [n, 2]")
    rows, columns = shape
    if rows < 2 or columns < 2:
        raise ValueError("both grid dimensions must be at least two")
    lower, upper = u.amin(dim=0), u.amax(dim=0)
    first = torch.linspace(lower[0], upper[0], rows, device=u.device)
    second = torch.linspace(lower[1], upper[1], columns, device=u.device)
    mesh = torch.meshgrid(first, second, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, 2)


def fit_global_ols(x: Tensor, y: Tensor) -> Tensor:
    """Fit intercept plus slopes using a stable least-squares solve."""

    if x.ndim != 2 or y.ndim != 2 or y.shape[1] != 1 or len(x) != len(y):
        raise ValueError("x and y must be aligned rank-2 tensors with scalar y")
    design = torch.cat((torch.ones_like(x[:, :1]), x), dim=1).double()
    solution = torch.linalg.lstsq(design, y.double()).solution
    return solution.float().reshape(-1)


class GCNNWR(nn.Module):
    """Predict spatially varying OLS multipliers from coordinate distances."""

    def __init__(
        self,
        input_dim: int,
        grid: Tensor,
        distance_mean: Tensor,
        distance_scale: Tensor,
        ols_coefficients: Tensor,
        *,
        grid_shape: tuple[int, int],
        conv_channels: Sequence[int] = (4, 8),
        hidden_dim: int = 37,
        pooled_shape: tuple[int, int] = (4, 4),
    ) -> None:
        super().__init__()
        if len(conv_channels) != 2:
            raise ValueError("conv_channels must contain exactly two widths")
        if grid.shape != (grid_shape[0] * grid_shape[1], 2):
            raise ValueError("grid shape does not match grid_shape")
        if distance_mean.shape != (grid.shape[0],):
            raise ValueError("distance_mean must contain one value per grid cell")
        if distance_scale.shape != distance_mean.shape:
            raise ValueError("distance_scale must match distance_mean")
        if ols_coefficients.shape != (input_dim + 1,):
            raise ValueError("OLS coefficients must contain intercept plus slopes")

        self.input_dim = input_dim
        self.grid_shape = grid_shape
        self.register_buffer("grid", grid.detach().float())
        self.register_buffer("distance_mean", distance_mean.detach().float())
        self.register_buffer(
            "distance_scale", distance_scale.detach().float().clamp_min(1e-6)
        )
        self.register_buffer("ols_coefficients", ols_coefficients.detach().float())

        first, second = (int(width) for width in conv_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, first, kernel_size=3, padding=1),
            nn.PReLU(first),
            nn.MaxPool2d(2),
            nn.Conv2d(first, second, kernel_size=3, padding=1),
            nn.PReLU(second),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(pooled_shape),
            nn.Flatten(),
        )
        encoded_dim = second * pooled_shape[0] * pooled_shape[1]
        self.coefficient_head = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.PReLU(hidden_dim),
            nn.Linear(hidden_dim, input_dim + 1),
        )
        # Start from global OLS, as in the original model's coefficient
        # modulation, rather than from arbitrary random local coefficients.
        final = self.coefficient_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.ones_(final.bias)

    @classmethod
    def from_data(
        cls,
        u: Tensor,
        x: Tensor,
        y: Tensor,
        *,
        grid_shape: tuple[int, int],
        conv_channels: Sequence[int] = (4, 8),
        hidden_dim: int = 37,
        pooled_shape: tuple[int, int] = (4, 4),
        distance_chunk_size: int = 2048,
    ) -> "GCNNWR":
        """Fit the grid-distance scaling and OLS base from training rows only."""

        if len(u) != len(x) or len(x) != len(y):
            raise ValueError("u, x, and y must contain the same number of rows")
        grid = regular_spatial_grid(u.float(), grid_shape)
        sums = torch.zeros(len(grid), device=u.device, dtype=torch.float64)
        squares = torch.zeros_like(sums)
        for start in range(0, len(u), distance_chunk_size):
            distance = torch.cdist(u[start : start + distance_chunk_size].float(), grid)
            sums += distance.double().sum(dim=0)
            squares += distance.double().square().sum(dim=0)
        mean = sums / len(u)
        variance = (squares / len(u) - mean.square()).clamp_min(1e-12)
        return cls(
            x.shape[1],
            grid,
            mean.float(),
            variance.sqrt().float(),
            fit_global_ols(x, y),
            grid_shape=grid_shape,
            conv_channels=conv_channels,
            hidden_dim=hidden_dim,
            pooled_shape=pooled_shape,
        )

    def coefficient_multipliers(self, u: Tensor) -> Tensor:
        """Return the spatial multiplier for every OLS coefficient."""

        distance = torch.cdist(u.float(), self.grid)
        image = ((distance - self.distance_mean) / self.distance_scale).reshape(
            -1, 1, *self.grid_shape
        )
        return self.coefficient_head(self.encoder(image))

    def local_coefficients(self, u: Tensor) -> Tensor:
        """Return intercept and slopes at each query coordinate."""

        return self.coefficient_multipliers(u) * self.ols_coefficients

    def forward(self, u: Tensor, x: Tensor) -> Tensor:
        """Predict a standardized response from query coordinates/features."""

        if len(u) != len(x) or x.shape[1] != self.input_dim:
            raise ValueError("u and x have incompatible shapes")
        design = torch.cat((torch.ones_like(x[:, :1]), x.float()), dim=1)
        return (design * self.local_coefficients(u)).sum(dim=1, keepdim=True)
