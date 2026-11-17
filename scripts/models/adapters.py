"""Capacity-controlled DLAN/GeoAggregator/GCNNWR case-study adapters.

The established networks are unchanged. Widths are chosen by parameter count
alone, not validation/test accuracy; all configurations have 6,957 trainable
parameters. GVI uses its original 176 fixed features and two coordinates.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import GeoAggregator as geo_module
from .DLAN import (
    DeepGWRTrainer, GTDRDeepGWRTransformer, TrainerConfig, build_knn,
)
from .GCNNWR import GCNNWR, regular_spatial_grid
from .geometry import RotaryEmbeddingND
from .training import GTDRReferenceTrainer, reference_candidates


PARAMETER_BUDGET = 6957
ARCHITECTURES = {
    'CO2': dict(x_dim=8, coordinate_dim=3, dlan=(13, 12, 24), geo=(48, 14, 22)),
    'OD-Wuhan': dict(x_dim=35, coordinate_dim=4, dlan=(10, 36, 3), geo=(48, 4, 32)),
    'WuhanHP': dict(x_dim=17, coordinate_dim=2, dlan=(14, 20, 17), geo=(32, 62, 14),
                    gcnn=(5, 16, 2, 73)),
    'GVI': dict(x_dim=176, coordinate_dim=2, dlan=(7, 12, 58), geo=(32, 7, 11),
                gcnn=(8, 17, 2, 22)),
}


class MatchedGeoAggregator(nn.Module):
    """Use existing 2D/ND rotary embeddings without discarding coordinates."""

    def __init__(self, x_dim: int, coordinate_dim: int, widths: tuple) -> None:
        super().__init__()
        self.x_dim, self.coordinate_dim = x_dim, coordinate_dim
        d_model, first, second = widths
        original = geo_module.RotaryEmbedding2D
        try:
            if coordinate_dim != 2:
                geo_module.RotaryEmbedding2D = RotaryEmbeddingND
            self.network = geo_module.GeoAggregator(
                x_dims=tuple(range(x_dim)),
                spa_dims=tuple(range(x_dim, x_dim + coordinate_dim)), y_dims=(-1,),
                d_model=d_model, n_attn_layer=1, idu_points=4, attn_dropout=0.1,
                attn_variant='MCPA', dc_lin_dims=[first, second, 1])
        finally:
            geo_module.RotaryEmbedding2D = original

    def forward(self, query_x: Tensor, query_coordinates: Tensor,
                context_x: Tensor, context_coordinates: Tensor,
                neighbor_indices: Tensor, context_y: Tensor) -> Tensor:
        reference_u = context_coordinates[neighbor_indices]
        reference = torch.cat((context_x[neighbor_indices], reference_u,
                               context_y[neighbor_indices, None]), -1)
        query = torch.cat((query_x, query_coordinates,
                           torch.full_like(query_x[:, :1], torch.nan)), -1)[:, None]
        distance = torch.linalg.vector_norm(reference_u - query_coordinates[:, None], dim=-1)
        distance = torch.cat((distance, torch.zeros_like(distance[:, :1])), -1)
        return self.network(torch.cat((reference, query), 1), distance).flatten()


class MatchedGCNNWR(nn.Module):
    """Fit only train-reference grid statistics and OLS; save them as buffers."""

    def __init__(self, x_dim: int, widths: tuple) -> None:
        super().__init__()
        first, second, pool, hidden = widths
        self.network = GCNNWR(x_dim, torch.zeros(256, 2), torch.zeros(256),
            torch.ones(256), torch.zeros(x_dim + 1), grid_shape=(16, 16),
            conv_channels=(first, second), pooled_shape=(pool, pool), hidden_dim=hidden)
        self.register_buffer('reference_fitted', torch.tensor(False))

    @torch.no_grad()
    def fit_reference(self, x: Tensor, u: Tensor, y: Tensor) -> None:
        """Initialise once before the optimizer; do not refit during prediction."""
        if self.reference_fitted.item():
            return
        network = self.network
        grid = regular_spatial_grid(u, network.grid_shape)
        total = torch.zeros(len(grid), dtype=torch.float64, device=u.device)
        squares = torch.zeros_like(total)
        for start in range(0, len(u), 2048):
            distance = torch.cdist(u[start:start + 2048], grid).double()
            total += distance.sum(0)
            squares += distance.square().sum(0)
        mean = total / len(u)
        scale = (squares / len(u) - mean.square()).clamp_min(1e-12).sqrt()
        # CPU SVD handles rank-deficient frozen feature matrices; CUDA's gels
        # assumes full rank. This is the training-only OLS baseline, not a fit
        # of neural parameters, and the returned coefficients move to the GPU.
        design = torch.cat((torch.ones_like(x[:, :1]), x), 1).cpu().double()
        beta = torch.linalg.lstsq(design, y[:, None].cpu().double(), driver='gelsd').solution
        network.grid.copy_(grid)
        network.distance_mean.copy_(mean.float())
        network.distance_scale.copy_(scale.float())
        network.ols_coefficients.copy_(beta.flatten().float().to(u.device))
        self.reference_fitted.fill_(True)

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        return self.network(u, x).flatten()


def make_model(case: str, kind: str) -> nn.Module:
    """Construct a supported architecture and enforce the exact capacity."""
    spec = ARCHITECTURES[case]
    if kind == 'dlan':
        d_model, first, second = spec['dlan']
        model = GTDRDeepGWRTransformer(spec['x_dim'], spec['coordinate_dim'],
            d_model=d_model, num_heads=1, num_blocks=1, dropout=0.1,
            local_hidden_dims=(first, second))
    elif kind == 'geoaggregator':
        model = MatchedGeoAggregator(spec['x_dim'], spec['coordinate_dim'], spec['geo'])
    elif kind == 'gcnnwr' and 'gcnn' in spec:
        model = MatchedGCNNWR(spec['x_dim'], spec['gcnn'])
    else:
        raise ValueError(f'Unsupported configuration: {case}/{kind}')
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != PARAMETER_BUDGET:
        raise AssertionError(f'{case}/{kind}: {count} != {PARAMETER_BUDGET}')
    return model


class MatchedCaseTrainer(GTDRReferenceTrainer):
    """Same train-only scaling, MSE, AdamW, accumulation and early stopping."""

    def __init__(self, case: str, kind: str, config: TrainerConfig, *,
                 microbatch_size: int = 64, progress_path: str | Path | None = None) -> None:
        DeepGWRTrainer.__init__(self, make_model(case, kind), config)
        self.policy = kind
        self.candidate_count = 256 * ARCHITECTURES[case]['coordinate_dim']
        self.microbatch_size = microbatch_size
        self.progress_path = Path(progress_path) if progress_path else None

    def _move_context(self) -> None:
        super()._move_context()
        if self.policy == 'gcnnwr':
            self.model.fit_reference(self._context_x_device, self._context_u_device,
                                     self._context_y_device)

    def _forward(self, x: Tensor, u: Tensor, self_positions: Tensor | None = None) -> Tensor:
        if self.policy == 'gcnnwr':
            return self.model(x, u)
        if self.policy == 'dlan':
            bandwidths = [F.softplus(b.raw_bandwidth.detach()) + 1e-6 for b in self.model.blocks]
            neighbors = reference_candidates(u, self._context_u_device, bandwidths,
                policy='product_knn', k=self.candidate_count, self_positions=self_positions)
        else:
            ids = torch.arange(len(self._context_u_device), device=u.device)
            exclude = int(self_positions is not None and (self_positions >= 0).any())
            neighbors = build_knn(u, self._context_u_device,
                k=min(self.config.k_neighbors, len(ids) - exclude), query_ids=self_positions,
                context_ids=ids if self_positions is not None else None,
                chunk_size=self.microbatch_size)
        return self.model(query_x=x, query_coordinates=u, context_x=self._context_x_device,
            context_coordinates=self._context_u_device, context_y=self._context_y_device,
            neighbor_indices=neighbors)
