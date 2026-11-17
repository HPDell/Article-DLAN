"""Controlled baselines using pinned zjuwss/gnnwr SWNN and STPNN code.

Preserve upstream's coefficient-wise OLS modulation, not a generic neural
regressor. Fixed reference coordinates use only training rows; they never carry
query labels. Full-reference defaults cannot fit the small matched budgets, so
the supported Reference option is represented by training-only landmarks.
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import Tensor, nn

from .DLAN import DeepGWRTrainer, TrainerConfig
from .training import GTDRReferenceTrainer
# Load only the installed library's network module: no bundled author source.
import importlib.metadata
import importlib.util
_network_path = importlib.metadata.distribution("gnnwr").locate_file("gnnwr/networks.py")
_network_spec = importlib.util.spec_from_file_location("_article_gnnwr_networks", _network_path)
_network_module = importlib.util.module_from_spec(_network_spec)
_network_spec.loader.exec_module(_network_module)
STPNN, SWNN = _network_module.STPNN, _network_module.SWNN


ARCHITECTURES = {
    'CO2': dict(kind='gtnnwr', x_dim=8, coordinate_dim=3, references=128,
                hidden=(40, 32), proximity_hidden=(7,), parameters=6957),
    'OD-Wuhan': dict(kind='gnnwr', x_dim=35, coordinate_dim=4, references=117,
                     hidden=(41, 25), parameters=6957),
    'WuhanHP': dict(kind='gnnwr', x_dim=17, coordinate_dim=2, references=134,
                    hidden=(44, 14), parameters=6957),
    'GVI': dict(kind='gnnwr', x_dim=176, coordinate_dim=2, references=134,
                hidden=(31, 12), parameters=6957),
    'OD-Wuhan-Attention': dict(kind='gnnwr', x_dim=35, coordinate_dim=4, references=119,
                               hidden=(51, 9), parameters=7069),
}


class UpstreamWeightedRegressor(nn.Module):
    """Original library networks, plus frozen OLS and train-fitted geometry."""

    def __init__(self, case: str) -> None:
        super().__init__()
        spec = ARCHITECTURES[case]
        self.spec = spec
        self.kind = spec['kind']
        self.x_dim, self.coordinate_dim = spec['x_dim'], spec['coordinate_dim']
        self.network = SWNN(list(spec['hidden']), spec['references'], self.x_dim + 1,
                            drop_out=0.1, activate_func=nn.PReLU(init=0.4), batch_norm=True)
        if self.kind == 'gtnnwr':
            self.network = nn.Sequential(
                STPNN(list(spec['proximity_hidden']), 2, 1, drop_out=0.1, batch_norm=True),
                self.network)
        self.register_buffer('reference_coordinates', torch.zeros(spec['references'], self.coordinate_dim))
        self.register_buffer('reference_positions', torch.zeros(spec['references'], dtype=torch.long))
        distance_shape = 2 if self.kind == 'gtnnwr' else spec['references']
        self.register_buffer('distance_mean', torch.zeros(distance_shape))
        self.register_buffer('distance_scale', torch.ones(distance_shape))
        self.register_buffer('ols_coefficients', torch.zeros(self.x_dim + 1))
        self.register_buffer('reference_fitted', torch.tensor(False))
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert count == spec['parameters'], (count, spec)

    def distance_features(self, u: Tensor) -> Tensor:
        """GTNNWR receives (spatial Euclidean distance, absolute time gap)."""
        if self.kind == 'gtnnwr':
            spatial = torch.cdist(u[:, :2], self.reference_coordinates[:, :2])
            temporal = (u[:, 2, None] - self.reference_coordinates[None, :, 2]).abs()
            return torch.stack((spatial, temporal), -1)
        # scipy.cdist in the upstream BasicDistance supports D dimensions:
        # WuhanOD retains all four origin/destination coordinates, not a midpoint.
        return torch.cdist(u, self.reference_coordinates)

    @torch.no_grad()
    def fit_reference(self, x: Tensor, u: Tensor, y: Tensor) -> None:
        """Same linear-regression/intercept-last convention as upstream OLS."""
        if self.reference_fitted.item():
            return
        count = self.spec['references']
        if len(u) < count:
            raise ValueError(f'Need at least {count} training rows for reference landmarks')
        # Deterministic coordinate-only farthest-point coverage; seed-independent
        # references. Selection is not the learned GTNNWR space-time metric.
        coordinates = u.detach().cpu().double()
        selected = []
        nearest = torch.full((len(u),), torch.inf, dtype=torch.float64)
        next_id = int(((coordinates - coordinates.mean(0))**2).sum(1).argmin())
        for _ in range(count):
            selected.append(next_id)
            delta = (coordinates - coordinates[next_id]).square().sum(1)
            nearest = torch.minimum(nearest, delta)
            nearest[selected] = -1
            next_id = int(nearest.argmax())
        positions = torch.tensor(selected, dtype=torch.long, device=u.device)
        self.reference_positions.copy_(positions)
        self.reference_coordinates.copy_(u[positions])
        # Library distance standardisation: GNNWR per reference column;
        # GTNNWR per distance channel over training-row/reference pairs.
        values = self.distance_features(u)
        dims = (0, 1) if self.kind == 'gtnnwr' else (0,)
        self.distance_mean.copy_(values.double().mean(dims).float())
        scale = values.double().std(dims, unbiased=False)
        self.distance_scale.copy_(torch.where(scale < 1e-12, 1.0, scale).float())
        ols = LinearRegression(fit_intercept=True).fit(x.cpu().double().numpy(), y.cpu().double().numpy())
        beta = np.r_[ols.coef_, ols.intercept_]
        self.ols_coefficients.copy_(torch.as_tensor(beta, dtype=torch.float32, device=u.device))
        self.reference_fitted.fill_(True)

    def forward(self, x: Tensor, u: Tensor) -> Tensor:
        distance = (self.distance_features(u) - self.distance_mean) / self.distance_scale
        multipliers = self.network(distance)
        design = torch.cat((x, torch.ones_like(x[:, :1])), -1)
        return (multipliers * design * self.ols_coefficients).sum(-1)


class UpstreamGNNWRTrainer(GTDRReferenceTrainer):
    """Keep previous AdamW/MSE/early-stopping protocol for controlled scoring."""

    def __init__(self, case: str, config: TrainerConfig, *, microbatch_size: int = 64,
                 progress_path: str | Path | None = None) -> None:
        DeepGWRTrainer.__init__(self, UpstreamWeightedRegressor(case), config)
        self.policy = ARCHITECTURES[case]['kind']
        self.microbatch_size = microbatch_size
        self.progress_path = Path(progress_path) if progress_path else None

    def _move_context(self) -> None:
        super()._move_context()
        self.model.fit_reference(self._context_x_device, self._context_u_device,
                                 self._context_y_device)

    def _forward(self, x: Tensor, u: Tensor, self_positions: Tensor | None = None) -> Tensor:
        # A training reference's zero self-distance is part of the original
        # geometry-only input; it carries no response and does not leak y.
        return self.model(x, u)
