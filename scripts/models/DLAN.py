"""Reusable DeepGWRTransformer from ``P19_DeepGWT.ipynb``.

The notebook prototype assumes two coordinates. This module keeps the same
architecture while allowing arbitrary coordinate dimensions for OD data.
``attention_mode="axis"`` computes independent coordinate-axis attention and
projects concatenated branch contexts back to the shared FiLM condition.
``attention_mode="gtdr"`` combines per-axis log kernels on shared candidates
before a single joint softmax, while retaining nonlinear FiLM prediction.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


class FourierSpatialEncoder(nn.Module):
    """Encode coordinates with learned projections of Fourier features."""

    def __init__(
        self,
        coordinate_dim: int,
        d_model: int,
        num_bands: int = 8,
        max_freq: float = 8.0,
    ) -> None:
        super().__init__()
        if coordinate_dim < 1:
            raise ValueError("coordinate_dim must be positive")
        self.coordinate_dim = coordinate_dim
        frequencies = torch.logspace(0, math.log10(max_freq), steps=num_bands)
        self.register_buffer("frequencies", frequencies)
        input_dim = coordinate_dim * (1 + 2 * num_bands)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, coordinates: Tensor) -> Tensor:
        if coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"expected {self.coordinate_dim} coordinates, got "
                f"{coordinates.shape[-1]}"
            )
        angle = 2 * math.pi * coordinates.unsqueeze(-1) * self.frequencies
        features = torch.cat(
            [coordinates, torch.sin(angle).flatten(-2), torch.cos(angle).flatten(-2)],
            dim=-1,
        )
        return self.mlp(features)


class RelativeGeoBias(nn.Module):
    """Map offsets, distance, and direction to per-head attention bias."""

    def __init__(
        self, coordinate_dim: int, num_heads: int, hidden_dim: int = 32
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * coordinate_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, relative_geography: Tensor) -> Tensor:
        return self.net(relative_geography).permute(0, 2, 1)


class GeoAttentionBlock(nn.Module):
    """Joint, axis-wise, or product-kernel content attention.

    Axis mode has separate Q/K/V, geometric biases and bandwidths for each
    coordinate dimension. Each branch normalizes over neighbors independently;
    only its aggregated context is mixed by the output projection. The input
    content/Fourier tokens remain shared. Joint mode preserves legacy weights.
    GTDR mode shares ID-aligned candidates, sums axis log kernels and performs
    one joint softmax per head; axis Q/K/V and output fusion remain trainable.
    """

    def __init__(
        self,
        coordinate_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        init_bandwidth: float = 1.0,
        use_gaussian: bool = True,
        attention_mode: str = "joint",
    ) -> None:
        super().__init__()
        if attention_mode not in ("joint", "axis", "gtdr"):
            raise ValueError("attention_mode must be 'joint', 'axis' or 'gtdr'")
        if coordinate_dim < 1:
            raise ValueError("coordinate_dim must be positive")
        if num_heads < 1 or d_model < 1:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.attention_mode = attention_mode
        self.coordinate_dim = coordinate_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        branch_count = coordinate_dim if attention_mode in ("axis", "gtdr") else 1
        self.q_proj = nn.Linear(d_model, branch_count * d_model)
        self.k_proj = nn.Linear(d_model, branch_count * d_model)
        self.v_proj = nn.Linear(d_model, branch_count * d_model)
        self.out_proj = nn.Linear(branch_count * d_model, d_model)
        self.geo_bias = (
            nn.ModuleList([
                RelativeGeoBias(1, num_heads) for _ in range(coordinate_dim)
            ])
            if attention_mode in ("axis", "gtdr")
            else RelativeGeoBias(coordinate_dim, num_heads)
        )
        self.use_gaussian = use_gaussian
        if use_gaussian:
            raw_init = math.log(math.expm1(init_bandwidth))
            bandwidth_shape = (
                (coordinate_dim, num_heads)
                if attention_mode in ("axis", "gtdr") else (num_heads,)
            )
            self.raw_bandwidth = nn.Parameter(torch.full(bandwidth_shape, raw_init))
        else:
            self.register_parameter("raw_bandwidth", None)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        relative_geography: Tensor,
        return_attention: bool = False,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.attention_mode == "gtdr":
            return self._forward_gtdr(
                query, key, value, relative_geography, return_attention,
                key_padding_mask,
            )
        if self.attention_mode == "axis":
            return self._forward_axis(
                query, key, value, relative_geography, return_attention
            )
        batch_size, neighbor_count, _ = key.shape
        query_heads = self.q_proj(query).view(
            batch_size, self.num_heads, self.head_dim
        )
        key_heads = self.k_proj(key).view(
            batch_size, neighbor_count, self.num_heads, self.head_dim
        )
        value_heads = self.v_proj(value).view(
            batch_size, neighbor_count, self.num_heads, self.head_dim
        )
        score = (
            query_heads[:, None, :, :] * key_heads
        ).sum(dim=-1).permute(0, 2, 1)
        score = score / math.sqrt(self.head_dim)
        score = score + self.geo_bias(relative_geography)
        if self.use_gaussian:
            distance = relative_geography[..., self.coordinate_dim]
            bandwidth = F.softplus(self.raw_bandwidth) + 1e-6
            score = score - 0.5 * (
                distance[:, None, :] / bandwidth[None, :, None]
            ).square()
        attention = self.dropout(torch.softmax(score, dim=-1))
        context = (
            attention.permute(0, 2, 1).unsqueeze(-1) * value_heads
        ).sum(dim=1)
        context = self.out_proj(context.reshape(batch_size, self.d_model))
        hidden = self.norm1(query + self.dropout(context))
        hidden = self.norm2(hidden + self.dropout(self.ffn(hidden)))
        return hidden, attention if return_attention else None

    def _forward_gtdr(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        relative_geography: Tensor,
        return_attention: bool,
        key_padding_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Product geometry on ID-aligned candidates, with a nonlinear head.

        Content logits are averaged across axis projections; geometric log
        kernels are summed (never averaged) before ONE neighbor softmax.
        The same joint attention pools each axis's own V projection, retaining
        the concatenation/output projection used by the axis architecture.
        """
        if key.ndim != 3 or value.shape != key.shape:
            raise ValueError("gtdr mode requires shared [batch, candidate, width] tokens")
        batch, count, _ = key.shape
        axes, heads, width = self.coordinate_dim, self.num_heads, self.head_dim
        if relative_geography.shape != (batch, count, 2 * axes + 1):
            raise ValueError("gtdr geography must contain all coordinate offsets")
        q = self.q_proj(query).view(batch, axes, heads, width)
        k = self.k_proj(key).view(batch, count, axes, heads, width)
        v = self.v_proj(value).view(batch, count, axes, heads, width)
        content = (q[:, None] * k).sum(-1) / math.sqrt(width)
        score = content.mean(dim=2).transpose(1, 2)
        delta = relative_geography[..., :axes]
        for axis in range(axes):
            offset = delta[..., axis]
            features = torch.stack((offset, offset.abs(), offset.sign()), dim=-1)
            score = score + self.geo_bias[axis](features)
        if self.use_gaussian:
            bandwidth = F.softplus(self.raw_bandwidth) + 1e-6
            scaled = delta[:, :, :, None] / bandwidth[None, None]
            score = score - 0.5 * scaled.square().sum(dim=2).transpose(1, 2)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, count):
                raise ValueError("key_padding_mask must have shape [batch, candidate]")
            if key_padding_mask.all(dim=-1).any():
                raise ValueError("each query requires at least one valid candidate")
            score = score.masked_fill(key_padding_mask[:, None], float("-inf"))
        attention = self.dropout(score.softmax(dim=-1))
        context = torch.einsum("bhk,bkahw->bahw", attention, v)
        context = self.out_proj(context.reshape(batch, axes * self.d_model))
        hidden = self.norm1(query + self.dropout(context))
        hidden = self.norm2(hidden + self.dropout(self.ffn(hidden)))
        return hidden, attention if return_attention else None

    def _forward_axis(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        relative_geography: Tensor,
        return_attention: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute B×axis×head×neighbor scores, then project axis contexts."""
        axes, heads, width = self.coordinate_dim, self.num_heads, self.head_dim
        batch_size = len(query)
        query_heads = self.q_proj(query).view(batch_size, axes, heads, width)
        if key.ndim == 4:
            # Branch a may only project the tokens gathered by axis a's KNN.
            if key.shape[1] != axes or value.shape != key.shape:
                raise ValueError("axis reference tokens must have shape [batch, axis, k, d_model]")
            neighbor_count = key.shape[2]
            key_projected = torch.einsum(
                "baki,aoi->bako", key,
                self.k_proj.weight.view(axes, self.d_model, self.d_model),
            ) + self.k_proj.bias.view(1, axes, 1, self.d_model)
            value_projected = torch.einsum(
                "baki,aoi->bako", value,
                self.v_proj.weight.view(axes, self.d_model, self.d_model),
            ) + self.v_proj.bias.view(1, axes, 1, self.d_model)
            key_heads = key_projected.view(batch_size, axes, neighbor_count, heads, width).transpose(2, 3)
            value_heads = value_projected.view(batch_size, axes, neighbor_count, heads, width).transpose(2, 3)
        else:
            # Low-level compatibility for callers explicitly supplying a shared
            # candidate set; model/trainer axis mode requires per-axis indices.
            neighbor_count = key.shape[1]
            key_heads = self.k_proj(key).view(
                batch_size, neighbor_count, axes, heads, width
            ).permute(0, 2, 3, 1, 4)
            value_heads = self.v_proj(value).view(
                batch_size, neighbor_count, axes, heads, width
            ).permute(0, 2, 3, 1, 4)
        score = (query_heads.unsqueeze(-2) * key_heads).sum(-1)
        score = score / math.sqrt(width)

        # Do not use the joint norm or its direction components: either would
        # couple an axis's geometric score to the magnitudes of other axes.
        if relative_geography.ndim == 4:
            if relative_geography.shape != (batch_size, axes, neighbor_count, 3):
                raise ValueError("axis geography must have shape [batch, axis, k, 3]")
            axis_geography = relative_geography
        else:
            delta = relative_geography[..., :axes].transpose(1, 2)
            axis_geography = torch.stack((delta, delta.abs(), delta.sign()), dim=-1)
        bias = torch.stack([
            self.geo_bias[axis](axis_geography[:, axis])
            for axis in range(axes)
        ], dim=1)
        score = score + bias
        if self.use_gaussian:
            bandwidth = F.softplus(self.raw_bandwidth) + 1e-6
            distance = axis_geography[..., 1].unsqueeze(2)
            score = score - 0.5 * (
                distance / bandwidth[None, :, :, None]
            ).square()
        attention = self.dropout(torch.softmax(score, dim=-1))
        # Concatenation preserves separate axis channels until W_out merges
        # them. Averaging logits before softmax would not implement this model.
        context = (attention.unsqueeze(-1) * value_heads).sum(-2)
        context = self.out_proj(context.reshape(batch_size, axes * self.d_model))
        hidden = self.norm1(query + self.dropout(context))
        hidden = self.norm2(hidden + self.dropout(self.ffn(hidden)))
        return hidden, attention if return_attention else None


class FiLMLocalPredictor(nn.Module):
    """Shared nonlinear predictor modulated by a location condition."""

    def __init__(
        self,
        x_dim: int,
        condition_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        current_dim = x_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            film = nn.Linear(condition_dim, 2 * hidden_dim)
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
            self.film_layers.append(film)
            current_dim = hidden_dim
        self.output_layer = nn.Linear(current_dim, 1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        hidden = x
        for layer, film in zip(self.layers, self.film_layers):
            hidden = layer(hidden)
            gamma, beta = film(condition).chunk(2, dim=-1)
            hidden = F.gelu((1 + torch.tanh(gamma)) * hidden + beta)
            hidden = self.dropout(hidden)
        return self.output_layer(hidden).squeeze(-1)


class DeepGWRTransformer(nn.Module):
    """KNN transformer with nonlinear FiLM and selectable geometry.

    ``joint`` remains the default so existing notebooks, parameter budgets and
    historical checkpoints do not silently change architecture.
    ``gtdr`` uses a product-distance prior over deduplicated axis candidates.
    """

    def __init__(
        self,
        x_dim: int,
        coordinate_dim: int = 2,
        d_model: int = 16,
        num_heads: int = 1,
        num_blocks: int = 1,
        dropout: float = 0.1,
        use_context_y: bool = True,
        local_hidden_dims: tuple[int, ...] = (8, 8),
        init_bandwidth: float = 0.1,
        use_gaussian: bool = True,
        attention_mode: str = "joint",
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.coordinate_dim = coordinate_dim
        self.use_context_y = use_context_y
        self.attention_mode = attention_mode
        if attention_mode == "axis":
            # Reject old shared-candidate axis checkpoints under strict loading.
            self.register_buffer("_axis_knn_version", torch.tensor(1))
        elif attention_mode == "gtdr":
            self.register_buffer("_gtdr_product_version", torch.tensor(1))
        self.feature_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.spatial_encoder = FourierSpatialEncoder(coordinate_dim, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.y_encoder = (
            nn.Sequential(
                nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
            if use_context_y
            else None
        )
        self.blocks = nn.ModuleList(
            [
                GeoAttentionBlock(
                    coordinate_dim,
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    init_bandwidth=init_bandwidth,
                    use_gaussian=use_gaussian,
                    attention_mode=attention_mode,
                )
                for _ in range(num_blocks)
            ]
        )
        self.local_predictor = FiLMLocalPredictor(
            x_dim, d_model, hidden_dims=local_hidden_dims, dropout=dropout
        )

    @staticmethod
    def build_relative_geography(
        query_coordinates: Tensor, neighbor_coordinates: Tensor
    ) -> Tensor:
        if neighbor_coordinates.ndim == 4:
            axes = query_coordinates.shape[-1]
            if neighbor_coordinates.shape[1] != axes or neighbor_coordinates.shape[-1] != axes:
                raise ValueError("axis reference coordinates must have shape [batch, axis, k, axis]")
            selected = neighbor_coordinates.diagonal(dim1=1, dim2=3).transpose(1, 2)
            delta = selected - query_coordinates[:, :, None]
            return torch.stack((delta, delta.abs(), delta.sign()), dim=-1)
        delta = neighbor_coordinates - query_coordinates[:, None, :]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        direction = delta / (distance + 1e-6)
        return torch.cat([delta, distance, direction], dim=-1)

    def forward(
        self,
        query_x: Tensor,
        query_coordinates: Tensor,
        context_x: Tensor,
        context_coordinates: Tensor,
        neighbor_indices: Tensor,
        context_y: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, list[Tensor]]]:
        expected_rank = 3 if self.attention_mode == "axis" else 2
        if neighbor_indices.ndim != expected_rank:
            raise ValueError("axis mode requires [batch, axis, k] neighbor_indices; joint mode requires [batch, k]")
        if self.attention_mode == "axis" and neighbor_indices.shape[1] != self.coordinate_dim:
            raise ValueError("one independently selected neighborhood is required per coordinate axis")
        padding_mask = None
        if self.attention_mode == "gtdr":
            neighbor_indices, padding_mask = _prepare_gtdr_indices(
                neighbor_indices, len(context_x)
            )
        query_token = self.token_norm(
            self.feature_encoder(query_x) + self.spatial_encoder(query_coordinates)
        )
        neighbor_x = context_x[neighbor_indices]
        neighbor_coordinates = context_coordinates[neighbor_indices]
        key_token = self.token_norm(
            self.feature_encoder(neighbor_x)
            + self.spatial_encoder(neighbor_coordinates)
        )
        value_token = key_token
        if self.use_context_y:
            if context_y is None or self.y_encoder is None:
                raise ValueError("context_y is required when use_context_y=True")
            value_token = value_token + self.y_encoder(
                context_y[neighbor_indices].unsqueeze(-1)
            )
        relative_geography = self.build_relative_geography(
            query_coordinates, neighbor_coordinates
        )
        hidden = query_token
        attention_maps = []
        for block in self.blocks:
            hidden, attention = block(
                hidden,
                key_token,
                value_token,
                relative_geography,
                return_attention,
                key_padding_mask=padding_mask,
            )
            if attention is not None:
                attention_maps.append(attention)
        prediction = self.local_predictor(query_x, hidden)
        if not return_attention:
            return prediction
        return prediction, {
            "attention": attention_maps,
            "bandwidths": [
                F.softplus(block.raw_bandwidth)
                for block in self.blocks
                if block.raw_bandwidth is not None
            ],
        }


class AxisDeepGWRTransformer(DeepGWRTransformer):
    """Axis-distance attention and projection fusion, retaining the FiLM head.

    Use this entrypoint for new axis-wise experiments. The original class keeps
    its joint default for reproducibility of historical notebook configurations.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop("attention_mode", "axis") != "axis":
            raise ValueError("AxisDeepGWRTransformer requires attention_mode='axis'")
        super().__init__(*args, attention_mode="axis", **kwargs)


class GTDRDeepGWRTransformer(DeepGWRTransformer):
    """GTDR product-distance prior with content attention and nonlinear FiLM."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop("attention_mode", "gtdr") != "gtdr":
            raise ValueError("GTDRDeepGWRTransformer requires attention_mode='gtdr'")
        super().__init__(*args, attention_mode="gtdr", **kwargs)


def _prepare_gtdr_indices(indices: Tensor, context_size: int) -> tuple[Tensor, Tensor]:
    """Validate shared candidates and safely gather around -1 padding."""
    if indices.ndim != 2 or indices.dtype != torch.long:
        raise ValueError("gtdr indices must be a [query, candidate] long tensor")
    if (indices < -1).any() or (indices >= context_size).any():
        raise ValueError("gtdr candidate index is out of range")
    padding = indices == -1
    if padding.all(dim=-1).any():
        raise ValueError("each query requires at least one valid candidate")
    return indices.clamp_min(0), padding


class Standardizer:
    """Column-wise standardizer fitted on training observations only."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.mean: Tensor | None = None
        self.std: Tensor | None = None
        self.eps = eps

    def fit(self, values: Tensor) -> "Standardizer":
        self.mean = values.mean(dim=0)
        self.std = values.std(dim=0, unbiased=False).clamp_min(self.eps)
        return self

    def transform(self, values: Tensor) -> Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("fit() must be called before transform().")
        return (values - self.mean) / self.std

    def inverse_transform(self, values: Tensor) -> Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("fit() must be called before inverse_transform().")
        return values * self.std + self.mean


class CoordinateNormalizer:
    """Normalize every coordinate axis with one shared spatial scale."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.center: Tensor | None = None
        self.scale: Tensor | None = None
        self.eps = eps

    def fit(self, coordinates: Tensor) -> "CoordinateNormalizer":
        self.center = coordinates.mean(dim=0)
        axis_std = coordinates.std(dim=0, unbiased=False)
        self.scale = axis_std.mean().clamp_min(self.eps)
        return self

    def transform(self, coordinates: Tensor) -> Tensor:
        if self.center is None or self.scale is None:
            raise RuntimeError("fit() must be called before transform().")
        return (coordinates - self.center) / self.scale


def build_knn(
    query_coordinates: Tensor,
    context_coordinates: Tensor,
    k: int,
    query_ids: Tensor | None = None,
    context_ids: Tensor | None = None,
    chunk_size: int = 2048,
) -> Tensor:
    """Return KNN context indices, optionally excluding identical row IDs."""

    if query_coordinates.ndim != 2 or context_coordinates.ndim != 2:
        raise ValueError("query_coordinates and context_coordinates must be matrices")
    if query_coordinates.shape[1] != context_coordinates.shape[1]:
        raise ValueError("query and context coordinate dimensions must match")
    if not 1 <= k <= len(context_coordinates):
        raise ValueError("k must be between 1 and the number of context rows")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if (query_ids is None) != (context_ids is None):
        raise ValueError("query_ids and context_ids must be provided together")
    if query_ids is not None and len(query_ids) != len(query_coordinates):
        raise ValueError("query_ids length must match query_coordinates")
    if context_ids is not None and len(context_ids) != len(context_coordinates):
        raise ValueError("context_ids length must match context_coordinates")

    results = []
    for start in range(0, len(query_coordinates), chunk_size):
        end = min(start + chunk_size, len(query_coordinates))
        distance = torch.cdist(query_coordinates[start:end], context_coordinates)
        if query_ids is not None and context_ids is not None:
            same_point = query_ids[start:end, None] == context_ids[None, :]
            distance = distance.masked_fill(same_point, float("inf"))
        results.append(torch.topk(distance, k=k, dim=1, largest=False).indices)
    return torch.cat(results, dim=0)


def build_axis_knn(
    query_coordinates: Tensor,
    context_coordinates: Tensor,
    k: int,
    query_ids: Tensor | None = None,
    context_ids: Tensor | None = None,
    chunk_size: int = 2048,
) -> Tensor:
    """Exact 1D KNN for each axis, returning [query, axis, k] indices.

    Each search uses only abs(query[a] - context[a]); dimensions are never
    combined, unioned or deduplicated across branches. Row IDs exclude all
    matching references, including repeated IDs. One-dimensional KD trees keep
    large datasets from allocating query-by-reference distance matrices.
    """
    from sklearn.neighbors import NearestNeighbors

    if query_coordinates.ndim != 2 or context_coordinates.ndim != 2:
        raise ValueError("coordinates must be matrices")
    if query_coordinates.shape[1] != context_coordinates.shape[1] or query_coordinates.shape[1] < 1:
        raise ValueError("coordinate dimensions must match and be positive")
    if not 1 <= k <= len(context_coordinates) or chunk_size < 1:
        raise ValueError("k must fit the reference count and chunk_size must be positive")
    if (query_ids is None) != (context_ids is None):
        raise ValueError("query_ids and context_ids must be provided together")
    query = query_coordinates.detach().cpu().numpy()
    context = context_coordinates.detach().cpu().numpy()
    if not np.isfinite(query).all() or not np.isfinite(context).all():
        raise ValueError("coordinates must be finite")
    qids = cids = None
    excluded = 0
    if query_ids is not None:
        if query_ids.shape != (len(query),) or context_ids.shape != (len(context),):
            raise ValueError("row IDs must be one-dimensional and match coordinate rows")
        qids = query_ids.detach().cpu().numpy()
        cids = context_ids.detach().cpu().numpy()
        values, counts = np.unique(cids, return_counts=True)
        count_by_id = dict(zip(values.tolist(), counts.tolist()))
        excluded = max((count_by_id.get(row, 0) for row in qids.tolist()), default=0)
        if len(context) - excluded < k:
            raise ValueError("fewer than k references remain after self exclusion")
    output = np.empty((len(query), query.shape[1], k), dtype=np.int64)
    for axis in range(query.shape[1]):
        search = NearestNeighbors(algorithm="kd_tree").fit(context[:, axis:axis + 1])
        for start in range(0, len(query), chunk_size):
            stop = min(start + chunk_size, len(query))
            candidates = search.kneighbors(
                query[start:stop, axis:axis + 1], n_neighbors=k + excluded,
                return_distance=False,
            )
            if qids is None:
                output[start:stop, axis] = candidates
            else:
                for offset, references in enumerate(candidates):
                    keep = references[cids[references] != qids[start + offset]]
                    output[start + offset, axis] = keep[:k]
    return torch.from_numpy(output).to(query_coordinates.device)


def merge_axis_neighbors(indices: Tensor) -> Tensor:
    """Deduplicate per-axis reference IDs; pad variable-length unions with -1.

    IDs are sorted for deterministic alignment, not ranked by a joint distance.
    Each valid reference occurs once, preventing duplicate kernel mass.
    """
    if indices.ndim != 3 or indices.dtype != torch.long:
        raise ValueError("axis indices must be a [query, axis, k] long tensor")
    rows = [np.unique(row) for row in indices.detach().cpu().numpy()]
    count = max((len(row) for row in rows), default=0)
    merged = np.full((len(rows), count), -1, dtype=np.int64)
    for position, row in enumerate(rows):
        merged[position, :len(row)] = row
    return torch.from_numpy(merged).to(indices.device)


def build_gtdr_knn(
    query_coordinates: Tensor,
    context_coordinates: Tensor,
    k: int,
    query_ids: Tensor | None = None,
    context_ids: Tensor | None = None,
    chunk_size: int = 2048,
) -> Tensor:
    """ID-aligned union of exact per-axis KNN, with no joint prefilter.

    K controls candidate screening, NOT a GTDR adaptive kernel bandwidth.
    Product weights are evaluated on every axis for every union candidate.
    Using K equal to the available reference count recovers full support.
    """
    return merge_axis_neighbors(build_axis_knn(
        query_coordinates, context_coordinates, k,
        query_ids=query_ids, context_ids=context_ids, chunk_size=chunk_size,
    ))


def _neighbor_builder(attention_mode: str) -> Callable[..., Tensor]:
    """Select the reference policy without changing legacy joint/axis modes."""
    return {"axis": build_axis_knn, "gtdr": build_gtdr_knn}.get(
        attention_mode, build_knn
    )


@dataclass
class TrainerConfig:
    """Optimization and neighborhood settings for :class:`DeepGWRTrainer`."""

    k_neighbors: int = 256
    batch_size: int = 256
    epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30
    min_delta: float = 1e-5
    grad_clip: float = 5.0
    knn_chunk_size: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DeepGWRTrainer:
    """Fit, predict, and evaluate ``DeepGWRTransformer`` models.

    Feature, response, and coordinate transformations are estimated from the
    training split only. Training neighborhoods exclude the query observation;
    validation and prediction neighborhoods draw only from the training set.
    """

    def __init__(
        self,
        model: DeepGWRTransformer,
        config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config if config is not None else TrainerConfig()
        self.device = torch.device(self.config.device)
        self.x_scaler = Standardizer()
        self.y_scaler = Standardizer()
        self.coord_scaler = CoordinateNormalizer()
        self.context_x: Tensor | None = None
        self.context_coordinates: Tensor | None = None
        self.context_y: Tensor | None = None
        self.context_ids: Tensor | None = None
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    @property
    def context_coord(self) -> Tensor | None:
        """Notebook-compatible alias for the fitted context coordinates."""

        return self.context_coordinates

    @staticmethod
    def to_tensor(values: Any, dtype: torch.dtype = torch.float32) -> Tensor:
        """Create a detached CPU tensor with a predictable dtype."""

        if isinstance(values, Tensor):
            return values.detach().cpu().to(dtype)
        if isinstance(values, np.ndarray) and not values.flags.writeable:
            values = values.copy()
        return torch.as_tensor(values, dtype=dtype)

    def make_loader(
        self,
        x: Tensor,
        coordinates: Tensor,
        y: Tensor,
        neighbor_indices: Tensor,
        *,
        shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(x, coordinates, y, neighbor_indices)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: DataLoader,
        context_x: Tensor,
        context_coordinates: Tensor,
        context_y: Tensor | None,
    ) -> float:
        """Return mean standardized MSE for a prepared data loader."""

        self.model.eval()
        total_loss = 0.0
        observation_count = 0
        for x, coordinates, y, neighbor_indices in loader:
            x = x.to(self.device)
            coordinates = coordinates.to(self.device)
            y = y.to(self.device)
            neighbor_indices = neighbor_indices.to(self.device)
            prediction = self.model(
                query_x=x,
                query_coordinates=coordinates,
                context_x=context_x,
                context_coordinates=context_coordinates,
                neighbor_indices=neighbor_indices,
                context_y=context_y,
            )
            total_loss += F.mse_loss(prediction, y, reduction="sum").item()
            observation_count += y.numel()
        return total_loss / observation_count

    def fit(
        self,
        x: Any,
        coords: Any,
        y: Any,
        train_idx: Any,
        val_idx: Any,
        *,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Fit the model with early stopping and restore its best state."""

        x = self.to_tensor(x)
        coordinates = self.to_tensor(coords)
        y = self.to_tensor(y).flatten()
        train_indices = self.to_tensor(train_idx, torch.long)
        validation_indices = self.to_tensor(val_idx, torch.long)
        if len(train_indices) <= 1:
            raise ValueError("training set must contain at least two observations")
        if len(validation_indices) == 0:
            raise ValueError("validation set must contain at least one observation")
        if x.ndim != 2 or coordinates.ndim != 2:
            raise ValueError("x and coordinates must be matrices")
        if len(x) != len(coordinates) or len(x) != len(y):
            raise ValueError("x, coordinates, and y must contain the same rows")
        if x.shape[1] != self.model.x_dim:
            raise ValueError("x columns do not match the model x_dim")
        if coordinates.shape[1] != self.model.coordinate_dim:
            raise ValueError("coordinate columns do not match the model")

        self.x_scaler.fit(x[train_indices])
        self.y_scaler.fit(y[train_indices])
        self.coord_scaler.fit(coordinates[train_indices])
        scaled_x = self.x_scaler.transform(x)
        scaled_y = self.y_scaler.transform(y)
        scaled_coordinates = self.coord_scaler.transform(coordinates)

        context_x = scaled_x[train_indices]
        context_coordinates = scaled_coordinates[train_indices]
        context_y = scaled_y[train_indices]
        train_k = min(self.config.k_neighbors, len(train_indices) - 1)
        validation_k = min(self.config.k_neighbors, len(train_indices))
        neighbor_builder = _neighbor_builder(getattr(self.model, "attention_mode", "joint"))
        train_neighbor_indices = neighbor_builder(
            scaled_coordinates[train_indices],
            context_coordinates,
            k=train_k,
            query_ids=train_indices,
            context_ids=train_indices,
            chunk_size=self.config.knn_chunk_size,
        )
        validation_neighbor_indices = neighbor_builder(
            scaled_coordinates[validation_indices],
            context_coordinates,
            k=validation_k,
            chunk_size=self.config.knn_chunk_size,
        )
        train_loader = self.make_loader(
            scaled_x[train_indices],
            scaled_coordinates[train_indices],
            scaled_y[train_indices],
            train_neighbor_indices,
            shuffle=True,
        )
        validation_loader = self.make_loader(
            scaled_x[validation_indices],
            scaled_coordinates[validation_indices],
            scaled_y[validation_indices],
            validation_neighbor_indices,
            shuffle=False,
        )

        self.context_x = context_x
        self.context_coordinates = context_coordinates
        self.context_y = context_y
        self.context_ids = train_indices.clone()
        self.history = {"train_loss": [], "val_loss": []}

        self.model.to(self.device)
        context_x_device = context_x.to(self.device)
        context_coordinates_device = context_coordinates.to(self.device)
        context_y_device = (
            context_y.to(self.device) if self.model.use_context_y else None
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(5, self.config.patience // 3),
        )
        best_validation_loss = float("inf")
        best_state: dict[str, Tensor] | None = None
        waiting = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0
            observation_count = 0
            for x_batch, coordinates_batch, y_batch, neighbors_batch in train_loader:
                x_batch = x_batch.to(self.device)
                coordinates_batch = coordinates_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                neighbors_batch = neighbors_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(
                    query_x=x_batch,
                    query_coordinates=coordinates_batch,
                    context_x=context_x_device,
                    context_coordinates=context_coordinates_device,
                    neighbor_indices=neighbors_batch,
                    context_y=context_y_device,
                )
                loss = F.mse_loss(prediction, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                optimizer.step()
                total_loss += loss.item() * y_batch.numel()
                observation_count += y_batch.numel()

            train_loss = total_loss / observation_count
            validation_loss = self.evaluate_loader(
                validation_loader,
                context_x_device,
                context_coordinates_device,
                context_y_device,
            )
            scheduler.step(validation_loss)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(validation_loss)

            improved = validation_loss < best_validation_loss - self.config.min_delta
            if improved:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                waiting = 0
            else:
                waiting += 1

            if verbose and (epoch == 1 or epoch % 10 == 0 or improved):
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:03d} | train={train_loss:.6f} | "
                    f"val={validation_loss:.6f} | lr={current_lr:.2e}"
                )
            if waiting >= self.config.patience:
                if verbose:
                    print(
                        "Early stopping | "
                        f"best val={best_validation_loss:.6f}"
                    )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    @torch.no_grad()
    def predict(
        self,
        x: Any,
        coords: Any,
        query_ids: Any | None = None,
    ) -> Tensor:
        """Predict on the original response scale using fitted context."""

        if (
            self.context_x is None
            or self.context_coordinates is None
            or self.context_y is None
        ):
            raise RuntimeError("fit() must be called before predict().")
        x = self.to_tensor(x)
        coordinates = self.to_tensor(coords)
        scaled_x = self.x_scaler.transform(x)
        scaled_coordinates = self.coord_scaler.transform(coordinates)
        query_id_tensor = (
            self.to_tensor(query_ids, torch.long) if query_ids is not None else None
        )
        available = len(self.context_coordinates)
        if query_id_tensor is not None and self.context_ids is not None:
            values, counts = torch.unique(self.context_ids, return_counts=True)
            counts_by_id = dict(zip(values.tolist(), counts.tolist()))
            available -= max((counts_by_id.get(row, 0) for row in query_id_tensor.tolist()), default=0)
        if available < 1:
            raise ValueError("no references remain after self exclusion")
        k = min(self.config.k_neighbors, available)
        neighbor_builder = _neighbor_builder(getattr(self.model, "attention_mode", "joint"))
        neighbor_indices = neighbor_builder(
            scaled_coordinates,
            self.context_coordinates,
            k=k,
            query_ids=query_id_tensor,
            context_ids=self.context_ids if query_id_tensor is not None else None,
            chunk_size=self.config.knn_chunk_size,
        )
        loader = DataLoader(
            TensorDataset(scaled_x, scaled_coordinates, neighbor_indices),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        self.model.eval()
        context_x = self.context_x.to(self.device)
        context_coordinates = self.context_coordinates.to(self.device)
        context_y = (
            self.context_y.to(self.device) if self.model.use_context_y else None
        )
        predictions = []
        for x_batch, coordinates_batch, neighbors_batch in loader:
            prediction = self.model(
                query_x=x_batch.to(self.device),
                query_coordinates=coordinates_batch.to(self.device),
                context_x=context_x,
                context_coordinates=context_coordinates,
                neighbor_indices=neighbors_batch.to(self.device),
                context_y=context_y,
            )
            predictions.append(prediction.cpu())
        scaled_prediction = torch.cat(predictions)
        return self.y_scaler.inverse_transform(scaled_prediction)

    @torch.no_grad()
    def evaluate(
        self,
        x: Any,
        coords: Any,
        y: Any,
        query_ids: Any | None = None,
    ) -> dict[str, float]:
        """Return MAE, RMSE, and R2 on the original response scale."""

        y_true = self.to_tensor(y).flatten()
        y_prediction = self.predict(x, coords, query_ids=query_ids).flatten()
        error = y_prediction - y_true
        squared_error = error.square()
        ss_total = (y_true - y_true.mean()).square().sum()
        return {
            "MAE": error.abs().mean().item(),
            "RMSE": torch.sqrt(squared_error.mean()).item(),
            "R2": (1 - squared_error.sum() / ss_total.clamp_min(1e-12)).item(),
        }

# Public article name; state-dict keys and numerical operations are unchanged.
DLAN = GTDRDeepGWRTransformer
