"""Reference-support ablations of GTDRDeepGWR, without changing its network.

Retrieval is discrete (no gradient through top-k), refreshed from current
bandwidths on every microbatch. Full support uses one softmax over ALL training
references, never a separate softmax per reference chunk.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .DLAN import DeepGWRTrainer, TrainerConfig


@torch.no_grad()
def reference_candidates(
    query: Tensor,
    reference: Tensor,
    bandwidths: list[Tensor],
    *,
    policy: str,
    k: int,
    self_positions: Tensor | None = None,
) -> Tensor:
    """Exact bandwidth-metric top-k union across blocks/heads, or full support.

    Inputs use the trainer's coordinate normalization. self_positions contains
    reference-bank positions, NOT original row IDs; -1 means an external query.
    Ties use stable reference-bank order. No joint-Euclidean prefilter is used.
    """
    if policy not in {"product_knn", "full_reference"}:
        raise ValueError(f"unknown reference policy: {policy}")
    count = len(reference)
    if count < 1 or k < 1:
        raise ValueError("positive reference count and k required")
    excluded = torch.zeros((len(query), count), dtype=torch.bool, device=query.device)
    if self_positions is not None:
        if self_positions.shape != (len(query),):
            raise ValueError("one self position required per query")
        valid = self_positions >= 0
        if (self_positions[valid] >= count).any():
            raise ValueError("self position outside reference bank")
        excluded[torch.arange(len(query), device=query.device)[valid], self_positions[valid]] = True
    if excluded.all(-1).any():
        raise ValueError("no references remain after self exclusion")
    all_ids = torch.arange(count, device=query.device).expand(len(query), -1)
    if policy == "full_reference":
        return all_ids.masked_fill(excluded, -1)
    if not bandwidths:
        raise ValueError("product retrieval requires Gaussian bandwidths")
    available = int((~excluded).sum(-1).min())
    selected = torch.zeros_like(excluded)
    for bandwidth in bandwidths:
        if bandwidth.ndim != 2 or bandwidth.shape[0] != query.shape[-1]:
            raise ValueError("bandwidth must have shape [coordinate, head]")
        for head in range(bandwidth.shape[1]):
            # Direct differences avoid cancellation for repeated/nearby locations.
            distance = torch.zeros_like(excluded, dtype=query.dtype)
            for axis in range(query.shape[-1]):
                delta = query[:, axis, None] - reference[None, :, axis]
                distance.add_((delta / bandwidth[axis, head]).square())
            distance.masked_fill_(excluded, torch.inf)
            indices = distance.argsort(dim=-1, stable=True)[:, :min(k, available)]
            selected.scatter_(1, indices, True)
    # ID-sorted, deduplicated union. Padding does not enter the shared softmax.
    packed = all_ids.masked_fill(~selected, count).sort(-1).values
    width = int(selected.sum(-1).max())
    return packed[:, :width].masked_fill(packed[:, :width] == count, -1)


class GTDRReferenceTrainer(DeepGWRTrainer):
    """Same AdamW/FiLM model, changing only reference support and microbatching."""

    def __init__(
        self, model: nn.Module, config: TrainerConfig, *, policy: str,
        candidate_count: int = 512, microbatch_size: int = 8,
        progress_path: str | Path | None = None,
    ) -> None:
        super().__init__(model, config)
        if model.attention_mode != "gtdr" or not all(b.use_gaussian for b in model.blocks):
            raise ValueError("requires the Gaussian GTDR model")
        if policy not in {"product_knn", "full_reference"}:
            raise ValueError("unknown reference policy")
        if candidate_count < 1 or microbatch_size < 1:
            raise ValueError("candidate and microbatch sizes must be positive")
        self.policy = policy
        self.candidate_count = candidate_count
        self.microbatch_size = microbatch_size
        self.progress_path = Path(progress_path) if progress_path else None

    def _forward(self, x: Tensor, u: Tensor, self_positions: Tensor | None = None) -> Tensor:
        bandwidths = [(F.softplus(b.raw_bandwidth.detach()) + 1e-6) for b in self.model.blocks]
        neighbors = reference_candidates(
            u, self._context_u_device, bandwidths, policy=self.policy,
            k=self.candidate_count, self_positions=self_positions,
        )
        return self.model(
            query_x=x, query_coordinates=u, context_x=self._context_x_device,
            context_coordinates=self._context_u_device, neighbor_indices=neighbors,
            context_y=self._context_y_device if self.model.use_context_y else None,
        )

    def _move_context(self) -> None:
        self._context_x_device = self.context_x.to(self.device)
        self._context_u_device = self.context_coordinates.to(self.device)
        self._context_y_device = self.context_y.to(self.device)
        self.model.to(self.device)

    @torch.no_grad()
    def _validation_loss(self, x: Tensor, u: Tensor, y: Tensor) -> float:
        self.model.eval()
        total = 0.0
        for start in range(0, len(y), self.microbatch_size):
            batch = slice(start, start + self.microbatch_size)
            pred = self._forward(x[batch].to(self.device), u[batch].to(self.device))
            total += F.mse_loss(pred, y[batch].to(self.device), reduction="sum").item()
        return total / len(y)

    def fit(self, x: Any, coords: Any, y: Any, train_idx: Any, val_idx: Any,
            *, verbose: bool = True) -> dict[str, list[float]]:
        """Accumulate exact observation-weighted loss per effective minibatch."""
        x, u, y = self.to_tensor(x), self.to_tensor(coords), self.to_tensor(y).flatten()
        train = self.to_tensor(train_idx, torch.long)
        val = self.to_tensor(val_idx, torch.long)
        if len(train) < 2 or len(val) < 1:
            raise ValueError("need >=2 training observations and a nonempty validation set")
        if len(torch.unique(train)) != len(train) or torch.isin(train, val).any():
            raise ValueError("training IDs must be unique and disjoint from validation")
        self.x_scaler.fit(x[train])
        self.y_scaler.fit(y[train])
        self.coord_scaler.fit(u[train])
        sx, su, sy = self.x_scaler.transform(x), self.coord_scaler.transform(u), self.y_scaler.transform(y)
        self.context_x, self.context_coordinates, self.context_y = sx[train], su[train], sy[train]
        self.context_ids = train.clone()
        self._move_context()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(5, self.config.patience // 3))
        best, best_state, waiting = float("inf"), None, 0
        self.history = {"train_loss": [], "val_loss": []}
        for epoch in range(1, self.config.epochs + 1):
            epoch_started = time.perf_counter()
            self.model.train()
            permutation = torch.randperm(len(train))
            total = 0.0
            for offset in range(0, len(train), self.config.batch_size):
                effective = permutation[offset:offset + self.config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, len(effective), self.microbatch_size):
                    positions = effective[start:start + self.microbatch_size].to(self.device)
                    prediction = self._forward(self._context_x_device[positions],
                                               self._context_u_device[positions], positions)
                    loss_sum = F.mse_loss(prediction, self._context_y_device[positions], reduction="sum")
                    (loss_sum / len(effective)).backward()
                    total += loss_sum.detach().item()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                optimizer.step()
                step = offset // self.config.batch_size + 1
                steps = (len(train) + self.config.batch_size - 1) // self.config.batch_size
                if verbose and (step == 1 or step % 10 == 0 or step == steps):
                    print(f"[{self.policy}] epoch {epoch} optimizer step {step}/{steps} "
                          f"elapsed={time.perf_counter() - epoch_started:.1f}s", flush=True)
            validation = self._validation_loss(sx[val], su[val], sy[val])
            train_loss = total / len(train)
            if not torch.isfinite(torch.tensor([train_loss, validation])).all():
                raise FloatingPointError("nonfinite training/validation loss")
            scheduler.step(validation)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(validation)
            if validation < best - self.config.min_delta:
                best, best_state, waiting = validation, copy.deepcopy(self.model.state_dict()), 0
            else:
                waiting += 1
            status = dict(policy=self.policy, epoch=epoch, max_epochs=self.config.epochs,
                          train_loss=train_loss, val_loss=validation, best_val_loss=best,
                          patience_used=waiting, patience=self.config.patience,
                          microbatch_size=self.microbatch_size)
            if self.progress_path:
                self.progress_path.write_text(json.dumps(status, indent=2))
            if verbose:
                print(f"[{self.policy}] epoch {epoch}/{self.config.epochs} "
                      f"train={train_loss:.6f} val={validation:.6f} "
                      f"best={best:.6f} patience={waiting}/{self.config.patience}", flush=True)
            if waiting >= self.config.patience:
                break
        self.model.load_state_dict(best_state)
        return self.history

    @torch.no_grad()
    def predict(self, x: Any, coords: Any, query_ids: Any | None = None) -> Tensor:
        """Use restored best bandwidths for retrieval, full training bank only."""
        if self.context_x is None:
            raise RuntimeError("fit or restore a training reference bank first")
        self._move_context()
        self.model.eval()
        x = self.x_scaler.transform(self.to_tensor(x))
        u = self.coord_scaler.transform(self.to_tensor(coords))
        positions = torch.full((len(x),), -1, dtype=torch.long)
        if query_ids is not None:
            lookup = {value: i for i, value in enumerate(self.context_ids.tolist())}
            positions = torch.tensor([lookup.get(value, -1) for value in self.to_tensor(query_ids, torch.long).tolist()])
        predictions = []
        for start in range(0, len(x), self.microbatch_size):
            batch = slice(start, start + self.microbatch_size)
            predictions.append(self._forward(x[batch].to(self.device), u[batch].to(self.device),
                                             positions[batch].to(self.device)).cpu())
        return self.y_scaler.inverse_transform(torch.cat(predictions))
