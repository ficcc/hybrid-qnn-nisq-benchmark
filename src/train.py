"""Mini-batch training with validation-based early stopping (test set never touched)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DataSplits


@dataclass
class TrainResult:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_epoch: int = -1
    train_time_s: float = 0.0


def train_model(
    model: nn.Module,
    splits: DataSplits,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
    verbose: bool = False,
) -> TrainResult:
    """Adam + BCEWithLogits. Restores the weights with the lowest validation loss."""
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(splits.X_train, splits.y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    result = TrainResult()
    best_val, best_state, stale = float("inf"), None, 0
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running, n_seen = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            n_seen += len(xb)

        model.eval()
        with torch.no_grad():
            val_logits = model(splits.X_val)
            val_loss = loss_fn(val_logits, splits.y_val).item()
            val_acc = ((val_logits > 0).to(splits.y_val.dtype) == splits.y_val).double().mean().item()

        result.train_loss.append(running / n_seen)
        result.val_loss.append(val_loss)
        result.val_acc.append(val_acc)
        if verbose:
            print(f"  epoch {epoch:3d}  train {running / n_seen:.4f}  val {val_loss:.4f}  val_acc {val_acc:.3f}")

        if val_loss < best_val - 1e-6:
            best_val, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            result.best_epoch = epoch
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    result.train_time_s = time.perf_counter() - t0
    return result
