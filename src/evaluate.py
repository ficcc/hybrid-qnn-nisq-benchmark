"""Metrics, multi-seed aggregation and noise sweeps."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from .models import HybridQNN
from .noise import NoiseModel


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(X)).cpu().numpy()


def classification_metrics(y_true: torch.Tensor | np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    return classification_metrics(y, predict_proba(model, X))


def aggregate(runs: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """mean and sample std (ddof=1) per metric across seeds."""
    keys = runs[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


@torch.no_grad()
def recalibrated_metrics(
    model: HybridQNN,
    X_fit: torch.Tensor,
    y_fit: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    seed: int,
) -> dict[str, float]:
    """Keep the trained circuit fixed and refit only the classical linear head
    on features measured under the *current* noise (n+1 classical parameters).

    Rationale: local depolarizing noise mostly shrinks each <Z_i> toward 0.
    The trained head's bias then sits at the wrong scale and accuracy drops,
    although the ranking of samples (ROC-AUC) can survive. Refitting the head
    on calibration data run on the same noisy device is a cheap classical
    post-processing step, related to readout/zero-noise error mitigation.
    `X_fit` must be training data: the test split is never used for fitting.
    """
    f_fit = model.quantum_features(X_fit).cpu().numpy()
    f_test = model.quantum_features(X_test).cpu().numpy()
    # Weak regularisation: attenuated features need large weights, and a
    # strong L2 penalty would confound noise with regularisation.
    head = LogisticRegression(C=1e4, max_iter=5000, random_state=seed)
    head.fit(f_fit, np.asarray(y_fit).astype(int))
    return classification_metrics(y_test, head.predict_proba(f_test)[:, 1])


def noise_sweep(
    model: HybridQNN,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    ps: Iterable[float],
    channel: str = "depolarizing",
    *,
    shots: int | None = None,
    seed: int = 42,
    X_fit: torch.Tensor | None = None,
    y_fit: torch.Tensor | None = None,
) -> list[dict]:
    """Evaluate a *noiselessly trained* model under increasing per-layer noise.

    Each row gives the raw metrics (trained head as-is) and, if X_fit/y_fit are
    given, the metrics after head recalibration (prefixed `recal_`). The model's
    original simulator is restored afterwards.
    """
    original = (model.noise, model.shots)
    rows = []
    try:
        for p in ps:
            model.set_noise(NoiseModel(channel=channel, p=float(p)), shots=shots, seed=seed)
            row = {"channel": channel, "p": float(p), "shots": shots, **evaluate(model, X_test, y_test)}
            if X_fit is not None and y_fit is not None:
                recal = recalibrated_metrics(model, X_fit, y_fit, X_test, y_test, seed)
                row.update({f"recal_{k}": v for k, v in recal.items()})
            rows.append(row)
    finally:
        model.set_noise(*original)
    return rows
