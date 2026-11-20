"""Leakage-free data loading: split first, then fit every transform on train only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _breast_cancer(seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = load_breast_cancer(return_X_y=True)
    return X, y


def _moons(seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=600, noise=0.2, random_state=seed)
    return X, y


DATASETS = {"breast_cancer": _breast_cancer, "moons": _moons}


@dataclass
class DataSplits:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    n_features: int
    pca_explained_variance: float  # fraction of train variance kept by PCA


def load_splits(
    name: str,
    n_components: int,
    seed: int,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> DataSplits:
    """Stratified train/val/test split, then StandardScaler -> PCA -> MinMax[0, pi].

    The pipeline is fit on the training split only and *applied* to val/test,
    so no statistic from held-out data reaches the model. Val/test angles can
    fall slightly outside [0, pi]; that is expected and harmless for rotation
    encodings (they are periodic), and clipping would hide distribution shift.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    X, y = DATASETS[name](seed)
    n_components = min(n_components, X.shape[1])

    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=rel_val, stratify=y_trval, random_state=seed
    )

    pipe = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
            ("to_angles", MinMaxScaler(feature_range=(0.0, np.pi))),
        ]
    )
    pipe.fit(X_train)
    explained = float(pipe.named_steps["pca"].explained_variance_ratio_.sum())

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float64)

    return DataSplits(
        X_train=t(pipe.transform(X_train)),
        y_train=t(y_train),
        X_val=t(pipe.transform(X_val)),
        y_val=t(y_val),
        X_test=t(pipe.transform(X_test)),
        y_test=t(y_test),
        n_features=n_components,
        pca_explained_variance=explained,
    )
