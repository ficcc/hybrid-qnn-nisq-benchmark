"""Global determinism and experiment configuration."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

DEFAULT_SEED = 42


def set_global_determinism(seed: int = DEFAULT_SEED) -> None:
    """Make every run reproducible and force float64 everywhere.

    PennyLane's analytic simulators (shots=None) are deterministic, so seeding
    NumPy / PyTorch / Python covers weight init, data splits and batch order.
    float64 avoids Float32/Float64 mismatches between torch layers and QNode
    outputs (PennyLane returns float64 expectation values).
    """
    torch.set_default_dtype(torch.float64)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class ExperimentConfig:
    dataset: str = "breast_cancer"
    n_qubits: int = 4  # = number of PCA components fed to the circuit
    n_layers: int = 3  # data re-uploading layers
    ansatz: str = "reupload_hea"  # "reupload_hea" | "sel"
    init: str = "uniform"  # "uniform" in [0, 2pi) | "small" N(0, 0.1^2)
    val_size: float = 0.2
    test_size: float = 0.2
    epochs: int = 60
    batch_size: int = 32
    # Separate learning rates: PQC rotation angles live on a periodic scale of
    # ~[0, 2pi] and tolerate larger steps than MLP weights. A per-model LR grid
    # selected on the validation split is the stricter alternative.
    lr_quantum: float = 0.05
    lr_classical: float = 0.01
    patience: int = 15  # early stopping on validation loss
    mlp_activation: str = "tanh"

    def to_dict(self) -> dict:
        return asdict(self)
