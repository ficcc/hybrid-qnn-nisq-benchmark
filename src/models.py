"""Hybrid QNN and parameter-matched classical MLP."""

from __future__ import annotations

import math

import torch
from torch import nn

from .circuits import make_qnode, weight_shape
from .noise import NoiseModel


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class HybridQNN(nn.Module):
    """x -> PQC -> [<Z_0>, ..., <Z_{n-1}>] -> Linear(n, 1) -> logit.

    The quantum layer is the feature extractor; a tiny linear head maps the
    n bounded expectation values in [-1, 1] to an unbounded logit for
    BCEWithLogitsLoss. Its n+1 parameters are counted toward the total so the
    classical baseline comparison stays honest.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        ansatz: str = "reupload_hea",
        init: str = "uniform",
        noise: NoiseModel | None = None,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.ansatz = ansatz
        shape = weight_shape(ansatz, n_layers, n_qubits)
        if init == "uniform":
            w = torch.rand(shape) * 2 * math.pi
        elif init == "small":
            w = torch.randn(shape) * 0.1  # near-identity init, a barren-plateau mitigation
        else:
            raise ValueError(f"init must be 'uniform' or 'small', got {init!r}")
        self.q_weights = nn.Parameter(w)
        self.head = nn.Linear(n_qubits, 1)
        self.set_noise(noise)

    def set_noise(
        self, noise: NoiseModel | None, shots: int | None = None, seed: int | None = None
    ) -> None:
        """Rebuild the QNode on a new simulator; trained weights are untouched."""
        self.noise, self.shots = noise, shots
        self._qnode = make_qnode(self.n_qubits, self.ansatz, noise, shots=shots, seed=seed)

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._qnode(x, self.q_weights)
        z = torch.stack(list(out), dim=-1)  # (batch, n_qubits)
        return z.to(torch.get_default_dtype())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.quantum_features(x)).squeeze(-1)


class ClassicalMLP(nn.Module):
    """in -> hidden -> act -> 1 logit. Params: in*h + h + h + 1 = h(in + 2) + 1."""

    def __init__(self, in_features: int, hidden: int, activation: str = "tanh") -> None:
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        self.net = nn.Sequential(nn.Linear(in_features, hidden), act(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def matched_mlp_hidden(target_params: int, in_features: int) -> int:
    """Hidden width whose one-hidden-layer MLP parameter count is closest to target
    (ties -> the smaller network, which favours the baseline less)."""
    best_h, best_gap = 1, float("inf")
    for h in range(1, 4096):
        gap = abs(h * (in_features + 2) + 1 - target_params)
        if gap < best_gap:
            best_h, best_gap = h, gap
    return best_h
