"""Barren-plateau diagnostics: gradient variance over random initialisations.

Setting. For a cost C(theta) = <psi(x, theta)| O |psi(x, theta)>, a barren
plateau means Var_theta[dC/dtheta_k] shrinks exponentially in the number of
qubits n. A gradient that small cannot be estimated with polynomially many
shots, so training stalls no matter which optimizer is used.

Linear-algebra intuition. Deep random circuits produce approximately
Haar-random states. A random unit vector in C^(2^n) has tiny overlap with any
fixed structure, so <O> concentrates around its mean with variance ~ 2^-n.

What we measure. For each (n, L, init) we draw B independent pairs
(theta, x), with theta from the init distribution and x ~ U[0, pi]^n, and
compute exact backprop gradients of two costs:
  local : O = Z_0               (one-qubit observable, the regime our classifier uses)
  global: O = Z_0 Z_1 ... Z_{n-1}  (n-qubit observable)
Cerezo et al. (Nat. Commun. 2021) prove that global costs plateau even at
shallow depth, while local costs stay trainable when L = O(log n).

Efficiency. All B circuits are simulated in a single broadcast pass. Since
they share no parameters, d(sum_b C_b)/d(theta_b) = dC_b/d(theta_b), so a
single backward call yields every per-sample gradient. A test checks this
identity against one-at-a-time evaluation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pennylane as qml
import torch

from .circuits import apply_ansatz

COSTS = ("local", "global")
INITS = ("uniform", "small")


def make_cost_qnode(n_qubits: int, entangler: str = "ring"):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode(inputs, weights):
        apply_ansatz(inputs, weights, "reupload_hea", n_qubits, entangler=entangler)
        global_obs = qml.prod(*[qml.PauliZ(w) for w in range(n_qubits)])
        return qml.expval(qml.PauliZ(0)), qml.expval(global_obs)

    return qnode


def sample_weights(n_samples: int, n_layers: int, n_qubits: int, init: str, gen: torch.Generator) -> torch.Tensor:
    shape = (n_samples, n_layers, n_qubits, 2)
    if init == "uniform":
        return torch.rand(shape, generator=gen) * 2 * math.pi
    if init == "small":
        return torch.randn(shape, generator=gen) * 0.1
    raise ValueError(f"init must be one of {INITS}, got {init!r}")


@dataclass
class GradStats:
    entangler: str
    n_qubits: int
    n_layers: int
    init: str
    cost: str
    n_samples: int
    var_mean: float  # per-parameter variance, averaged over all parameters
    var_first: float  # variance of dC/d theta[layer 0, qubit 0, RZ]
    var_last: float  # variance of dC/d theta[last layer, qubit 0, RY]
    cost_var: float  # variance of C itself (concentration of the cost)

    def to_dict(self) -> dict:
        return asdict(self)


def per_sample_gradients(
    n_qubits: int, weights: torch.Tensor, inputs: torch.Tensor, entangler: str = "ring"
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Returns ({cost: grads of shape (B, L, n, 2)}, {cost: values of shape (B,)})."""
    qnode = make_cost_qnode(n_qubits, entangler)
    w = weights.detach().clone().requires_grad_(True)
    local, glob = qnode(inputs, w)
    g_local = torch.autograd.grad(local.sum(), w, retain_graph=True)[0]
    g_global = torch.autograd.grad(glob.sum(), w)[0]
    return {"local": g_local, "global": g_global}, {"local": local.detach(), "global": glob.detach()}


def gradient_variance(
    n_qubits: int, n_layers: int, init: str, n_samples: int, seed: int, entangler: str = "ring"
) -> list[GradStats]:
    gen = torch.Generator().manual_seed(seed)
    weights = sample_weights(n_samples, n_layers, n_qubits, init, gen)
    inputs = torch.rand((n_samples, n_qubits), generator=gen) * math.pi
    grads, values = per_sample_gradients(n_qubits, weights, inputs, entangler)

    out = []
    for cost in COSTS:
        g = grads[cost]
        per_param_var = g.reshape(n_samples, -1).var(dim=0, unbiased=True)
        out.append(
            GradStats(
                entangler=entangler,
                n_qubits=n_qubits,
                n_layers=n_layers,
                init=init,
                cost=cost,
                n_samples=n_samples,
                var_mean=per_param_var.mean().item(),
                var_first=g[:, 0, 0, 0].var(unbiased=True).item(),
                var_last=g[:, -1, 0, 1].var(unbiased=True).item(),
                cost_var=values[cost].var(unbiased=True).item(),
            )
        )
    return out


def fit_exponential_decay(ns: np.ndarray, variances: np.ndarray) -> dict[str, float]:
    """Least-squares fit of log Var = a - n log(b). Returns the per-qubit decay
    factor b (Var shrinks b-fold per added qubit) and R^2 of the log-linear fit.
    For reference, Haar-random states give b = 2."""
    ns, logv = np.asarray(ns, float), np.log(np.asarray(variances, float))
    slope, intercept = np.polyfit(ns, logv, 1)
    pred = intercept + slope * ns
    ss_res, ss_tot = np.sum((logv - pred) ** 2), np.sum((logv - logv.mean()) ** 2)
    return {"decay_factor": float(np.exp(-slope)), "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0}
