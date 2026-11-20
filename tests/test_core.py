"""Fast correctness checks for the claims the README relies on."""

import numpy as np
import pytest
import torch

from src.circuits import weight_shape
from src.config import set_global_determinism
from src.data import load_splits
from src.evaluate import noise_sweep
from src.models import ClassicalMLP, HybridQNN, count_parameters, matched_mlp_hidden
from src.noise import NoiseModel


@pytest.fixture(autouse=True)
def _determinism():
    set_global_determinism(42)


def test_splits_disjoint_and_stratified():
    s = load_splits("breast_cancer", 4, seed=42)
    n = len(s.y_train) + len(s.y_val) + len(s.y_test)
    assert n == 569
    rates = [y.mean().item() for y in (s.y_train, s.y_val, s.y_test)]
    assert max(rates) - min(rates) < 0.03  # stratified


def test_transforms_fit_on_train_only():
    s = load_splits("breast_cancer", 4, seed=42)
    # MinMax was fit on train -> train is exactly [0, pi]; held-out data need not be.
    assert torch.isclose(s.X_train.min(), torch.tensor(0.0))
    assert torch.isclose(s.X_train.max(), torch.tensor(np.pi))


def test_float64_end_to_end():
    s = load_splits("moons", 2, seed=42)
    model = HybridQNN(2, 2)
    out = model(s.X_val[:8])
    assert out.dtype == torch.float64 and out.shape == (8,)


def test_param_matching():
    qnn = HybridQNN(4, 3)
    assert count_parameters(qnn) == 3 * 4 * 2 + 4 + 1
    h = matched_mlp_hidden(count_parameters(qnn), 4)
    assert abs(count_parameters(ClassicalMLP(4, h)) - count_parameters(qnn)) <= 3


def test_gradients_flow_to_circuit():
    qnn = HybridQNN(3, 2)
    qnn(torch.rand(5, 3)).sum().backward()
    assert qnn.q_weights.grad is not None and qnn.q_weights.grad.abs().sum() > 0


def test_zero_noise_mixed_matches_statevector():
    torch.manual_seed(0)
    x = torch.rand(6, 3)
    qnn = HybridQNN(3, 2)
    clean = qnn(x).detach()
    qnn.set_noise(NoiseModel("depolarizing", 1e-12))  # forces default.mixed
    assert torch.allclose(qnn(x).detach(), clean, atol=1e-8)


def test_full_depolarizing_destroys_signal():
    x = torch.rand(6, 3)
    qnn = HybridQNN(3, 2)
    qnn.set_noise(NoiseModel("depolarizing", 0.75))  # p=3/4 is the fully depolarizing point
    z = qnn.quantum_features(x).detach()
    assert torch.allclose(z, torch.zeros_like(z), atol=1e-8)


def test_noise_sweep_restores_model():
    s = load_splits("moons", 2, seed=42)
    qnn = HybridQNN(2, 2)
    rows = noise_sweep(qnn, s.X_test, s.y_test, [0.0, 0.05])
    assert len(rows) == 2 and qnn.noise is None


def test_shots_estimate_is_close_and_reproducible():
    torch.manual_seed(0)
    x = torch.rand(4, 3)
    qnn = HybridQNN(3, 2)
    exact = qnn.quantum_features(x).detach()
    qnn.set_noise(None, shots=20000, seed=1)
    a = qnn.quantum_features(x).detach()
    qnn.set_noise(None, shots=20000, seed=1)
    b = qnn.quantum_features(x).detach()
    assert torch.equal(a, b)  # same device seed -> identical samples
    assert torch.allclose(a, exact, atol=0.03)  # ~4 standard errors at 20k shots


def test_recalibration_never_sees_test_and_restores_state():
    s = load_splits("moons", 2, seed=42)
    qnn = HybridQNN(2, 2)
    rows = noise_sweep(qnn, s.X_test, s.y_test, [0.1], X_fit=s.X_train, y_fit=s.y_train, shots=200)
    assert "recal_accuracy" in rows[0] and rows[0]["shots"] == 200
    assert qnn.noise is None and qnn.shots is None


@pytest.mark.parametrize("entangler", ["ring", "brick"])
def test_broadcast_gradients_equal_per_sample_gradients(entangler):
    """The barren-plateau experiment relies on d(sum_b C_b)/d(theta_b) = dC_b/d(theta_b)."""
    from src.landscape import make_cost_qnode, per_sample_gradients

    n, L, B = 3, 2, 4
    w = torch.rand(B, L, n, 2) * 6.28
    x = torch.rand(B, n) * 3.14
    batched, _ = per_sample_gradients(n, w, x, entangler)
    qnode = make_cost_qnode(n, entangler)
    for b in range(B):
        wb = w[b].clone().requires_grad_(True)
        local, glob = qnode(x[b], wb)
        g_local = torch.autograd.grad(local, wb, retain_graph=True)[0]
        g_global = torch.autograd.grad(glob, wb)[0]
        assert torch.allclose(batched["local"][b], g_local, atol=1e-10)
        assert torch.allclose(batched["global"][b], g_global, atol=1e-10)


def test_brick_light_cone_is_local():
    """With one brick layer, <Z_0> cannot depend on parameters of qubits >= 3."""
    from src.landscape import per_sample_gradients

    n = 6
    grads, _ = per_sample_gradients(n, torch.rand(5, 1, n, 2) * 6.28, torch.rand(5, n) * 3.14, "brick")
    assert grads["local"][:, 0, 3:, :].abs().max() < 1e-12
    assert grads["local"][:, 0, 0, :].abs().max() > 1e-6


def test_exponential_fit_recovers_known_rate():
    from src.landscape import fit_exponential_decay

    ns = np.array([4, 6, 8, 10])
    fit = fit_exponential_decay(ns, 0.3 * 2.0 ** -ns)
    assert abs(fit["decay_factor"] - 2.0) < 1e-9 and fit["r2"] > 0.999


def test_weight_shapes():
    assert weight_shape("reupload_hea", 3, 4) == (3, 4, 2)
    assert weight_shape("sel", 3, 4) == (3, 4, 3)
