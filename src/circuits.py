"""Parameterised quantum circuits (ansätze) and QNode construction.

Two ansätze, both on n qubits with L layers:

reupload_hea (default) - data re-uploading + hardware-efficient layer
    per layer:  RY(x_i) on each qubit            (data encoding, re-uploaded)
                RZ(t) RY(t) on each qubit         (2n trainable angles)
                CNOT ring                         (n two-qubit gates, nearest-neighbour)
    params: 2nL.  depth/layer: 3 single-qubit + n CNOTs.
    Why: re-uploading the input L times makes the model a Fourier series in x
    with frequencies up to L (Schuld, Sweke & Meyer 2021); a single encoding
    caps it at frequency 1 regardless of depth. Nearest-neighbour CNOTs map to
    linear/ring hardware without SWAP overhead.

sel - single encoding + qml.StronglyEntanglingLayers
    per layer: Rot(a,b,c) on each qubit (3n params) + CNOTs with range r_l.
    params: 3nL. More expressive per layer but 1.5x the parameters, long-range
    CNOTs (SWAPs on real hardware), and only frequency-1 data dependence.

Readout: <Z_i> on every qubit (local observables). Local costs keep gradient
variance polynomial in n for shallow circuits, whereas global observables
(e.g. Z⊗Z⊗...⊗Z) induce barren plateaus even at shallow depth (Cerezo et al. 2021).
"""

from __future__ import annotations

import pennylane as qml
import torch

from .noise import NoiseModel, apply_layer_noise

ANSATZE = ("reupload_hea", "sel")


def weight_shape(ansatz: str, n_layers: int, n_qubits: int) -> tuple[int, ...]:
    if ansatz == "reupload_hea":
        return (n_layers, n_qubits, 2)
    if ansatz == "sel":
        return (n_layers, n_qubits, 3)
    raise ValueError(f"Unknown ansatz {ansatz!r}; choose from {ANSATZE}")


def _cnot_ring(wires: list[int]) -> None:
    n = len(wires)
    if n < 2:
        return
    pairs = [(wires[i], wires[i + 1]) for i in range(n - 1)]
    if n > 2:  # close the ring (for n == 2 the "ring" is a single CNOT)
        pairs.append((wires[-1], wires[0]))
    for a, b in pairs:
        qml.CNOT(wires=[a, b])


def _cnot_brick(wires: list[int]) -> None:
    """Two sub-layers of nearest-neighbour CNOTs on disjoint pairs:
    (0,1),(2,3),... then (1,2),(3,4),... Circuit depth 2 per layer, and each
    layer widens the backward light cone of a single-qubit observable by
    at most 2 qubits on either side."""
    n = len(wires)
    for start in (0, 1):
        for i in range(start, n - 1, 2):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


ENTANGLERS = {"ring": _cnot_ring, "brick": _cnot_brick}


def apply_ansatz(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    ansatz: str,
    n_qubits: int,
    noise: NoiseModel | None = None,
    entangler: str = "ring",
) -> None:
    """Apply encoding + trainable layers (no measurement).

    Broadcasting: `inputs` may be (B, n_features) and, for reupload_hea,
    `weights` may carry a leading batch axis (B, L, n, 2) as well. PennyLane
    then simulates B independent circuits in one pass, which the
    barren-plateau experiment uses to evaluate many random initialisations
    at once.

    entangler (reupload_hea only): "ring" is a sequential staircase of CNOTs
    (0,1),(1,2),...,(n-1,0). Because the gates act one after another, a
    single layer already has causal depth n, so <Z_0> can depend on every
    qubit after one layer. "brick" uses parallel even/odd pairs and keeps
    the light cone local for small L.
    """
    entangle = ENTANGLERS[entangler]
    wires = list(range(n_qubits))
    n_layers = weights.shape[-3]

    if ansatz == "reupload_hea":
        for layer in range(n_layers):
            qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
            for i, w in enumerate(wires):
                qml.RZ(weights[..., layer, i, 0], wires=w)
                qml.RY(weights[..., layer, i, 1], wires=w)
            entangle(wires)
            apply_layer_noise(noise, wires)
    elif ansatz == "sel":
        if weights.ndim != 3:
            raise ValueError("ansatz 'sel' does not support batched weights")
        qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
        apply_layer_noise(noise, wires)
        for layer in range(n_layers):
            qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=wires)
            apply_layer_noise(noise, wires)
    else:
        raise ValueError(f"Unknown ansatz {ansatz!r}; choose from {ANSATZE}")


def circuit(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    ansatz: str,
    n_qubits: int,
    noise: NoiseModel | None = None,
):
    """Quantum function for the classifier. `inputs` has shape (batch, n_features)
    or (n_features,); PennyLane parameter broadcasting runs the whole batch in one
    simulation."""
    apply_ansatz(inputs, weights, ansatz, n_qubits, noise)
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]


def make_qnode(
    n_qubits: int,
    ansatz: str,
    noise: NoiseModel | None = None,
    shots: int | None = None,
    seed: int | None = None,
):
    """Statevector simulator when noiseless (2^n amplitudes), density-matrix
    simulator when noisy (2^n x 2^n).

    shots=None returns exact expectation values (differentiated by backprop).
    shots=N estimates each <Z_i> from N measurement samples, as on hardware:
    the estimate has standard error sqrt((1 - <Z>^2) / N). Gradients then
    use the parameter-shift rule, since sampling is not backprop-differentiable.
    """
    noisy = noise is not None and noise.is_noisy
    device_name = "default.mixed" if noisy else "default.qubit"
    dev = qml.device(device_name, wires=n_qubits, seed=seed)
    diff_method = "backprop" if shots is None else "parameter-shift"

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def qnode(inputs, weights):
        return circuit(inputs, weights, ansatz=ansatz, n_qubits=n_qubits, noise=noise)

    if shots is not None:
        qnode = qml.set_shots(qnode, shots=shots)
    return qnode
