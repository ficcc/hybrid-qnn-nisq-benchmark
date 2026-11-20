"""NISQ noise models applied inside the circuit on the `default.mixed` simulator.

Linear-algebra view: a noisy circuit acts on a density matrix rho (2^n x 2^n)
instead of a statevector. A noise channel is a set of Kraus matrices {K_i} with
sum K_i^dagger K_i = I, and maps rho -> sum_i K_i rho K_i^dagger.

* Depolarizing(p) on one qubit: with probability p the qubit's state is
  replaced by noise (Kraus ops sqrt(1-p) I, sqrt(p/3) X, Y, Z). Models
  generic gate error.
* AmplitudeDamping(gamma): |1> decays to |0> with probability gamma.
  Models T1 energy relaxation.

We use a layer-wise model: after every circuit layer, each qubit receives one
channel with strength p. This is an *effective* per-layer error rate rather
than a calibrated per-gate device model, so p should be read as "error per
layer per qubit".
"""

from __future__ import annotations

from dataclasses import dataclass

import pennylane as qml

CHANNELS = ("depolarizing", "amplitude_damping")


@dataclass(frozen=True)
class NoiseModel:
    channel: str = "depolarizing"
    p: float = 0.0

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}, got {self.channel!r}")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")

    @property
    def is_noisy(self) -> bool:
        return self.p > 0.0


def apply_layer_noise(noise: NoiseModel | None, wires) -> None:
    """Insert one noise channel per wire. No-op for noiseless models."""
    if noise is None or not noise.is_noisy:
        return
    for w in wires:
        if noise.channel == "depolarizing":
            qml.DepolarizingChannel(noise.p, wires=w)
        else:
            qml.AmplitudeDamping(noise.p, wires=w)
