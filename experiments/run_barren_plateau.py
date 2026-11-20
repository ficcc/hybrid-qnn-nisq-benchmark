"""Barren-plateau study: gradient variance vs. qubits, depth, cost locality,
entangler topology and initialisation for the re-uploading HEA ansatz.

Run from the repository root:
    python -m experiments.run_barren_plateau            # full grid (~5 min on CPU)
    python -m experiments.run_barren_plateau --quick    # small grid smoke test

Outputs:
    results/barren_plateau.csv / .json
    assets/gradient_variance.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_SEED, set_global_determinism
from src.landscape import INITS, fit_exponential_decay, gradient_variance
from src.plotting import plot_gradient_variance

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qubits", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4, 8, 12])
    p.add_argument("--entanglers", nargs="+", default=["ring", "brick"], choices=["ring", "brick"])
    p.add_argument("--inits", nargs="+", default=list(INITS), choices=INITS)
    p.add_argument("--samples", type=int, default=300, help="random (theta, x) draws per configuration")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--operating-layers", type=int, default=3, help="depth of the trained classifier (panel a)")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.quick:
        args.qubits, args.layers, args.samples = [2, 4, 6, 8], [1, 3], 100
    set_global_determinism(args.seed)

    rows, t0 = [], time.perf_counter()
    for ent in args.entanglers:
        for init in args.inits:
            for n in args.qubits:
                for L in args.layers:
                    # Same seed per (n, L) across settings: paired comparisons share the data draws.
                    rows += [s.to_dict() for s in gradient_variance(n, L, init, args.samples, args.seed, ent)]
            print(f"{ent:>5} / {init:<7} done ({time.perf_counter() - t0:.0f}s)")
    df = pd.DataFrame(rows)

    # --- Exponential decay fits: Var ∝ b^-n over n >= 4 --------------------
    fits = []
    for (ent, init, cost, L), g in df[df["n_qubits"] >= 4].groupby(["entangler", "init", "cost", "n_layers"]):
        if g["n_qubits"].nunique() >= 3:
            fits.append({"entangler": ent, "init": init, "cost": cost, "n_layers": L,
                         **fit_exponential_decay(g["n_qubits"].values, g["var_mean"].values)})
    fits = pd.DataFrame(fits, columns=["entangler", "init", "cost", "n_layers", "decay_factor", "r2"])

    show_L = sorted({args.operating_layers, max(args.layers)} & set(args.layers))
    print("\nPer-qubit decay factor b of Var[dC/dθ] ∝ b^-n (n ≥ 4; b = 1 no decay, b ≈ 2 Haar-random)\n")
    print("| Entangler | Init | Cost | " + " | ".join(f"L={L}" for L in show_L) + " |")
    print("|---|---|---|" + "---|" * len(show_L))
    for (ent, init, cost), g in fits.groupby(["entangler", "init", "cost"]):
        cells = []
        for L in show_L:
            r = g[g["n_layers"] == L]
            cells.append(f"{r['decay_factor'].iloc[0]:.2f} (R²={r['r2'].iloc[0]:.2f})" if len(r) else "—")
        print(f"| {ent} | {init} | {cost} | " + " | ".join(cells) + " |")

    suffix = "_quick" if args.quick else ""
    stem = ROOT / "results" / f"barren_plateau{suffix}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".json").write_text(json.dumps(
        {"args": {k: v for k, v in vars(args).items()}, "fits": fits.to_dict("records"), "rows": rows}, indent=2
    ))
    print(f"\nSaved {stem.relative_to(ROOT)}.csv/.json")

    if not args.quick:
        out = ROOT / "assets" / "gradient_variance.png"
        plot_gradient_variance(df, out, operating_point=(4, args.operating_layers))
        print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
