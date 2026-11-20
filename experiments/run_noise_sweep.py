"""NISQ robustness: evaluate the noiselessly trained QNN checkpoints under noise.

Requires a finished baseline run (reads its JSON for config, seeds and the MLP
reference). For every seed, channel and error rate p we report:
  * raw          - circuit and trained head unchanged (naive deployment)
  * recalibrated - circuit unchanged, linear head refit on noisy *training*
                   features (n+1 classical params; test data never used to fit)
each with exact expectation values and with finite measurement shots.

Run from the repository root:
    python -m experiments.run_noise_sweep
    python -m experiments.run_noise_sweep --quick     # 2 seeds, 4 noise levels

Outputs:
    results/noise_sweep_<dataset>_<ansatz>.json / .csv
    assets/noise_degradation.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from src.config import set_global_determinism
from src.data import load_splits
from src.evaluate import noise_sweep
from src.models import HybridQNN
from src.noise import CHANNELS
from src.plotting import plot_noise_degradation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, default=ROOT / "results" / "baseline_breast_cancer_reupload_hea.json")
    p.add_argument("--ps", type=float, nargs="+", default=DEFAULT_PS)
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=CHANNELS)
    p.add_argument("--shots", type=int, default=1000, help="finite-shot setting (exact is always included)")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    base = json.loads(args.baseline.read_text())
    cfg, seeds = base["config"], base["seeds"]
    ps = args.ps
    if args.quick:
        seeds, ps = seeds[:2], [0.0, 0.01, 0.1, 0.3]

    rows = []
    t0 = time.perf_counter()
    for seed in seeds:
        set_global_determinism(seed)
        splits = load_splits(cfg["dataset"], cfg["n_qubits"], seed, cfg["val_size"], cfg["test_size"])
        qnn = HybridQNN(splits.n_features, cfg["n_layers"], cfg["ansatz"], cfg["init"])
        ckpt = ROOT / "results" / "checkpoints" / f"qnn_{cfg['dataset']}_{cfg['ansatz']}_L{cfg['n_layers']}_seed{seed}.pt"
        qnn.load_state_dict(torch.load(ckpt, weights_only=True))

        for channel in args.channels:
            for shots in (None, args.shots):
                rows += [
                    {"seed": seed, **r}
                    for r in noise_sweep(
                        qnn, splits.X_test, splits.y_test, ps, channel,
                        shots=shots, seed=seed, X_fit=splits.X_train, y_fit=splits.y_train,
                    )
                ]
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f}s elapsed)")

    df = pd.DataFrame(rows)
    df["shots"] = df["shots"].astype("Int64")

    # --- Report: accuracy, mean ± std over seeds ------------------------------
    def cell(g: pd.Series) -> str:
        return f"{g.mean():.3f} ± {g.std(ddof=1):.3f}"

    ref = base["aggregate"]["MLP (param-matched)"]
    print(f"\nTest accuracy vs. noise, {len(seeds)} seeds. "
          f"Noiseless param-matched MLP: {ref['accuracy']['mean']:.3f} ± {ref['accuracy']['std']:.3f}\n")
    for channel in args.channels:
        print(f"### {channel}\n")
        print(f"| p | raw (exact) | recal. (exact) | raw ({args.shots} shots) | recal. ({args.shots} shots) |")
        print("|---:|---|---|---|---|")
        sub = df[df["channel"] == channel]
        for p in ps:
            ex = sub[(sub["p"] == p) & sub["shots"].isna()]
            sh = sub[(sub["p"] == p) & (sub["shots"] == args.shots)]
            print(f"| {p:g} | {cell(ex['accuracy'])} | {cell(ex['recal_accuracy'])} | "
                  f"{cell(sh['accuracy'])} | {cell(sh['recal_accuracy'])} |")
        print()

    suffix = "_quick" if args.quick else ""
    stem = ROOT / "results" / f"noise_sweep_{cfg['dataset']}_{cfg['ansatz']}{suffix}"
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".json").write_text(json.dumps(
        {"baseline": str(args.baseline.relative_to(ROOT)), "seeds": seeds, "ps": ps,
         "channels": args.channels, "shots": args.shots, "rows": df.astype(object).where(df.notna(), None).to_dict("records")},
        indent=2,
    ))
    print(f"Saved {stem.relative_to(ROOT)}.csv/.json")

    if not args.quick:
        out = ROOT / "assets" / "noise_degradation.png"
        plot_noise_degradation(df, ref, out)
        print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
