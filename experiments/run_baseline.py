"""Baseline experiment: noiseless Hybrid QNN vs. parameter-matched MLP vs. logistic regression.

Run from the repository root:
    python -m experiments.run_baseline                 # 5 seeds, full run
    python -m experiments.run_baseline --quick         # 2 seeds x 5 epochs smoke test

Outputs:
    results/baseline_<dataset>_<ansatz>.json   per-seed + aggregated metrics, config, versions
    results/checkpoints/*.pt                   trained weights (reused by the noise-sweep experiment)
    assets/learning_curves.png, assets/circuit_architecture.png
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
import sklearn
import torch
from sklearn.linear_model import LogisticRegression

from src.config import DEFAULT_SEED, ExperimentConfig, set_global_determinism
from src.data import load_splits
from src.evaluate import aggregate, classification_metrics, evaluate
from src.models import ClassicalMLP, HybridQNN, count_parameters, matched_mlp_hidden
from src.plotting import plot_circuit, plot_learning_curves
from src.train import train_model

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    d = ExperimentConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=d.dataset, choices=["breast_cancer", "moons"])
    p.add_argument("--n-qubits", type=int, default=d.n_qubits)
    p.add_argument("--n-layers", type=int, default=d.n_layers)
    p.add_argument("--ansatz", default=d.ansatz, choices=["reupload_hea", "sel"])
    p.add_argument("--init", default=d.init, choices=["uniform", "small"])
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr-quantum", type=float, default=d.lr_quantum)
    p.add_argument("--lr-classical", type=float, default=d.lr_classical)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--quick", action="store_true", help="2 seeds x 5 epochs smoke test")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def fmt(agg: dict, key: str) -> str:
    return f"{agg[key]['mean']:.3f} ± {agg[key]['std']:.3f}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    args = parse_args()
    if args.quick:
        args.n_seeds, args.epochs = 2, 5
    cfg = ExperimentConfig(
        dataset=args.dataset,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        ansatz=args.ansatz,
        init=args.init,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_quantum=args.lr_quantum,
        lr_classical=args.lr_classical,
        patience=args.patience,
    )
    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    ckpt_dir = ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    per_seed = {"Hybrid QNN": [], "MLP (param-matched)": [], "Logistic regression": []}
    histories = {"Hybrid QNN": [], "MLP (param-matched)": []}
    param_counts: dict[str, int] = {}
    pca_var = []

    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        set_global_determinism(seed)
        splits = load_splits(cfg.dataset, cfg.n_qubits, seed, cfg.val_size, cfg.test_size)
        pca_var.append(splits.pca_explained_variance)
        train_kw = dict(epochs=cfg.epochs, batch_size=cfg.batch_size, patience=cfg.patience, seed=seed, verbose=args.verbose)

        # --- Hybrid QNN ---------------------------------------------------
        torch.manual_seed(seed)
        qnn = HybridQNN(splits.n_features, cfg.n_layers, cfg.ansatz, cfg.init)
        q_params = count_parameters(qnn)
        hist_q = train_model_logged("Hybrid QNN", qnn, splits, lr=cfg.lr_quantum, **train_kw)
        m_q = evaluate(qnn, splits.X_test, splits.y_test)
        per_seed["Hybrid QNN"].append({"seed": seed, **m_q, "best_epoch": hist_q.best_epoch, "train_time_s": hist_q.train_time_s})
        histories["Hybrid QNN"].append(hist_q)
        torch.save(qnn.state_dict(), ckpt_dir / f"qnn_{cfg.dataset}_{cfg.ansatz}_L{cfg.n_layers}_seed{seed}.pt")

        # --- Parameter-matched MLP ----------------------------------------
        torch.manual_seed(seed)
        hidden = matched_mlp_hidden(q_params, splits.n_features)
        mlp = ClassicalMLP(splits.n_features, hidden, cfg.mlp_activation)
        hist_c = train_model_logged("MLP (param-matched)", mlp, splits, lr=cfg.lr_classical, **train_kw)
        m_c = evaluate(mlp, splits.X_test, splits.y_test)
        per_seed["MLP (param-matched)"].append({"seed": seed, **m_c, "best_epoch": hist_c.best_epoch, "train_time_s": hist_c.train_time_s})
        histories["MLP (param-matched)"].append(hist_c)
        torch.save(mlp.state_dict(), ckpt_dir / f"mlp_{cfg.dataset}_h{hidden}_seed{seed}.pt")

        # --- Linear reference (n+1 params): is any nonlinearity needed? ----
        logreg = LogisticRegression(max_iter=1000, random_state=seed)
        logreg.fit(splits.X_train.numpy(), splits.y_train.numpy())
        m_l = classification_metrics(splits.y_test, logreg.predict_proba(splits.X_test.numpy())[:, 1])
        per_seed["Logistic regression"].append({"seed": seed, **m_l})

        param_counts = {
            "Hybrid QNN": q_params,
            "MLP (param-matched)": count_parameters(mlp),
            "Logistic regression": splits.n_features + 1,
        }
        print(f"  test acc  QNN {m_q['accuracy']:.3f} | MLP {m_c['accuracy']:.3f} | LogReg {m_l['accuracy']:.3f}")

    metric_keys = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    agg = {name: aggregate([{k: r[k] for k in metric_keys} for r in runs]) for name, runs in per_seed.items()}

    # --- Report -------------------------------------------------------------
    print(f"\nTest metrics, mean ± std over {len(seeds)} seeds ({cfg.dataset}, {cfg.n_qubits} PCA comps, "
          f"{np.mean(pca_var):.1%} train variance kept)\n")
    print("| Model | Params | Accuracy | Balanced acc. | F1 | ROC-AUC |")
    print("|---|---:|---|---|---|---|")
    for name in per_seed:
        a = agg[name]
        print(f"| {name} | {param_counts[name]} | {fmt(a, 'accuracy')} | {fmt(a, 'balanced_accuracy')} | "
              f"{fmt(a, 'f1')} | {fmt(a, 'roc_auc')} |")

    out = {
        "config": cfg.to_dict(),
        "seeds": seeds,
        "param_counts": param_counts,
        "pca_explained_variance_mean": float(np.mean(pca_var)),
        "aggregate": agg,
        "per_seed": per_seed,
        "histories": {n: [vars(h) for h in hs] for n, hs in histories.items()},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pennylane": qml.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }
    suffix = "_quick" if args.quick else ""
    out_path = ROOT / "results" / f"baseline_{cfg.dataset}_{cfg.ansatz}{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path.relative_to(ROOT)}")

    if not args.quick:
        plot_learning_curves(histories, ROOT / "assets" / "learning_curves.png")
        plot_circuit(qnn, splits.X_test[0], ROOT / "assets" / "circuit_architecture.png")
        print("Saved assets/learning_curves.png, assets/circuit_architecture.png")


def train_model_logged(name, model, splits, **kw):
    res = train_model(model, splits, **kw)
    print(f"  {name:<22} params={count_parameters(model):>3}  best_epoch={res.best_epoch:>3}  "
          f"val_loss={min(res.val_loss):.4f}  time={res.train_time_s:.1f}s")
    return res


if __name__ == "__main__":
    main()
