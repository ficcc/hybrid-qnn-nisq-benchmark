"""Publication-style figures (300 DPI PNG) for assets/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pennylane as qml  # noqa: E402
import torch  # noqa: E402

from .models import HybridQNN  # noqa: E402
from .train import TrainResult  # noqa: E402

DPI = 300
COLORS = {"Hybrid QNN": "#2E5EAA", "MLP (param-matched)": "#D1495B"}


def _pad(curves: list[list[float]]) -> np.ndarray:
    """Early stopping gives ragged curves; pad with NaN so nanmean/nanstd only
    average over seeds that were still training at that epoch."""
    longest = max(len(c) for c in curves)
    arr = np.full((len(curves), longest), np.nan)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = c
    return arr


def plot_learning_curves(histories: dict[str, list[TrainResult]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for name, runs in histories.items():
        color = COLORS.get(name)
        for ax, attr, title in [
            (axes[0], "train_loss", "Training loss (BCE)"),
            (axes[1], "val_loss", "Validation loss (BCE)"),
        ]:
            arr = _pad([getattr(r, attr) for r in runs])
            mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, label=f"{name} (n={len(runs)})", color=color, lw=1.8)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, lw=0)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
    axes[0].set_ylabel("Loss (mean ± std over seeds)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_noise_degradation(
    df,
    reference: dict[str, dict[str, float]],
    path: Path,
    reference_label: str = "MLP (noiseless, param-matched)",
) -> None:
    """df: one row per (seed, channel, p, shots) with metric columns
    `accuracy`, `roc_auc`, `recal_accuracy`, `recal_roc_auc`.
    reference: {metric: {"mean", "std"}} drawn as a horizontal band."""
    channels = list(dict.fromkeys(df["channel"]))
    metrics = [("accuracy", "Test accuracy"), ("roc_auc", "Test ROC-AUC")]
    fig, axes = plt.subplots(len(metrics), len(channels), figsize=(5.2 * len(channels), 7), sharex=True, sharey="row", squeeze=False)
    shot_settings = sorted(df["shots"].fillna(0).unique())

    for col, channel in enumerate(channels):
        sub = df[df["channel"] == channel]
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            ref = reference[metric]
            ax.axhspan(ref["mean"] - ref["std"], ref["mean"] + ref["std"], color="0.6", alpha=0.25, lw=0)
            ax.axhline(ref["mean"], color="0.4", lw=1, ls=":", label=reference_label)
            for prefix, color, name in [("", "#2E5EAA", "QNN, trained head"), ("recal_", "#E07A1F", "QNN, recalibrated head")]:
                for shots in shot_settings:
                    s = sub[sub["shots"].fillna(0) == shots]
                    g = s.groupby("p")[prefix + metric]
                    mean, std = g.mean(), g.std(ddof=1).fillna(0)
                    ls = "-" if shots == 0 else "--"
                    shot_lbl = "exact" if shots == 0 else f"{int(shots)} shots"
                    ax.plot(mean.index, mean.values, ls=ls, marker="o", ms=3, color=color, lw=1.6, label=f"{name} ({shot_lbl})")
                    ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.12, lw=0)
            ax.set_xscale("symlog", linthresh=1e-3)
            ax.grid(alpha=0.3, which="both")
            if row == 0:
                ax.set_title(channel.replace("_", " ").capitalize() + " noise")
            if row == len(metrics) - 1:
                ax.set_xlabel("Error probability p per qubit per layer")
            if col == 0:
                ax.set_ylabel(f"{ylabel} (mean ± std)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_gradient_variance(df, path: Path, operating_point: tuple[int, int] = (4, 3)) -> None:
    """Three panels from the barren-plateau grid (uniform init unless stated):
    (a) Var vs n at the classifier's depth, (b) Var vs L at the largest n,
    (c) Var vs n at the largest depth, uniform vs. small init (local cost)."""
    n_op, L_op = operating_point
    L_max, n_max = df["n_layers"].max(), df["n_qubits"].max()
    styles = {
        ("ring", "local"): dict(color="#2E5EAA", ls="-", marker="o"),
        ("ring", "global"): dict(color="#2E5EAA", ls="--", marker="s"),
        ("brick", "local"): dict(color="#E07A1F", ls="-", marker="o"),
        ("brick", "global"): dict(color="#E07A1F", ls="--", marker="s"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    uni = df[df["init"] == "uniform"]

    def series(ax, data, x, key, label, **kw):
        d = data.sort_values(x)
        ax.plot(d[x], d["var_mean"], label=label, ms=4, lw=1.6, **kw)

    ax = axes[0]
    sub = uni[uni["n_layers"] == L_op]
    for (ent, cost), st in styles.items():
        series(ax, sub[(sub["entangler"] == ent) & (sub["cost"] == cost)], "n_qubits", None, f"{ent}, {cost} cost", **st)
    ns = np.array(sorted(sub["n_qubits"].unique()))
    anchor = sub[(sub["n_qubits"] == ns[0])]["var_mean"].max()
    ax.plot(ns, anchor * 2.0 ** -(ns - ns[0]), color="0.5", ls=":", lw=1.2, label=r"$\propto 2^{-n}$ (Haar)")
    op = sub[(sub["entangler"] == "ring") & (sub["cost"] == "local") & (sub["n_qubits"] == n_op)]
    if len(op):
        ax.scatter([n_op], op["var_mean"], s=160, facecolors="none", edgecolors="k", lw=1.5, zorder=5,
                   label="trained classifier")
    ax.set_title(f"(a) Width scaling at L = {L_op}")
    ax.set_xlabel("Qubits n")

    ax = axes[1]
    sub = uni[uni["n_qubits"] == n_max]
    for (ent, cost), st in styles.items():
        series(ax, sub[(sub["entangler"] == ent) & (sub["cost"] == cost)], "n_layers", None, f"{ent}, {cost}", **st)
    ax.set_title(f"(b) Depth scaling at n = {n_max}")
    ax.set_xlabel("Layers L")

    ax = axes[2]
    sub = df[(df["n_layers"] == L_max) & (df["cost"] == "local")]
    init_style = {"uniform": dict(ls="-", marker="o"), "small": dict(ls="-.", marker="^")}
    for ent in ["ring", "brick"]:
        for init, st in init_style.items():
            d = sub[(sub["entangler"] == ent) & (sub["init"] == init)]
            if len(d):
                series(ax, d, "n_qubits", None, f"{ent}, {init} init",
                       color=styles[(ent, "local")]["color"], **st)
    ax.set_title(f"(c) Initialisation at L = {L_max} (local cost)")
    ax.set_xlabel("Qubits n")

    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(r"Var$[\partial C / \partial \theta]$ (mean over parameters)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_circuit(model: HybridQNN, x_example: torch.Tensor, path: Path) -> None:
    """Draw the circuit with templates expanded into native gates."""
    with torch.no_grad():
        fig, _ = qml.draw_mpl(model._qnode, decimals=None, style="pennylane", level="device")(
            x_example, model.q_weights
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
