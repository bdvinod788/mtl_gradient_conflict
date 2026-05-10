"""
plot_results.py  (lives in pcgrad/)

Generates comparison plots for the CSCI 567 MTL project.
Supports one run (vanilla or PCGrad alone) or two runs side-by-side.

Usage:
    # PCGrad only
    python plot_results.py \\
        --history /path/to/pcgrad/training_history.json \\
        --output  /path/to/plots/pcgrad \\
        --title   "PCGrad MTL"

    # Comparison: vanilla vs PCGrad
    python plot_results.py \\
        --history  /path/to/vanilla/training_history.json \\
        --history2 /path/to/pcgrad/training_history.json \\
        --output   /path/to/plots/comparison \\
        --title    "Vanilla MTL" \\
        --title2   "PCGrad MTL"

Outputs (all saved to --output directory):
    accuracy_per_task.png     — per-task val accuracy across epochs
    task_accuracy_overlay.png — all 4 tasks on one chart
    avg_val_loss.png          — average val loss across epochs
    conflict_rate.png         — gradient conflict rate across epochs
    gradient_score.png        — combined gradient conflict score across epochs
    pcgrad_conflicts.png      — PCGrad conflicts per step (PCGrad run only)
    summary_grid.png          — 2x2 grid of the four key plots (for the report)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # no display needed — works on CARC
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Style ─────────────────────────────────────────────────────────────────────

VANILLA_COLOR = "#4C9BE8"   # blue
PCGRAD_COLOR  = "#E87C4C"   # orange
TASK_COLORS   = {
    "yelp": "#5C85D6",
    "qnli": "#56B87E",
    "qqp":  "#D6A23C",
    "mnli": "#C7567A",
}
TASK_MARKERS  = {"yelp": "o", "qnli": "s", "qqp": "^", "mnli": "D"}
TASK_LABELS   = {
    "yelp": "Yelp (Sentiment)",
    "qnli": "QNLI (NLI)",
    "qqp":  "QQP (Paraphrase)",
    "mnli": "MNLI (3-class NLI)",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
})


# ── JSON loading ──────────────────────────────────────────────────────────────

def load_history(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def _extract(history: List[Dict], key: str) -> List[float]:
    return [ep[key] for ep in history]


def _extract_task(history: List[Dict], task: str, metric: str) -> List[float]:
    return [ep["per_task_val"][task][metric] for ep in history]


def _frozen_at(history: List[Dict]) -> Dict[str, Optional[int]]:
    """Return {task: epoch_number} when each task was first frozen. None if never."""
    frozen_epoch: Dict[str, Optional[int]] = {task: None for task in TASK_COLORS}
    seen: set = set()
    for ep in history:
        for task in ep.get("frozen_tasks", []):
            if task not in seen:
                frozen_epoch[task] = ep["epoch"]
                seen.add(task)
    return frozen_epoch


# ── Individual plots ──────────────────────────────────────────────────────────

def plot_accuracy_per_task(
    history1: List[Dict],
    title1: str,
    output_dir: Path,
    history2: Optional[List[Dict]] = None,
    title2: Optional[str] = None,
) -> Path:
    """Per-task validation accuracy over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    fig.suptitle("Per-Task Validation Accuracy", fontsize=15, fontweight="bold", y=1.01)

    tasks   = list(TASK_COLORS.keys())
    frozen1 = _frozen_at(history1)
    frozen2 = _frozen_at(history2) if history2 else {}
    epochs1 = _extract(history1, "epoch")
    epochs2 = _extract(history2, "epoch") if history2 else []

    for ax, task in zip(axes.flatten(), tasks):
        acc1 = [v * 100 for v in _extract_task(history1, task, "acc")]
        ax.plot(epochs1, acc1, color=VANILLA_COLOR, linewidth=2,
                marker=TASK_MARKERS[task], markersize=5, label=title1)

        if history2:
            acc2 = [v * 100 for v in _extract_task(history2, task, "acc")]
            ax.plot(epochs2, acc2, color=PCGRAD_COLOR, linewidth=2,
                    marker=TASK_MARKERS[task], markersize=5,
                    linestyle="--", label=title2)

        if frozen1[task]:
            ax.axvline(frozen1[task], color=VANILLA_COLOR, alpha=0.5,
                       linestyle=":", linewidth=1.5, label=f"{title1} frozen")
        if history2 and frozen2.get(task):
            ax.axvline(frozen2[task], color=PCGRAD_COLOR, alpha=0.5,
                       linestyle=":", linewidth=1.5, label=f"{title2} frozen")

        ax.set_title(TASK_LABELS[task], fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out = output_dir / "accuracy_per_task.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_task_accuracy_overlay(
    history1, title1, output_dir,
    history2=None, title2=None,
) -> Path:
    """All 4 tasks on one plot, run-1 solid / run-2 dashed."""
    fig, ax = plt.subplots(figsize=(10, 6))
    e1 = _extract(history1, "epoch")

    for task in TASK_COLORS:
        acc1 = [v * 100 for v in _extract_task(history1, task, "acc")]
        ax.plot(e1, acc1, color=TASK_COLORS[task], linewidth=2,
                marker=TASK_MARKERS[task], markersize=5,
                label=f"{TASK_LABELS[task]} ({title1})")

        if history2:
            e2   = _extract(history2, "epoch")
            acc2 = [v * 100 for v in _extract_task(history2, task, "acc")]
            ax.plot(e2, acc2, color=TASK_COLORS[task], linewidth=2,
                    marker=TASK_MARKERS[task], markersize=5,
                    linestyle="--", alpha=0.75,
                    label=f"{TASK_LABELS[task]} ({title2})")

    ax.set_title("All Tasks — Validation Accuracy", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    out = output_dir / "task_accuracy_overlay.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_avg_val_loss(
    history1, title1, output_dir,
    history2=None, title2=None,
) -> Path:
    """Average validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs1 = _extract(history1, "epoch")
    loss1   = _extract(history1, "avg_val_loss")

    ax.plot(epochs1, loss1, color=VANILLA_COLOR, linewidth=2,
            marker="o", markersize=5, label=title1)

    if history2:
        epochs2 = _extract(history2, "epoch")
        loss2   = _extract(history2, "avg_val_loss")
        ax.plot(epochs2, loss2, color=PCGRAD_COLOR, linewidth=2,
                marker="o", markersize=5, linestyle="--", label=title2)

    ax.set_title("Average Validation Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Val Loss")
    ax.legend()
    plt.tight_layout()
    out = output_dir / "avg_val_loss.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_conflict_rate(
    history1, title1, output_dir,
    history2=None, title2=None,
) -> Path:
    """Gradient conflict rate over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs1 = _extract(history1, "epoch")
    rate1   = _extract(history1, "conflict_rate")

    ax.plot(epochs1, rate1, color=VANILLA_COLOR, linewidth=2,
            marker="o", markersize=5, label=title1)

    if history2:
        epochs2 = _extract(history2, "epoch")
        rate2   = _extract(history2, "conflict_rate")
        ax.plot(epochs2, rate2, color=PCGRAD_COLOR, linewidth=2,
                marker="o", markersize=5, linestyle="--", label=title2)

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Gradient Conflict Rate", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of Task Pairs in Conflict")
    ax.legend()
    ax.annotate("← lower = less conflict", xy=(0.02, 0.05),
                xycoords="axes fraction", fontsize=9, color="gray")
    plt.tight_layout()
    out = output_dir / "conflict_rate.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_gradient_score(
    history1, title1, output_dir,
    history2=None, title2=None,
) -> Path:
    """Combined gradient conflict score over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs1 = _extract(history1, "epoch")
    score1  = _extract(history1, "combined_gradient_score")

    ax.plot(epochs1, score1, color=VANILLA_COLOR, linewidth=2,
            marker="o", markersize=5, label=title1)

    if history2:
        epochs2 = _extract(history2, "epoch")
        score2  = _extract(history2, "combined_gradient_score")
        ax.plot(epochs2, score2, color=PCGRAD_COLOR, linewidth=2,
                marker="o", markersize=5, linestyle="--", label=title2)

    ax.set_title("Combined Gradient Conflict Score (GCS)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GCS  [0 = no conflict, 1 = maximum]")
    ax.legend()
    ax.annotate(
        "GCS = 0.4·rate + 0.4·severity + 0.2·norm(variance)",
        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8, color="gray", va="top",
    )
    plt.tight_layout()
    out = output_dir / "gradient_score.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_pcgrad_conflicts(history: List[Dict], title: str, output_dir: Path) -> Path:
    """PCGrad-specific: gradient conflicts per step each epoch."""
    if "pcgrad_conflicts_per_step" not in history[0]:
        print("  Skipping pcgrad_conflicts.png — no PCGrad conflict data in history.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = _extract(history, "epoch")
    cps    = _extract(history, "pcgrad_conflicts_per_step")

    ax.bar(epochs, cps, color=PCGRAD_COLOR, alpha=0.75, edgecolor="white",
           linewidth=0.5)
    ax.set_title(f"{title} — PCGrad Conflicts per Step", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Gradient Conflicts per Training Step")
    ax.set_xticks(epochs)
    plt.tight_layout()
    out = output_dir / "pcgrad_conflicts.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_summary_grid(
    history1, title1, output_dir,
    history2=None, title2=None,
) -> Path:
    """2×2 grid combining four key metrics — suitable for the paper/report."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "MTL Training Summary" + (f": {title1} vs {title2}" if title2 else f": {title1}"),
        fontsize=15, fontweight="bold",
    )

    e1 = _extract(history1, "epoch")
    e2 = _extract(history2, "epoch") if history2 else []

    def _plot(ax, key, ylabel, title_str, pct=False):
        y1 = _extract(history1, key)
        ax.plot(e1, y1, color=VANILLA_COLOR, linewidth=2, marker="o",
                markersize=4, label=title1)
        if history2:
            y2 = _extract(history2, key)
            ax.plot(e2, y2, color=PCGRAD_COLOR, linewidth=2, marker="o",
                    markersize=4, linestyle="--", label=title2)
        if pct:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(title_str, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

    _plot(axes[0, 0], "avg_val_loss", "Avg Val Loss", "Average Validation Loss")

    # MNLI — hardest task, most interesting to compare
    y1_mnli = [v * 100 for v in _extract_task(history1, "mnli", "acc")]
    axes[0, 1].plot(e1, y1_mnli, color=VANILLA_COLOR, linewidth=2,
                    marker="D", markersize=4, label=title1)
    if history2:
        y2_mnli = [v * 100 for v in _extract_task(history2, "mnli", "acc")]
        axes[0, 1].plot(e2, y2_mnli, color=PCGRAD_COLOR, linewidth=2,
                        marker="D", markersize=4, linestyle="--", label=title2)
    axes[0, 1].set_title("MNLI Accuracy (Hardest Task)", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend(fontsize=9)

    _plot(axes[1, 0], "conflict_rate", "Conflict Rate",
          "Gradient Conflict Rate", pct=True)
    axes[1, 0].set_ylim(0, 1)

    _plot(axes[1, 1], "combined_gradient_score", "GCS",
          "Combined Gradient Conflict Score")

    plt.tight_layout()
    out = output_dir / "summary_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── Final-epoch comparison table ──────────────────────────────────────────────

def print_final_table(history1, title1, history2=None, title2=None):
    """Print a clean comparison table of final-epoch metrics to stdout."""
    last1 = history1[-1]
    last2 = history2[-1] if history2 else None
    tasks = list(TASK_COLORS.keys())

    print("\n" + "=" * 70)
    print("FINAL EPOCH COMPARISON")
    print("=" * 70)
    header = f"{'Task':<14}  {title1:<20}"
    if last2:
        header += f"  {title2:<20}  Delta"
    print(header)
    print("-" * 70)

    for task in tasks:
        acc1 = last1["per_task_val"][task]["acc"] * 100
        row  = f"{task.upper():<14}  {acc1:>6.2f}%"
        if last2:
            acc2  = last2["per_task_val"][task]["acc"] * 100
            delta = acc2 - acc1
            sign  = "+" if delta >= 0 else ""
            row  += f"            {acc2:>6.2f}%            {sign}{delta:.2f}%"
        print(row)

    print("-" * 70)
    avg1 = np.mean([last1["per_task_val"][t]["acc"] for t in tasks]) * 100
    row  = f"{'AVG ACC':<14}  {avg1:>6.2f}%"
    if last2:
        avg2  = np.mean([last2["per_task_val"][t]["acc"] for t in tasks]) * 100
        delta = avg2 - avg1
        sign  = "+" if delta >= 0 else ""
        row  += f"            {avg2:>6.2f}%            {sign}{delta:.2f}%"
    print(row)

    print("-" * 70)
    for key, label in [("conflict_rate", "Conflict Rate"),
                        ("combined_gradient_score", "GCS")]:
        v1  = last1[key]
        row = f"{label:<14}  {v1:.4f}"
        if last2:
            v2    = last2[key]
            delta = v2 - v1
            sign  = "+" if delta >= 0 else ""
            row  += f"              {v2:.4f}              {sign}{delta:.4f}"
        print(row)

    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot MTL training results (CSCI 567)")
    p.add_argument("--history",  required=True,
                   help="Path to training_history.json (run 1 / vanilla).")
    p.add_argument("--history2", default=None,
                   help="Path to training_history.json for run 2 (PCGrad). Optional.")
    p.add_argument("--output",   required=True,
                   help="Directory to save all plots.")
    p.add_argument("--title",    default="Vanilla MTL",
                   help="Legend label for run 1.")
    p.add_argument("--title2",   default="PCGrad MTL",
                   help="Legend label for run 2.")
    return p.parse_args()


def main():
    args     = parse_args()
    history1 = load_history(args.history)
    history2 = load_history(args.history2) if args.history2 else None

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots -> {output_dir}")
    print(f"  Run 1 ({args.title}): {len(history1)} epochs")
    if history2:
        print(f"  Run 2 ({args.title2}): {len(history2)} epochs")

    plot_accuracy_per_task(history1, args.title, output_dir, history2, args.title2)
    plot_task_accuracy_overlay(history1, args.title, output_dir, history2, args.title2)
    plot_avg_val_loss(history1, args.title, output_dir, history2, args.title2)
    plot_conflict_rate(history1, args.title, output_dir, history2, args.title2)
    plot_gradient_score(history1, args.title, output_dir, history2, args.title2)
    plot_summary_grid(history1, args.title, output_dir, history2, args.title2)

    if history2:
        plot_pcgrad_conflicts(history2, args.title2, output_dir)

    print_final_table(history1, args.title, history2, args.title2)
    print(f"\nDone. All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
