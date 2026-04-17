"""Plot precision/recall comparison charts for raw vs expanded searches.

Reads metrics/search_metrics.json (core) and metrics/search_metrics_extended.json
(P@K, PR curves, score distributions) and writes one PNG per chart to metrics/.
"""
import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
except Exception:
    print("matplotlib and numpy are required. Install with: pip install matplotlib numpy", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_metrics(path="metrics/search_metrics.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_extended(path="metrics/search_metrics_extended.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Extended metrics file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Chart 1 (original): Precision@50 and Recall@50 side by side
# ---------------------------------------------------------------------------

def plot_precision_recall(metrics, out_path="metrics/chart_precision_recall.png"):
    cats = [m["category"] for m in metrics]
    raw_prec = [m["raw_precision"] for m in metrics]
    exp_prec = [m["exp_precision"] for m in metrics]
    raw_rec  = [m["raw_recall"]    for m in metrics]
    exp_rec  = [m["exp_recall"]    for m in metrics]

    x = range(len(cats))
    width = 0.28

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.bar([i - width/2 for i in x], raw_prec, width, label="Raw")
    ax.bar([i + width/2 for i in x], exp_prec, width, label="Expanded")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision@50: Raw vs Expanded")
    ax.set_ylabel("Precision")
    ax.legend()
    for i, (r, e) in enumerate(zip(raw_prec, exp_prec)):
        ax.text(i - width/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + width/2, e + 0.02, f"{e:.2f}", ha="center", fontsize=8)

    ax2 = axes[1]
    ax2.bar([i - width/2 for i in x], raw_rec, width, label="Raw")
    ax2.bar([i + width/2 for i in x], exp_rec, width, label="Expanded")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Recall@50: Raw vs Expanded")
    ax2.set_ylabel("Recall")
    ax2.legend()
    for i, (r, e) in enumerate(zip(raw_rec, exp_rec)):
        ax2.text(i - width/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        ax2.text(i + width/2, e + 0.02, f"{e:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Chart 2: F1 Score per category
# ---------------------------------------------------------------------------

def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def plot_f1(metrics, out_path="metrics/chart_f1.png"):
    cats     = [m["category"] for m in metrics]
    raw_f1   = [f1(m["raw_precision"], m["raw_recall"]) for m in metrics]
    exp_f1   = [f1(m["exp_precision"], m["exp_recall"]) for m in metrics]

    x = range(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 0.9), 5))
    ax.bar([i - width/2 for i in x], raw_f1, width, label="Raw", color="steelblue")
    ax.bar([i + width/2 for i in x], exp_f1, width, label="Expanded", color="darkorange")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("F1 Score per Category: Raw vs Expanded")
    ax.set_ylabel("F1")
    ax.legend()
    for i, (r, e) in enumerate(zip(raw_f1, exp_f1)):
        ax.text(i - width/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + width/2, e + 0.02, f"{e:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Chart 3: Precision@K grouped bar (K = 1, 3, 5, 10) — averaged across cats
# ---------------------------------------------------------------------------

def plot_pak(extended, out_path="metrics/chart_pak.png"):
    ks = [1, 3, 5, 10]
    # average P@K across categories that have ground truth
    raw_means, exp_means = [], []
    for k in ks:
        raw_vals = [m["raw_pak"][str(k)] for m in extended if m["gt_count"] > 0]
        exp_vals = [m["exp_pak"][str(k)] for m in extended if m["gt_count"] > 0]
        raw_means.append(sum(raw_vals) / max(1, len(raw_vals)))
        exp_means.append(sum(exp_vals) / max(1, len(exp_vals)))

    x = range(len(ks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width/2 for i in x], raw_means, width, label="Raw", color="steelblue")
    ax.bar([i + width/2 for i in x], exp_means, width, label="Expanded", color="darkorange")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"P@{k}" for k in ks])
    ax.set_ylim(0, 1.15)
    ax.set_title("Mean Precision@K: Raw vs Expanded")
    ax.set_ylabel("Mean Precision")
    ax.legend()
    for i, (r, e) in enumerate(zip(raw_means, exp_means)):
        ax.text(i - width/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
        ax.text(i + width/2, e + 0.02, f"{e:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Chart 4: Precision–Recall curves (averaged across categories)
# ---------------------------------------------------------------------------

def _interpolate_pr(pr_points):
    """Return (recalls, precisions) arrays from a list of {k, recall, precision} dicts."""
    if not pr_points:
        return [], []
    recalls    = [p["recall"]    for p in pr_points]
    precisions = [p["precision"] for p in pr_points]
    return recalls, precisions


def plot_pr_curve(extended, out_path="metrics/chart_pr_curve.png"):
    # Build average PR curve by binning recall into 11 standard points [0..1]
    recall_levels = np.linspace(0, 1, 11)

    def avg_interp(all_pr_lists):
        interp_precs = []
        for pr_points in all_pr_lists:
            if not pr_points:
                continue
            recalls    = np.array([p["recall"]    for p in pr_points])
            precisions = np.array([p["precision"] for p in pr_points])
            # standard 11-point interpolation: for each recall level, max precision >= that recall
            ip = []
            for rl in recall_levels:
                mask = recalls >= rl
                ip.append(precisions[mask].max() if mask.any() else 0.0)
            interp_precs.append(ip)
        if not interp_precs:
            return np.zeros(len(recall_levels))
        return np.mean(interp_precs, axis=0)

    cats_with_gt = [m for m in extended if m["gt_count"] > 0]
    raw_avg = avg_interp([m["raw_pr_curve"] for m in cats_with_gt])
    exp_avg = avg_interp([m["exp_pr_curve"] for m in cats_with_gt])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_levels, raw_avg, "o-", label="Raw",      color="steelblue")
    ax.plot(recall_levels, exp_avg, "s-", label="Expanded", color="darkorange")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Mean Precision–Recall Curve: Raw vs Expanded")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Chart 5: Score distribution histogram (aggregate across all categories)
# ---------------------------------------------------------------------------

def plot_score_distribution(extended, out_path="metrics/chart_score_dist.png"):
    all_raw = []
    all_exp = []
    for m in extended:
        all_raw.extend(m["raw_scores"])
        all_exp.extend(m["exp_scores"])

    if not all_raw and not all_exp:
        print("No score data available — skipping score distribution chart.")
        return

    max_score = max(max(all_raw, default=1), max(all_exp, default=1))
    bins = range(1, int(max_score) + 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    axes[0].hist(all_raw, bins=bins, color="steelblue", edgecolor="white", align="left")
    axes[0].set_title("Raw Search: Score Distribution")
    axes[0].set_xlabel("Relevance Score")
    axes[0].set_ylabel("Number of Events")
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    axes[1].hist(all_exp, bins=bins, color="darkorange", edgecolor="white", align="left")
    axes[1].set_title("Expanded Search: Score Distribution")
    axes[1].set_xlabel("Relevance Score")
    axes[1].set_ylabel("Number of Events")
    axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.suptitle("Relevance Score Distributions (all categories combined)", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Chart 6: Ground-truth coverage per category
# ---------------------------------------------------------------------------

def plot_gt_coverage(metrics, out_path="metrics/chart_gt_coverage.png"):
    cats = [m["category"] for m in metrics]
    gt_counts = [m["gt_count"] for m in metrics]

    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 0.9), 5))
    bars = ax.bar(range(len(cats)), gt_counts, color="mediumseagreen", edgecolor="white")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Number of Events (ground truth)")
    ax.set_title("Ground-Truth Event Count per Category")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for bar, count in zip(bars, gt_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    metrics  = load_metrics()
    extended = load_extended()

    plot_precision_recall(metrics)
    plot_f1(metrics)
    plot_pak(extended)
    plot_pr_curve(extended)
    plot_score_distribution(extended)
    plot_gt_coverage(metrics)


if __name__ == "__main__":
    main()
