"""Plot precision and recall comparison for raw vs expanded searches.

Reads `metrics/search_metrics.json` and writes `metrics/search_metrics.png`.
"""
import json
import os
import sys

try:
    import matplotlib.pyplot as plt
except Exception:
    print("matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
    raise


def load_metrics(path="metrics/search_metrics.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot(metrics, out_path="metrics/search_metrics.png"):
    cats = [m["category"] for m in metrics]
    raw_prec = [m["raw_precision"] for m in metrics]
    exp_prec = [m["exp_precision"] for m in metrics]
    raw_rec = [m["raw_recall"] for m in metrics]
    exp_rec = [m["exp_recall"] for m in metrics]

    x = range(len(cats))
    # reduce bar width so labels have more room
    width = 0.28

    # increase figure width for long category lists and give more vertical space
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.bar([i - width/2 for i in x], raw_prec, width, label="Raw precision")
    ax.bar([i + width/2 for i in x], exp_prec, width, label="Expanded precision")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision@50 (raw vs expanded)")
    ax.legend()

    # annotate
    for i, (r, e) in enumerate(zip(raw_prec, exp_prec)):
        ax.text(i - width/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
        ax.text(i + width/2, e + 0.02, f"{e:.2f}", ha="center", fontsize=9)

    ax2 = axes[1]
    ax2.bar([i - width/2 for i in x], raw_rec, width, label="Raw recall")
    ax2.bar([i + width/2 for i in x], exp_rec, width, label="Expanded recall")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(cats, rotation=35, ha="right", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Recall@50 (raw vs expanded)")
    ax2.legend()

    for i, (r, e) in enumerate(zip(raw_rec, exp_rec)):
        ax2.text(i - width/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
        ax2.text(i + width/2, e + 0.02, f"{e:.2f}", ha="center", fontsize=9)

    # give extra bottom margin to avoid overlapping rotated labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote chart to: {out_path}")


def main():
    metrics = load_metrics()
    plot(metrics)


if __name__ == "__main__":
    main()
