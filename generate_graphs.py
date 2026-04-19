#!/usr/bin/env python3
"""Generate comparison charts for Curia model benchmarks.

Produces 5 figures comparing models by Jaccard, temporal accuracy, latency, and quality-vs-speed.
Hardcoded historical data requires no live models to run.
Optional: load JSON result files from compare_backends.py --save to overlay live runs.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

HISTORICAL_DATA = [
    {"short": "Gemini",     "label": "Gemini\n(gemma-3-27b-it)",        "params_B": 27.0,  "jaccard": 1.0,  "temporal_n": 8, "temporal_d": 8,  "latency_s": 7.8,   "status": "active"},
    {"short": "Qwen-0.5B",  "label": "Qwen2.5-0.5B\n(local default)",   "params_B": 0.5,   "jaccard": 0.00, "temporal_n": None,"temporal_d": 8, "latency_s": 3.2,   "status": "active"},
    {"short": "Qwen-1.5B",  "label": "Qwen2.5-1.5B\n(tested, not deployed)","params_B": 1.5, "jaccard": 0.18, "temporal_n": 3, "temporal_d": 7,  "latency_s": 129.0, "status": "tested"},
    {"short": "flan-t5",    "label": "flan-t5-base\n(abandoned)",        "params_B": 0.25,  "jaccard": 0.00, "temporal_n": None,"temporal_d": None,"latency_s": None,"status": "abandoned"},
    {"short": "t-tagger",   "label": "temporal_tagger\n(abandoned)",     "params_B": 0.25,  "jaccard": None, "temporal_n": 0, "temporal_d": 8,  "latency_s": None,  "status": "abandoned"},
    {"short": "MiniLM",     "label": "all-MiniLM-L6-v2\n(hybrid)",       "params_B": 0.022, "jaccard": None, "temporal_n": None,"temporal_d": None,"latency_s": 0.05,"status": "active"},
]

STATUS_COLORS = {"active": "#2ecc71", "tested": "#e67e22", "abandoned": "#95a5a6"}

plt.style.use("seaborn-v0_8-whitegrid")


def load_run_files(paths):
    """Load JSON result files from compare_backends.py --save runs."""
    runs = []
    for p in paths:
        try:
            with open(p) as f:
                runs.append(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: skipped {p}: {e}", flush=True)
    return runs


def fig1_jaccard_bar(out_dir):
    """Keyword Jaccard bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = [m for m in HISTORICAL_DATA if m["jaccard"] is not None]
    labels = [m["label"] for m in models]
    values = [m["jaccard"] for m in models]
    colors = [STATUS_COLORS[m["status"]] for m in models]

    bars = ax.bar(range(len(models)), values, color=colors, edgecolor="black", linewidth=1.2)

    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        label_text = f"{val:.2f}"
        if val == 0.00 and models[i]["short"] == "Qwen-0.5B":
            label_text += "*"
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, label_text,
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Gemini baseline")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean KW Jaccard", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Keyword Expansion Overlap vs Gemini Baseline", fontsize=12, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fig1_jaccard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig1_jaccard.png", flush=True)


def fig2_temporal_bar(out_dir):
    """Temporal accuracy bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = [m for m in HISTORICAL_DATA if m["temporal_n"] is not None]
    labels = [m["label"] for m in models]
    values = [m["temporal_n"] / m["temporal_d"] for m in models]
    fractions = [f"{m['temporal_n']}/{m['temporal_d']}" for m in models]
    colors = [STATUS_COLORS[m["status"]] for m in models]

    bars = ax.bar(range(len(models)), values, color=colors, edgecolor="black", linewidth=1.2)

    for i, (bar, frac) in enumerate(zip(bars, fractions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, frac,
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Temporal Accuracy (fraction)", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.set_title("Temporal Extraction Accuracy", fontsize=12, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fig2_temporal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig2_temporal.png", flush=True)


def fig3_latency_bar(out_dir):
    """Latency horizontal bar chart (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = [m for m in HISTORICAL_DATA if m["latency_s"] is not None]
    labels = [m["label"] for m in models]
    values = [m["latency_s"] for m in models]
    colors = [STATUS_COLORS[m["status"]] for m in models]

    bars = ax.barh(range(len(models)), values, color=colors, edgecolor="black", linewidth=1.2)

    for i, (bar, val) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width * 1.1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}s",
                ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Latency (seconds, log scale)", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim([0.01, 300])
    ax.grid(axis="x", which="both", alpha=0.3)
    ax.set_title("Query Latency Comparison", fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_latency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig3_latency.png", flush=True)


def fig4_quality_vs_speed(out_dir):
    """Quality vs Speed scatter (bubble chart)."""
    fig, ax = plt.subplots(figsize=(9, 7))

    models = [m for m in HISTORICAL_DATA if m["jaccard"] is not None and m["latency_s"] is not None]

    x = [m["latency_s"] for m in models]
    y = [m["jaccard"] for m in models]
    sizes = [max(30, m["params_B"] * 60) for m in models]
    colors = [STATUS_COLORS[m["status"]] for m in models]
    labels = [m["short"] for m in models]

    scatter = ax.scatter(x, y, s=sizes, c=colors, alpha=0.7, edgecolors="black", linewidth=1.5)

    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, xy=(xi, yi), xytext=(5, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold", alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Latency (seconds, log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Keyword Jaccard", fontsize=11, fontweight="bold")
    ax.set_xlim([0.01, 300])
    ax.set_ylim([-0.05, 1.15])
    ax.grid(True, alpha=0.3)
    ax.set_title("Quality vs Speed Trade-off", fontsize=12, fontweight="bold", pad=15)

    ax.text(0.015, 1.08, "Fast + Accurate\n(ideal region)", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3))

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_quality_vs_speed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig4_quality_vs_speed.png", flush=True)


def fig5_per_query_heatmap(live_runs, out_dir):
    """Per-query Jaccard heatmap (only if live runs exist)."""
    if not live_runs:
        print("  fig5_per_query_heatmap.png (skipped — no live run files)", flush=True)
        return

    queries = live_runs[0]["queries"]
    run_labels = []
    heatmap_data = []

    for run in live_runs:
        label = f"{run['model_local'].split('/')[-1]}\n{run['run_id'].replace('run_', '')}"
        run_labels.append(label)
        jaccards = [record["jaccard"] for record in run["per_query"]]
        heatmap_data.append(jaccards)

    data_matrix = np.array(heatmap_data).T

    fig, ax = plt.subplots(figsize=(max(8, 4 + len(live_runs)), 6))
    sns.heatmap(data_matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                xticklabels=run_labels, yticklabels=queries, cbar_kws={"label": "Jaccard"},
                ax=ax, linewidths=0.5, linecolor="gray")

    ax.set_title("Per-Query Keyword Jaccard Across Runs", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Run (model + timestamp)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Query", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "fig5_per_query_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig5_per_query_heatmap.png", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison charts for Curia model benchmarks."
    )
    parser.add_argument(
        "run_files", nargs="*",
        help="Optional JSON result files from compare_backends.py --save"
    )
    parser.add_argument(
        "--out-dir", default="figures",
        help="Output directory for PNG files (default: figures/)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    live_runs = load_run_files(args.run_files) if args.run_files else []

    print(f"Generating figures in {out_dir}/:", flush=True)
    fig1_jaccard_bar(out_dir)
    fig2_temporal_bar(out_dir)
    fig3_latency_bar(out_dir)
    fig4_quality_vs_speed(out_dir)
    fig5_per_query_heatmap(live_runs, out_dir)
    print(f"Done.", flush=True)


if __name__ == "__main__":
    main()
