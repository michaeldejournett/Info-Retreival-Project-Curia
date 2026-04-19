from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from testing.benchmark.schemas import ModelRunSummary, QueryRunResult


CANONICAL_FIGURE_FILES: Tuple[str, ...] = (
    "fig1_jaccard.png",
    "fig2_temporal.png",
    "fig3_latency.png",
    "fig4_quality_vs_speed.png",
    "fig5_per_query_heatmap.png",
)


def _plot_dependency_available():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _safe_metric(value: float | None, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    return float(value)


def _model_key(provider: str, model_name: str) -> str:
    return f"{provider}:{model_name}"


def _display_label(summary: ModelRunSummary) -> str:
    return f"{summary.model_name}\n({summary.provider})"


def _run_key(row: QueryRunResult) -> Tuple[str, int]:
    run_index_raw = row.invocation.metadata.get("run_index", 0)
    try:
        run_index = int(run_index_raw)
    except (TypeError, ValueError):
        run_index = 0
    return row.case.case_id, run_index


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    lset = set(x.strip().lower() for x in left if x and x.strip())
    rset = set(x.strip().lower() for x in right if x and x.strip())
    if not lset and not rset:
        return 1.0
    if not lset or not rset:
        return 0.0
    return float(len(lset.intersection(rset)) / len(lset.union(rset)))


def _find_gemini_baseline_key(summaries: Sequence[ModelRunSummary]) -> str:
    for summary in summaries:
        if summary.provider.strip().lower() == "gemini":
            return _model_key(summary.provider, summary.model_name)

    raise ValueError(
        "Canonical visuals require at least one Gemini model in the run. "
        "Add gemini:<model> to --models or choose a model set that includes Gemini."
    )


def _build_baseline_index(rows: Sequence[QueryRunResult]) -> Dict[Tuple[str, int], QueryRunResult]:
    return {_run_key(row): row for row in rows}


def _keyword_overlap_against_baseline(
    summaries: Sequence[ModelRunSummary],
    model_results: Dict[str, List[QueryRunResult]],
    baseline_key: str,
) -> Tuple[Dict[str, float | None], Dict[str, Dict[Tuple[str, int], float]]]:
    baseline_rows = model_results.get(baseline_key) or []
    baseline_index = _build_baseline_index(baseline_rows)

    mean_scores: Dict[str, float | None] = {}
    per_query_scores: Dict[str, Dict[Tuple[str, int], float]] = {}

    for summary in summaries:
        key = _model_key(summary.provider, summary.model_name)
        rows = model_results.get(key) or []

        if key == baseline_key:
            mean_scores[key] = 1.0 if rows else None
            per_query_scores[key] = {_run_key(row): 1.0 for row in rows}
            continue

        values: List[float] = []
        item_scores: Dict[Tuple[str, int], float] = {}
        for row in rows:
            base = baseline_index.get(_run_key(row))
            if base is None:
                continue
            score = _jaccard(row.invocation.keywords, base.invocation.keywords)
            values.append(score)
            item_scores[_run_key(row)] = score

        mean_scores[key] = (sum(values) / len(values)) if values else None
        per_query_scores[key] = item_scores

    return mean_scores, per_query_scores


def _temporal_agreement_against_baseline(
    summaries: Sequence[ModelRunSummary],
    model_results: Dict[str, List[QueryRunResult]],
    baseline_key: str,
) -> Dict[str, Tuple[int, int]]:
    baseline_rows = model_results.get(baseline_key) or []
    baseline_index = _build_baseline_index(baseline_rows)

    fractions: Dict[str, Tuple[int, int]] = {}

    for summary in summaries:
        key = _model_key(summary.provider, summary.model_name)
        rows = model_results.get(key) or []

        if key == baseline_key:
            fractions[key] = (len(rows), len(rows))
            continue

        numerator = 0
        denominator = 0

        for row in rows:
            baseline_row = baseline_index.get(_run_key(row))
            if baseline_row is None:
                continue

            date_pred = (row.invocation.date_from, row.invocation.date_to)
            date_base = (baseline_row.invocation.date_from, baseline_row.invocation.date_to)
            time_pred = (row.invocation.time_from, row.invocation.time_to)
            time_base = (baseline_row.invocation.time_from, baseline_row.invocation.time_to)

            date_comparable = any(date_pred) and any(date_base)
            time_comparable = any(time_pred) and any(time_base)
            if not date_comparable and not time_comparable:
                continue

            denominator += 1
            date_ok = (not date_comparable) or (date_pred == date_base)
            time_ok = (not time_comparable) or (time_pred == time_base)
            if date_ok and time_ok:
                numerator += 1

        fractions[key] = (numerator, denominator)

    return fractions


def _sync_visual(
    plt,
    out_path: Path,
    artifacts: List[str],
) -> None:
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    artifacts.append(str(out_path))


def sync_canonical_figures(artifacts: Sequence[str], destination_dir: Path) -> Dict[str, object]:
    if not artifacts:
        return {"synced": [], "warning": ""}

    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    missing: List[str] = []

    for artifact in artifacts:
        src = Path(artifact)
        if src.name not in CANONICAL_FIGURE_FILES:
            continue
        if not src.exists():
            missing.append(src.name)
            continue

        dst = destination_dir / src.name
        try:
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        except Exception:
            shutil.copy2(src, dst)
        copied[src.name] = str(dst)

    ordered_synced = [copied[name] for name in CANONICAL_FIGURE_FILES if name in copied]
    warning = ""
    if missing:
        unique_missing = sorted(set(missing))
        warning = "some canonical figures were missing before sync: " + ", ".join(unique_missing)

    return {
        "synced": ordered_synced,
        "warning": warning,
    }


def generate_visual_artifacts(
    run_dir: Path,
    summaries: Sequence[ModelRunSummary],
    model_results: Dict[str, List[QueryRunResult]],
) -> Dict[str, object]:
    plt = _plot_dependency_available()
    if plt is None:
        return {
            "artifacts": [],
            "warning": "matplotlib is not installed; skipping chart generation",
        }

    if not summaries:
        return {"artifacts": [], "warning": "no model summaries available for chart generation"}

    baseline_key = _find_gemini_baseline_key(summaries)

    output_dir = run_dir / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[str] = []

    model_keys = [_model_key(s.provider, s.model_name) for s in summaries]
    model_labels = {_model_key(s.provider, s.model_name): _display_label(s) for s in summaries}
    color_map = {key: ("#2ecc71" if key == baseline_key else "#e67e22") for key in model_keys}

    mean_jaccard, per_query_jaccard = _keyword_overlap_against_baseline(
        summaries=summaries,
        model_results=model_results,
        baseline_key=baseline_key,
    )

    temporal_agreement = _temporal_agreement_against_baseline(
        summaries=summaries,
        model_results=model_results,
        baseline_key=baseline_key,
    )

    # fig1_jaccard.png
    fig1_keys = [key for key in model_keys if mean_jaccard.get(key) is not None]
    if not fig1_keys:
        raise ValueError("No comparable keyword outputs were available for fig1_jaccard.png generation")

    fig1_values = [float(mean_jaccard[key]) for key in fig1_keys]
    fig1_colors = [color_map[key] for key in fig1_keys]
    fig1_x = list(range(len(fig1_keys)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(fig1_x, fig1_values, color=fig1_colors, edgecolor="black", linewidth=1.2)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Gemini baseline")
    ax.set_xticks(fig1_x)
    ax.set_xticklabels([model_labels[key] for key in fig1_keys], fontsize=10)
    ax.set_ylabel("Mean KW Jaccard", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Keyword Expansion Overlap vs Gemini Baseline", fontsize=12, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, fig1_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    _sync_visual(plt, output_dir / "fig1_jaccard.png", artifacts)

    # fig2_temporal.png
    fig2_keys = [key for key in model_keys if temporal_agreement.get(key, (0, 0))[1] > 0]
    if not fig2_keys:
        raise ValueError("No comparable temporal outputs were available for fig2_temporal.png generation")

    fig2_values = []
    fig2_labels = []
    fig2_colors = []
    for key in fig2_keys:
        numerator, denominator = temporal_agreement[key]
        fig2_values.append((numerator / denominator) if denominator > 0 else 0.0)
        fig2_labels.append(f"{numerator}/{denominator}")
        fig2_colors.append(color_map[key])

    fig2_x = list(range(len(fig2_keys)))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(fig2_x, fig2_values, color=fig2_colors, edgecolor="black", linewidth=1.2)
    ax.set_xticks(fig2_x)
    ax.set_xticklabels([model_labels[key] for key in fig2_keys], fontsize=10)
    ax.set_ylabel("Temporal Accuracy (fraction)", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.set_title("Temporal Extraction Accuracy", fontsize=12, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    for bar, fraction_label in zip(bars, fig2_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            fraction_label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    _sync_visual(plt, output_dir / "fig2_temporal.png", artifacts)

    # fig3_latency.png
    latency_values = {
        _model_key(s.provider, s.model_name): max(s.latency_mean_ms / 1000.0, 0.001) for s in summaries
    }
    fig3_keys = [key for key in model_keys if key in latency_values]
    fig3_vals = [latency_values[key] for key in fig3_keys]
    fig3_colors = [color_map[key] for key in fig3_keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(fig3_keys)), fig3_vals, color=fig3_colors, edgecolor="black", linewidth=1.2)
    ax.set_yticks(range(len(fig3_keys)))
    ax.set_yticklabels([model_labels[key] for key in fig3_keys], fontsize=10)
    ax.set_xlabel("Latency (seconds, log scale)", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    max_val = max(fig3_vals) if fig3_vals else 1.0
    ax.set_xlim([0.01, max(300.0, max_val * 1.8)])
    ax.grid(axis="x", which="both", alpha=0.3)
    ax.set_title("Query Latency Comparison", fontsize=12, fontweight="bold", pad=15)

    for bar, value in zip(bars, fig3_vals):
        ax.text(
            value * 1.1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}s",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    _sync_visual(plt, output_dir / "fig3_latency.png", artifacts)

    # fig4_quality_vs_speed.png
    fig4_keys = fig1_keys
    fig4_x = [latency_values[key] for key in fig4_keys]
    fig4_y = [_safe_metric(mean_jaccard.get(key), fallback=0.0) for key in fig4_keys]
    fig4_sizes = [1800.0 if key == baseline_key else 350.0 for key in fig4_keys]
    fig4_colors = [color_map[key] for key in fig4_keys]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(fig4_x, fig4_y, s=fig4_sizes, c=fig4_colors, alpha=0.75, edgecolors="black", linewidth=1.5)

    for xval, yval, key in zip(fig4_x, fig4_y, fig4_keys):
        label = model_labels[key].split("\n", 1)[0]
        ax.annotate(label, xy=(xval, yval), xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Latency (seconds, log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Keyword Jaccard", fontsize=11, fontweight="bold")
    ax.set_xlim([0.01, max(300.0, max(fig4_x) * 1.8)])
    ax.set_ylim([-0.05, 1.15])
    ax.grid(True, alpha=0.3)
    ax.set_title("Quality vs Speed Trade-off", fontsize=12, fontweight="bold", pad=15)
    ax.text(
        0.015,
        1.08,
        "Fast + Accurate\n(ideal region)",
        fontsize=9,
        style="italic",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "yellow", "alpha": 0.3},
    )

    _sync_visual(plt, output_dir / "fig4_quality_vs_speed.png", artifacts)

    # fig5_per_query_heatmap.png
    baseline_rows = model_results.get(baseline_key) or []
    if not baseline_rows:
        raise ValueError("Cannot generate fig5_per_query_heatmap.png because Gemini baseline has no query rows")

    query_keys: List[Tuple[str, int]] = []
    query_labels: List[str] = []
    seen: set[Tuple[str, int]] = set()
    include_run_index = len({key[1] for key in (_run_key(row) for row in baseline_rows)}) > 1
    for row in baseline_rows:
        key = _run_key(row)
        if key in seen:
            continue
        seen.add(key)
        query_keys.append(key)
        label = row.case.query
        if include_run_index:
            label = f"{label} [run {key[1] + 1}]"
        query_labels.append(label)

    heatmap_matrix: List[List[float]] = []
    for qkey in query_keys:
        row_values: List[float] = []
        for model_key in model_keys:
            score = per_query_jaccard.get(model_key, {}).get(qkey)
            row_values.append(_safe_metric(score, fallback=0.0))
        heatmap_matrix.append(row_values)

    fig, ax = plt.subplots(figsize=(max(8, 4 + len(model_keys)), max(6, 0.6 * len(query_labels))))
    image = ax.imshow(heatmap_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(model_keys)))
    ax.set_xticklabels([model_labels[key].replace("\n", " ") for key in model_keys], rotation=30, ha="right")
    ax.set_yticks(range(len(query_labels)))
    ax.set_yticklabels(query_labels)
    ax.set_title("Per-Query Keyword Jaccard Across Models", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax.set_ylabel("Query", fontsize=11, fontweight="bold")

    for i, row_values in enumerate(heatmap_matrix):
        for j, value in enumerate(row_values):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Jaccard")
    _sync_visual(plt, output_dir / "fig5_per_query_heatmap.png", artifacts)

    return {"artifacts": artifacts, "warning": ""}
