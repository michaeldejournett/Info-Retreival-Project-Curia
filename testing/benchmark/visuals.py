from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from testing.benchmark.schemas import ModelInvocationResult, ModelRunSummary, QueryRunResult


CANONICAL_FIGURE_FILES: Tuple[str, ...] = (
    "fig1_jaccard.png",
    "fig2_temporal.png",
    "fig3_latency.png",
    "fig4_quality_vs_speed.png",
    "fig5_per_query_heatmap.png",
    "fig6_time_vs_quality_bubble.png",
)


_KNOWN_MODEL_PARAM_B: Dict[str, float] = {
    "gemini:gemma-3-27b-it": 27.0,
    "gemini:gemma-3-27b": 27.0,
    "huggingface:qwen/qwen2.5-1.5b-instruct": 1.5,
    "huggingface:qwen/qwen2.5-0.5b-instruct": 0.5,
    "huggingface:tinyllama/tinyllama-1.1b-chat-v1.0": 1.1,
    "huggingface:google/flan-t5-base": 0.25,
    "huggingface:mbzuai/lamini-flan-t5-248m": 0.248,
    "huggingface:sentence-transformers/all-minilm-l6-v2": 0.022,
    "ollama:llama3:latest": 8.0,
}


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


def _safe_positive_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _estimate_params_billions(summary: ModelRunSummary) -> float:
    details = summary.details or {}

    # Prefer explicit metadata if the runner/adapters start providing it.
    for key in (
        "params_b",
        "parameters_b",
        "parameter_count_b",
        "parameter_count_billions",
        "model_size_b",
    ):
        parsed = _safe_positive_float(details.get(key))
        if parsed is not None:
            return parsed

    provider_lower = summary.provider.strip().lower()
    model_lower = summary.model_name.strip().lower()
    known_key = f"{provider_lower}:{model_lower}"
    if known_key in _KNOWN_MODEL_PARAM_B:
        return _KNOWN_MODEL_PARAM_B[known_key]

    match_billions = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model_lower)
    if match_billions:
        return float(match_billions.group(1))

    match_millions = re.search(r"(\d+(?:\.\d+)?)\s*m\b", model_lower)
    if match_millions:
        return float(match_millions.group(1)) / 1000.0

    if "llama3" in model_lower:
        return 8.0
    if "flan-t5-base" in model_lower or "temporal_tagger" in model_lower:
        return 0.25
    if "minilm" in model_lower:
        return 0.022

    return 1.0


def _sqrt_scaled_bubble_sizes(values: Sequence[float], min_size: float = 260.0, max_size: float = 1900.0):
    import numpy as np  # type: ignore

    arr = np.array(values, dtype=float)
    safe = np.clip(arr, a_min=1e-6, a_max=None)
    roots = np.sqrt(safe)
    lo = float(roots.min())
    hi = float(roots.max())
    if hi - lo <= 1e-12:
        return np.full(shape=roots.shape, fill_value=(min_size + max_size) / 2.0)
    return min_size + (roots - lo) / (hi - lo) * (max_size - min_size)


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


def _results_overlap_against_baseline(
    summaries: Sequence[ModelRunSummary],
    model_results: Dict[str, List[QueryRunResult]],
    baseline_key: str,
    top_k: int = 10,
    date_matched_only: bool = False,
) -> Tuple[Dict[str, float | None], Dict[str, Dict[Tuple[str, int], float]]]:
    """Jaccard overlap of retrieved result URLs vs Gemini baseline result URLs.

    date_matched_only=True restricts to queries where the model extracted the same
    date range as Gemini, isolating keyword search quality from date extraction failure.
    """
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
            if date_matched_only:
                date_match = (
                    row.invocation.date_from == base.invocation.date_from
                    and row.invocation.date_to == base.invocation.date_to
                )
                if not date_match:
                    continue
            model_urls = set(row.predicted_urls[:top_k])
            base_urls = set(base.predicted_urls[:top_k])
            score = _jaccard(list(model_urls), list(base_urls))
            values.append(score)
            item_scores[_run_key(row)] = score

        mean_scores[key] = (sum(values) / len(values)) if values else None
        per_query_scores[key] = item_scores

    return mean_scores, per_query_scores


def _keyword_only_results_overlap(
    summaries: Sequence[ModelRunSummary],
    model_results: Dict[str, List[QueryRunResult]],
    baseline_key: str,
    events: List[Dict[str, object]],
    top_k: int = 10,
) -> Dict[str, float | None]:
    """Re-run keyword search without any date filter for both model and baseline,
    then compare result sets. Isolates keyword quality from date extraction."""
    from testing.benchmark.search_compat import search  # type: ignore

    baseline_rows = model_results.get(baseline_key) or []
    baseline_index = _build_baseline_index(baseline_rows)

    # Pre-compute baseline keyword-only URLs per query.
    base_kw_urls: Dict[Tuple[str, int], List[str]] = {}
    for row in baseline_rows:
        kws = row.invocation.keywords
        if kws:
            ranked = search(events, kws, top_k)
            base_kw_urls[_run_key(row)] = [str(e.get("url")) for _, e in ranked if e.get("url")]
        else:
            base_kw_urls[_run_key(row)] = []

    mean_scores: Dict[str, float | None] = {}
    for summary in summaries:
        key = _model_key(summary.provider, summary.model_name)
        rows = model_results.get(key) or []

        if key == baseline_key:
            mean_scores[key] = 1.0 if rows else None
            continue

        values: List[float] = []
        for row in rows:
            rk = _run_key(row)
            base_urls = set(base_kw_urls.get(rk) or [])
            kws = row.invocation.keywords
            if not kws:
                values.append(0.0)
                continue
            ranked = search(events, kws, top_k)
            model_urls = set(str(e.get("url")) for _, e in ranked if e.get("url"))
            values.append(_jaccard(list(model_urls), list(base_urls)))

        mean_scores[key] = (sum(values) / len(values)) if values else None

    return mean_scores


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

    # Use dataset ground truth temporal labels for a consistent denominator
    # across all models. This avoids denominator drift caused by baseline
    # extraction behavior on individual queries.
    temporal_case_keys = {
        _run_key(brow)
        for brow in baseline_rows
        if any(
            (
                brow.case.expected.date_from,
                brow.case.expected.date_to,
                brow.case.expected.time_from,
                brow.case.expected.time_to,
            )
        )
    }
    denominator = len(temporal_case_keys)

    fractions: Dict[str, Tuple[int, int]] = {}

    for summary in summaries:
        key = _model_key(summary.provider, summary.model_name)
        rows = model_results.get(key) or []

        if key == baseline_key:
            fractions[key] = (denominator, denominator)
            continue

        numerator = 0

        for row in rows:
            if _run_key(row) not in temporal_case_keys:
                continue

            baseline_row = baseline_index.get(_run_key(row))
            if baseline_row is None:
                continue

            date_base = (baseline_row.invocation.date_from, baseline_row.invocation.date_to)
            time_base = (baseline_row.invocation.time_from, baseline_row.invocation.time_to)
            date_pred = (row.invocation.date_from, row.invocation.date_to)
            time_pred = (row.invocation.time_from, row.invocation.time_to)
            date_ok = date_pred == date_base
            time_ok = time_pred == time_base
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
    events: Optional[List[Dict[str, object]]] = None,
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

    # Canonical order: best top3_hit_rate first (Gemini → strongest local → weakest).
    # Used consistently across all figures so the reader can track each model by color.
    import numpy as np  # type: ignore
    summaries_sorted = sorted(summaries, key=lambda s: _safe_metric(s.top3_hit_rate, fallback=-1.0), reverse=True)
    model_keys = [_model_key(s.provider, s.model_name) for s in summaries_sorted]
    model_labels = {_model_key(s.provider, s.model_name): _display_label(s) for s in summaries_sorted}

    # Gradient palette: plasma colormap from best (bright yellow) to worst (dark purple).
    cmap = plt.get_cmap("plasma")
    n = len(model_keys)
    gradient_colors = [cmap(0.85 - 0.65 * i / max(n - 1, 1)) for i in range(n)]
    color_map = {key: gradient_colors[i] for i, key in enumerate(model_keys)}

    # Result overlap restricted to date-matched queries — isolates keyword search quality.
    mean_jaccard, per_query_jaccard = _results_overlap_against_baseline(
        summaries=summaries,
        model_results=model_results,
        baseline_key=baseline_key,
        date_matched_only=True,
    )

    # End-to-end result overlap across all queries (used in fig4/fig5).
    mean_jaccard_all, _ = _results_overlap_against_baseline(
        summaries=summaries,
        model_results=model_results,
        baseline_key=baseline_key,
        date_matched_only=False,
    )

    temporal_agreement = _temporal_agreement_against_baseline(
        summaries=summaries,
        model_results=model_results,
        baseline_key=baseline_key,
    )

    # fig1_jaccard.png — Temporal / date extraction accuracy.
    fig1_keys = model_keys
    fig1_values = []
    fig1_labels = []
    for key in fig1_keys:
        numerator, denominator = temporal_agreement[key]
        fig1_values.append((numerator / denominator) if denominator > 0 else 0.0)
        fig1_labels.append(f"{numerator}/{denominator}")
    fig1_colors = [color_map[key] for key in fig1_keys]
    fig1_x = list(range(len(fig1_keys)))

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(fig1_keys)), 6))
    bars = ax.bar(fig1_x, fig1_values, color=fig1_colors, edgecolor="black", linewidth=1.2)
    ax.set_xticks(fig1_x)
    ax.set_xticklabels(
        [model_labels[key] for key in fig1_keys],
        fontsize=9, rotation=30, ha="right", rotation_mode="anchor",
    )
    fig.subplots_adjust(bottom=0.28)
    ax.set_ylabel("Fraction of Queries Correct", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.set_title("Date / Time Extraction Accuracy vs Gemini Baseline", fontsize=12, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)
    for bar, lbl in zip(bars, fig1_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")

    _sync_visual(plt, output_dir / "fig1_jaccard.png", artifacts)

    # fig2_temporal.png — Keyword search quality: re-run keyword-only search (no date
    # filter) for both model and Gemini, then compare result sets. This purely reflects
    # whether the extracted keywords retrieve the same events, independent of dates.
    if events:
        kw_only_scores = _keyword_only_results_overlap(
            summaries=summaries,
            model_results=model_results,
            baseline_key=baseline_key,
            events=events,
        )
        fig2_subtitle = "Keyword-only search (no date filter), top-10 results"
    else:
        kw_only_scores = mean_jaccard
        fig2_subtitle = "Date-correct queries only (events corpus unavailable)"

    fig2_keys = [key for key in model_keys if kw_only_scores.get(key) is not None]
    if not fig2_keys:
        raise ValueError("No comparable result outputs were available for fig2_temporal.png generation")

    fig2_values = [float(kw_only_scores[key]) for key in fig2_keys]
    fig2_colors = [color_map[key] for key in fig2_keys]
    fig2_x = list(range(len(fig2_keys)))

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(fig2_keys)), 6))
    bars = ax.bar(fig2_x, fig2_values, color=fig2_colors, edgecolor="black", linewidth=1.2)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Gemini baseline")
    ax.set_xticks(fig2_x)
    ax.set_xticklabels(
        [model_labels[key] for key in fig2_keys],
        fontsize=9, rotation=30, ha="right", rotation_mode="anchor",
    )
    fig.subplots_adjust(bottom=0.28)
    ax.set_ylabel("Mean Result Overlap (top-10)", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.15])
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(f"Keyword Search Quality\n{fig2_subtitle}", fontsize=12, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, fig2_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    _sync_visual(plt, output_dir / "fig2_temporal.png", artifacts)

    # fig3_latency.png
    latency_values = {
        _model_key(s.provider, s.model_name): max(s.latency_mean_ms / 1000.0, 0.001) for s in summaries
    }
    fig3_keys = list(reversed([key for key in model_keys if key in latency_values]))
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

    # fig4_quality_vs_speed.png — dual-axis bar chart: result overlap (bars) + latency (line)
    # Sort models by result overlap descending so the ranking is immediately legible.
    fig4_keys_sorted = sorted(
        fig1_keys,
        key=lambda k: _safe_metric(mean_jaccard_all.get(k), fallback=0.0),
        reverse=True,
    )
    fig4_overlap = [_safe_metric(mean_jaccard_all.get(k), fallback=0.0) for k in fig4_keys_sorted]
    fig4_latency = [latency_values[k] for k in fig4_keys_sorted]  # already in seconds
    fig4_colors_sorted = [color_map[k] for k in fig4_keys_sorted]
    fig4_xlabels = [model_labels[k].split("\n", 1)[0] for k in fig4_keys_sorted]
    fig4_x_pos = list(range(len(fig4_keys_sorted)))

    fig, ax1 = plt.subplots(figsize=(max(10, 2 * len(fig4_keys_sorted)), 6))
    bars = ax1.bar(fig4_x_pos, fig4_overlap, color=fig4_colors_sorted, edgecolor="black", linewidth=1.2, zorder=2)
    ax1.set_ylabel("Result Overlap vs Gemini", fontsize=11, fontweight="bold", color="black")
    ax1.set_ylim([0, 1.25])
    ax1.set_xticks(fig4_x_pos)
    ax1.set_xticklabels(fig4_xlabels, fontsize=9, rotation=30, ha="right", rotation_mode="anchor")
    ax1.grid(axis="y", alpha=0.3, zorder=1)

    for bar, val in zip(bars, fig4_overlap):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax2 = ax1.twinx()
    ax2.plot(fig4_x_pos, fig4_latency, color="#2980b9", marker="D", linewidth=2,
             markersize=7, label="Latency (s)", zorder=3)
    for xp, lat in zip(fig4_x_pos, fig4_latency):
        ax2.text(xp, lat + max(fig4_latency) * 0.04, f"{lat:.1f}s",
                 ha="center", va="bottom", fontsize=8, color="#2980b9", fontweight="bold")
    ax2.set_ylabel("Latency p50 (seconds)", fontsize=11, fontweight="bold", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax2.set_ylim([0, max(fig4_latency) * 1.35])
    ax2.legend(loc="upper right", fontsize=9)

    fig.subplots_adjust(bottom=0.28)
    ax1.set_title("Quality vs Speed Trade-off", fontsize=12, fontweight="bold", pad=15)

    _sync_visual(plt, output_dir / "fig4_quality_vs_speed.png", artifacts)

    # fig5_per_query_heatmap.png — average Jaccard per model across all queries.
    avg_values: List[float] = []
    for model_key in model_keys:
        scores = list(per_query_jaccard.get(model_key, {}).values())
        avg_values.append(sum(scores) / len(scores) if scores else 0.0)

    fig, ax = plt.subplots(figsize=(max(8, 4 + len(model_keys)), 5))
    image = ax.imshow(
        [avg_values], cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto"
    )
    ax.set_xticks(range(len(model_keys)))
    ax.set_xticklabels(
        [model_labels[key].replace("\n", " ") for key in model_keys],
        rotation=30, ha="right", fontsize=10,
    )
    ax.set_yticks([0])
    ax.set_yticklabels(["Avg Result Overlap"], fontsize=11, fontweight="bold")
    ax.set_title("Average Search Result Overlap vs Gemini Baseline", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Model", fontsize=11, fontweight="bold")
    # Make the single row visually thick by fixing cell height via axes position.
    fig.subplots_adjust(top=0.82, bottom=0.32, left=0.12, right=0.88)

    for j, value in enumerate(avg_values):
        ax.text(j, 0, f"{value:.2f}", ha="center", va="center", fontsize=13, fontweight="bold")

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Result Overlap", fontsize=10, fontweight="bold")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=9)
    _sync_visual(plt, output_dir / "fig5_per_query_heatmap.png", artifacts)

    # fig6_time_vs_quality_bubble.png — p50 latency vs top-3 quality, with
    # bubble area scaled by sqrt(parameter count) to keep large models visible
    # without overwhelming smaller ones.
    fig6_summaries = [s for s in summaries_sorted if s.top3_hit_rate is not None]
    if not fig6_summaries:
        raise ValueError("No models with top3_hit_rate were available for fig6_time_vs_quality_bubble.png generation")

    fig6_keys = [_model_key(s.provider, s.model_name) for s in fig6_summaries]
    fig6_latency_s = [max(s.latency_p50_ms / 1000.0, 0.001) for s in fig6_summaries]
    fig6_top3 = [float(s.top3_hit_rate) for s in fig6_summaries]
    fig6_params_b = [_estimate_params_billions(s) for s in fig6_summaries]
    fig6_sizes = _sqrt_scaled_bubble_sizes(fig6_params_b)
    fig6_colors = [color_map.get(key, "#7f8c8d") for key in fig6_keys]
    fig6_labels = [model_labels.get(key, key).split("\n", 1)[0] for key in fig6_keys]

    fig, ax = plt.subplots(figsize=(max(9, int(1.4 * len(fig6_keys))), 6))
    ax.scatter(
        fig6_latency_s,
        fig6_top3,
        s=fig6_sizes,
        c=fig6_colors,
        alpha=0.76,
        edgecolors="black",
        linewidth=1.2,
        zorder=3,
    )

    for latency_s, top3, label, params_b in zip(fig6_latency_s, fig6_top3, fig6_labels, fig6_params_b):
        ax.annotate(
            f"{label}\n{params_b:.2g}B",
            xy=(latency_s, top3),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            alpha=0.95,
        )

    ax.set_xscale("log")
    x_min = min(fig6_latency_s)
    x_max = max(fig6_latency_s)
    ax.set_xlim([max(0.01, x_min * 0.75), max(1.0, x_max * 1.8)])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Latency p50 (seconds, log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Top-3 Hit Rate", fontsize=11, fontweight="bold")
    ax.set_title("Top-3 Quality vs Time", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, zorder=1)

    ref_params = sorted({min(fig6_params_b), float(np.median(fig6_params_b)), max(fig6_params_b)})
    ref_sizes = _sqrt_scaled_bubble_sizes(ref_params)
    ref_handles = [
        ax.scatter([], [], s=size, c="#ffffff", edgecolors="black", linewidth=1.2, alpha=0.7, label=f"{val:.2g}B")
        for val, size in zip(ref_params, ref_sizes)
    ]
    ax.legend(handles=ref_handles, title="Approx params", loc="lower right", fontsize=8, title_fontsize=9)

    _sync_visual(plt, output_dir / "fig6_time_vs_quality_bubble.png", artifacts)

    return {"artifacts": artifacts, "warning": ""}
