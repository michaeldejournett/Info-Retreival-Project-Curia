# Benchmark Summary

| Model | Parse Success | Latency p50 (ms) | Latency p95 (ms) | Keyword F1 | Top-3 Hit | Timeout Rate |
|---|---:|---:|---:|---:|---:|---:|
| ollama:llama3:latest | 1.000 | 3493.8 | 4038.1 | 0.288 | 0.000 | 0.000 |
| gemini:gemma-3-27b-it | 0.995 | 8053.8 | 9436.2 | 0.056 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-1.5B-Instruct | 1.000 | 2680.9 | 3326.7 | 0.096 | 0.000 | 0.000 |
| huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.000 | 1303.5 | 1840.0 | 0.048 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-0.5B-Instruct | 1.000 | 1625.5 | 2460.7 | 0.050 | 0.000 | 0.000 |
| huggingface:google/flan-t5-base | 0.931 | 836.3 | 1091.6 | 0.000 | 0.000 | 0.000 |
| huggingface:MBZUAI/LaMini-Flan-T5-248M | 0.986 | 843.5 | 1502.3 | 0.000 | 0.000 | 0.000 |

## Visual Artifacts

- `testing/benchmark/reports/run-20260430-161415/visuals/fig1_jaccard.png`
- `testing/benchmark/reports/run-20260430-161415/visuals/fig2_temporal.png`
- `testing/benchmark/reports/run-20260430-161415/visuals/fig3_latency.png`
- `testing/benchmark/reports/run-20260430-161415/visuals/fig4_quality_vs_speed.png`
- `testing/benchmark/reports/run-20260430-161415/visuals/fig5_per_query_heatmap.png`
- `testing/benchmark/reports/run-20260430-161415/visuals/fig6_time_vs_quality_bubble.png`

## Figures Directory Sync

Synchronized files:
- `figures/fig1_jaccard.png`
- `figures/fig2_temporal.png`
- `figures/fig3_latency.png`
- `figures/fig4_quality_vs_speed.png`
- `figures/fig5_per_query_heatmap.png`
- `figures/fig6_time_vs_quality_bubble.png`
