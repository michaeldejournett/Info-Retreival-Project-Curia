# Benchmark Summary

| Model | Parse Success | Latency p50 (ms) | Latency p95 (ms) | Keyword F1 | Top-3 Hit | Timeout Rate |
|---|---:|---:|---:|---:|---:|---:|
| ollama:llama3:latest | 1.000 | 3400.8 | 3876.6 | 0.224 | 0.000 | 0.000 |
| gemini:gemma-3-27b-it | 1.000 | 4346.0 | 4992.7 | 0.078 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-1.5B-Instruct | 1.000 | 2693.5 | 3464.6 | 0.060 | 0.000 | 0.000 |
| huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 0.986 | 1689.1 | 2187.1 | 0.170 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-0.5B-Instruct | 0.986 | 2177.1 | 2858.7 | 0.086 | 0.000 | 0.000 |
| huggingface:google/flan-t5-base | 0.861 | 824.8 | 1085.1 | 0.000 | 0.000 | 0.000 |
| huggingface:MBZUAI/LaMini-Flan-T5-248M | 0.986 | 777.0 | 1015.2 | 0.000 | 0.000 | 0.000 |

## Visual Artifacts

- `testing/benchmark/reports/run-20260430-223957/visuals/fig1_jaccard.png`
- `testing/benchmark/reports/run-20260430-223957/visuals/fig2_temporal.png`
- `testing/benchmark/reports/run-20260430-223957/visuals/fig3_latency.png`
- `testing/benchmark/reports/run-20260430-223957/visuals/fig4_quality_vs_speed.png`
- `testing/benchmark/reports/run-20260430-223957/visuals/fig5_per_query_heatmap.png`
- `testing/benchmark/reports/run-20260430-223957/visuals/fig6_time_vs_quality_bubble.png`

## Figures Directory Sync

Synchronized files:
- `figures/fig1_jaccard.png`
- `figures/fig2_temporal.png`
- `figures/fig3_latency.png`
- `figures/fig4_quality_vs_speed.png`
- `figures/fig5_per_query_heatmap.png`
- `figures/fig6_time_vs_quality_bubble.png`
