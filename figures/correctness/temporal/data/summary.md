# Benchmark Summary

| Model | Parse Success | Latency p50 (ms) | Latency p95 (ms) | Keyword F1 | Top-3 Hit | Timeout Rate |
|---|---:|---:|---:|---:|---:|---:|
| ollama:llama3:latest | 1.000 | 3490.4 | 4074.0 | 0.225 | 0.000 | 0.000 |
| gemini:gemma-3-27b-it | 1.000 | 8088.2 | 9787.7 | 0.065 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-1.5B-Instruct | 1.000 | 2744.6 | 3546.2 | 0.063 | 0.000 | 0.000 |
| huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 0.986 | 1782.7 | 1944.2 | 0.167 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-0.5B-Instruct | 0.944 | 2277.4 | 16186.9 | 0.098 | 0.000 | 0.000 |
| huggingface:google/flan-t5-base | 0.861 | 899.4 | 994.6 | 0.000 | 0.000 | 0.000 |
| huggingface:MBZUAI/LaMini-Flan-T5-248M | 0.986 | 835.4 | 889.0 | 0.000 | 0.000 | 0.000 |

## Visual Artifacts

- `testing/benchmark/reports/run-20260430-154151/visuals/fig1_jaccard.png`
- `testing/benchmark/reports/run-20260430-154151/visuals/fig2_temporal.png`
- `testing/benchmark/reports/run-20260430-154151/visuals/fig3_latency.png`
- `testing/benchmark/reports/run-20260430-154151/visuals/fig4_quality_vs_speed.png`
- `testing/benchmark/reports/run-20260430-154151/visuals/fig5_per_query_heatmap.png`
- `testing/benchmark/reports/run-20260430-154151/visuals/fig6_time_vs_quality_bubble.png`

## Figures Directory Sync

Synchronized files:
- `figures/fig1_jaccard.png`
- `figures/fig2_temporal.png`
- `figures/fig3_latency.png`
- `figures/fig4_quality_vs_speed.png`
- `figures/fig5_per_query_heatmap.png`
- `figures/fig6_time_vs_quality_bubble.png`
