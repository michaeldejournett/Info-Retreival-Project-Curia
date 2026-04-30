# Benchmark Summary

| Model | Parse Success | Latency p50 (ms) | Latency p95 (ms) | Keyword F1 | Top-3 Hit | Timeout Rate |
|---|---:|---:|---:|---:|---:|---:|
| ollama:llama3:latest | 1.000 | 3495.0 | 4114.9 | 0.278 | 0.000 | 0.000 |
| gemini:gemma-3-27b-it | 1.000 | 4335.4 | 6607.5 | 0.077 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-1.5B-Instruct | 1.000 | 2759.5 | 3288.7 | 0.099 | 0.000 | 0.000 |
| huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.000 | 1241.4 | 1795.2 | 0.048 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-0.5B-Instruct | 1.000 | 1602.9 | 2691.7 | 0.041 | 0.000 | 0.000 |
| huggingface:google/flan-t5-base | 0.931 | 861.6 | 959.1 | 0.000 | 0.000 | 0.000 |
| huggingface:MBZUAI/LaMini-Flan-T5-248M | 0.986 | 834.6 | 1568.4 | 0.000 | 0.000 | 0.000 |

## Visual Artifacts

- `testing/benchmark/reports/run-20260430-193335/visuals/fig1_jaccard.png`
- `testing/benchmark/reports/run-20260430-193335/visuals/fig2_temporal.png`
- `testing/benchmark/reports/run-20260430-193335/visuals/fig3_latency.png`
- `testing/benchmark/reports/run-20260430-193335/visuals/fig4_quality_vs_speed.png`
- `testing/benchmark/reports/run-20260430-193335/visuals/fig5_per_query_heatmap.png`
- `testing/benchmark/reports/run-20260430-193335/visuals/fig6_time_vs_quality_bubble.png`

## Figures Directory Sync

Synchronized files:
- `figures/fig1_jaccard.png`
- `figures/fig2_temporal.png`
- `figures/fig3_latency.png`
- `figures/fig4_quality_vs_speed.png`
- `figures/fig5_per_query_heatmap.png`
- `figures/fig6_time_vs_quality_bubble.png`
