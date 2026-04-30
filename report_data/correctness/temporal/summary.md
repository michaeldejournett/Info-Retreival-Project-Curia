# Benchmark Summary

| Model | Parse Success | Latency p50 (ms) | Latency p95 (ms) | Keyword F1 | Top-3 Hit | Timeout Rate |
|---|---:|---:|---:|---:|---:|---:|
| ollama:llama3:latest | 1.000 | 3474.2 | 4134.7 | 0.219 | 0.000 | 0.000 |
| gemini:gemma-3-27b-it | 1.000 | 4448.1 | 5046.4 | 0.075 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-1.5B-Instruct | 1.000 | 2751.3 | 3608.0 | 0.060 | 0.000 | 0.000 |
| huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 0.986 | 1792.3 | 2155.4 | 0.170 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-0.5B-Instruct | 0.958 | 2298.2 | 3058.1 | 0.087 | 0.000 | 0.000 |
| huggingface:google/flan-t5-base | 0.861 | 878.5 | 979.9 | 0.000 | 0.000 | 0.000 |
| huggingface:MBZUAI/LaMini-Flan-T5-248M | 0.986 | 814.8 | 1033.7 | 0.000 | 0.000 | 0.000 |

## Visual Artifacts

- `testing/benchmark/reports/run-20260430-195352/visuals/fig1_jaccard.png`
- `testing/benchmark/reports/run-20260430-195352/visuals/fig2_temporal.png`
- `testing/benchmark/reports/run-20260430-195352/visuals/fig3_latency.png`
- `testing/benchmark/reports/run-20260430-195352/visuals/fig4_quality_vs_speed.png`
- `testing/benchmark/reports/run-20260430-195352/visuals/fig5_per_query_heatmap.png`
- `testing/benchmark/reports/run-20260430-195352/visuals/fig6_time_vs_quality_bubble.png`

## Figures Directory Sync

Synchronized files:
- `figures/fig1_jaccard.png`
- `figures/fig2_temporal.png`
- `figures/fig3_latency.png`
- `figures/fig4_quality_vs_speed.png`
- `figures/fig5_per_query_heatmap.png`
- `figures/fig6_time_vs_quality_bubble.png`
