# Benchmark Summary

| Model | Parse Success | Latency p50 (ms) | Latency p95 (ms) | Keyword F1 | Top-3 Hit | Timeout Rate |
|---|---:|---:|---:|---:|---:|---:|
| ollama:llama3:latest | 0.995 | 3420.5 | 3901.1 | 0.280 | 0.000 | 0.005 |
| gemini:gemma-3-27b-it | 1.000 | 4160.2 | 4695.4 | 0.078 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-1.5B-Instruct | 1.000 | 2703.4 | 3372.5 | 0.094 | 0.000 | 0.000 |
| huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.000 | 1278.9 | 1888.3 | 0.038 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-0.5B-Instruct | 1.000 | 1587.5 | 2656.4 | 0.050 | 0.000 | 0.000 |
| huggingface:google/flan-t5-base | 0.931 | 866.1 | 985.5 | 0.000 | 0.000 | 0.000 |
| huggingface:MBZUAI/LaMini-Flan-T5-248M | 0.986 | 875.6 | 1615.5 | 0.000 | 0.000 | 0.000 |

## Visual Artifacts

- `testing/benchmark/reports/run-20260430-201958/visuals/fig1_jaccard.png`
- `testing/benchmark/reports/run-20260430-201958/visuals/fig2_temporal.png`
- `testing/benchmark/reports/run-20260430-201958/visuals/fig3_latency.png`
- `testing/benchmark/reports/run-20260430-201958/visuals/fig4_quality_vs_speed.png`
- `testing/benchmark/reports/run-20260430-201958/visuals/fig5_per_query_heatmap.png`
- `testing/benchmark/reports/run-20260430-201958/visuals/fig6_time_vs_quality_bubble.png`

## Figures Directory Sync

Synchronized files:
- `figures/fig1_jaccard.png`
- `figures/fig2_temporal.png`
- `figures/fig3_latency.png`
- `figures/fig4_quality_vs_speed.png`
- `figures/fig5_per_query_heatmap.png`
- `figures/fig6_time_vs_quality_bubble.png`
