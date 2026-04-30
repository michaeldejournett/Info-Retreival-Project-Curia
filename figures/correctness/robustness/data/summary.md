# Benchmark Summary

| Model | Parse Success | Latency p50 (ms) | Latency p95 (ms) | Keyword F1 | Top-3 Hit | Timeout Rate |
|---|---:|---:|---:|---:|---:|---:|
| ollama:llama3:latest | 1.000 | 3501.7 | 4050.3 | 0.290 | 0.000 | 0.000 |
| gemini:gemma-3-27b-it | 0.250 | 411.5 | 8398.0 | 0.013 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-1.5B-Instruct | 1.000 | 2681.7 | 3375.6 | 0.098 | 0.000 | 0.000 |
| huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.000 | 1343.5 | 1964.3 | 0.048 | 0.000 | 0.000 |
| huggingface:Qwen/Qwen2.5-0.5B-Instruct | 1.000 | 1723.7 | 2724.0 | 0.051 | 0.000 | 0.000 |
| huggingface:google/flan-t5-base | 0.931 | 870.8 | 990.7 | 0.000 | 0.000 | 0.000 |
| huggingface:MBZUAI/LaMini-Flan-T5-248M | 0.986 | 827.8 | 1537.3 | 0.000 | 0.000 | 0.000 |

## Visual Artifacts

- `testing/benchmark/reports/run-20260430-145130/visuals/fig1_jaccard.png`
- `testing/benchmark/reports/run-20260430-145130/visuals/fig2_temporal.png`
- `testing/benchmark/reports/run-20260430-145130/visuals/fig3_latency.png`
- `testing/benchmark/reports/run-20260430-145130/visuals/fig4_quality_vs_speed.png`
- `testing/benchmark/reports/run-20260430-145130/visuals/fig5_per_query_heatmap.png`

## Figures Directory Sync

Synchronized files:
- `figures/fig1_jaccard.png`
- `figures/fig2_temporal.png`
- `figures/fig3_latency.png`
- `figures/fig4_quality_vs_speed.png`
- `figures/fig5_per_query_heatmap.png`
