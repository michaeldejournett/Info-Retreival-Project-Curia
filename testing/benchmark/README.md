# Curia Model Benchmark Suite

This benchmark is intentionally constrained to a fixed evaluation matrix.

## Fixed strategy

- Suites: `correctness`, `latency-stability`
- Models (exact set):
  - `gemini:gemma-3-27b-it`
  - `ollama:llama3:latest`
  - `huggingface:Qwen/Qwen2.5-1.5B-Instruct`
  - `huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  - `huggingface:Qwen/Qwen2.5-0.5B-Instruct`
  - `huggingface:google/flan-t5-base`
  - `huggingface:MBZUAI/LaMini-Flan-T5-248M`
- Datasets:
  - `testing/benchmark/datasets/queries_all_events.json`
  - `testing/benchmark/datasets/queries_rigorous_robustness.json`
  - `testing/benchmark/datasets/queries_rigorous_temporal.json`

`queries_all_events.json` is generated from `scraped/events.json` with one case per event title.

## Local setup (recommended)

1. Create and activate a Python virtual environment in the repo root.
2. Install benchmark dependencies:

```bash
pip install -r api/requirements.txt pytest
```

3. For local Hugging Face inference (default), ensure local runtime packages are installed:

```bash
pip install -r testing/benchmark/requirements-hf-local.txt
```

The runner defaults to local Hugging Face backend unless you override it.

## CLI usage

Run correctness suite on each dataset:

```bash
python -m testing.benchmark.run_benchmark --suite correctness --dataset-key all-events
python -m testing.benchmark.run_benchmark --suite correctness --dataset-key rigorous-robustness
python -m testing.benchmark.run_benchmark --suite correctness --dataset-key rigorous-temporal
```

Run latency/stability suite on each dataset:

```bash
python -m testing.benchmark.run_benchmark --suite latency-stability --dataset-key all-events
python -m testing.benchmark.run_benchmark --suite latency-stability --dataset-key rigorous-robustness
python -m testing.benchmark.run_benchmark --suite latency-stability --dataset-key rigorous-temporal
```

Useful overrides:

```bash
# Explicitly force hosted HF API backend instead of local
python -m testing.benchmark.run_benchmark \
  --suite correctness \
  --dataset-key rigorous-temporal \
  --huggingface-backend api

# Run only specific models (overrides the fixed model-set alias)
python -m testing.benchmark.run_benchmark \
  --suite correctness \
  --dataset-key all-events \
  --models "gemini:gemma-3-27b-it,ollama:llama3:latest"
```

## npm wrappers

From repo root:

```bash
npm run benchmark:correctness
npm run benchmark:correctness:all-events
npm run benchmark:correctness:robustness
npm run benchmark:correctness:temporal

npm run benchmark:latency
npm run benchmark:latency:all-events
npm run benchmark:latency:robustness
npm run benchmark:latency:temporal
```

## Output files

Each run creates a timestamped folder in `testing/benchmark/reports/`:

- `summary.json`
- `summary.md`
- `per_query.csv`
- `per_query.json`
- `gate_results.json`
- `gate_results.md`
- `visuals/*.png`:
  - `fig1_jaccard.png`
  - `fig2_temporal.png`
  - `fig3_latency.png`
  - `fig4_quality_vs_speed.png`
  - `fig5_per_query_heatmap.png`
  - `fig6_time_vs_quality_bubble.png`

Latest canonical figures are synchronized to the root `figures/` directory.

## Environment variables

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `OLLAMA_BASE_URL` (optional, default `http://localhost:11434`)
- `HUGGINGFACE_BACKEND` (optional, default `local`)
- `HUGGINGFACE_API_TOKEN` or `HF_TOKEN` (required for hosted HF API mode)
- `HUGGINGFACE_BASE_URL` (optional, hosted API endpoint)
- `HUGGINGFACE_LOCAL_DEVICE` (optional, `auto`/`cpu`/`cuda`/`mps`)
- `HUGGINGFACE_LOCAL_DTYPE` (optional, `auto`/`float16`/`bfloat16`/`float32`/`none`)

## Notes

- Canonical visuals require at least one Gemini model in a run because Gemini is the baseline comparator.
- The rigorous datasets are auto-derived (`metadata.label_status=heuristic-auto`) and are useful for stress testing.

## Optional Overrides

For best performance it is **heavily** reccomended to run all local models using PyTorch CUDA support. This allows for the HuggingFace LLM models to use GPU acceleration, vastly boosting the speed of the model. To set up CUDA support for Python, follow these [instructions](https://pytorch.org/get-started/locally/). Note that you may need to uninstall any existing PyTorch dependencies based on the individual technology or hardware of your system (e.g. Windows 10/11, GPU version)

To do so, run the command:

```powershell
pip uninstall torch
```

By default, CUDA is disabled to prevent issues with non-compliant systems. To enable CUDA GPU acceleration, see [package.json](../../package.json), and run commands with the `:cuda`