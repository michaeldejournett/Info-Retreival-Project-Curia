# Curia Model Benchmark Suite

This benchmark compares model extraction quality and latency for Curia search.

Benchmark prompts are sourced from the same production template used by the API service to ensure prompt parity across providers.

## Scope

- Provider families: Gemini, Ollama, Hugging Face
- Primary focus: extraction outputs (`keywords`, `date/time` filters)
- Secondary focus: retrieval impact (`top-k` hit rates) using the local event corpus

## Quick start

Run smoke profile (small labeled dataset):

```bash
python -m testing.benchmark.run_benchmark --profile smoke
```

Run correctness suite (strict label validation + correctness gate profile):

```bash
python -m testing.benchmark.run_benchmark --suite correctness --profile smoke
```

Run latency/stability suite (repeated runs + latency/stability gate profile):

```bash
python -m testing.benchmark.run_benchmark --suite latency-stability --profile smoke
```

Run full profile (generates a 100-case template in-memory):

```bash
python -m testing.benchmark.run_benchmark --profile full
```

Run the first-pass Hugging Face model slate:

```bash
python -m testing.benchmark.run_benchmark --profile smoke --model-set hf-first-pass
```

Run local Hugging Face models in-process (no remote API calls):

```bash
python -m testing.benchmark.run_benchmark --profile smoke --model-set hf-local-lite --huggingface-backend local
```

Run specific models:

```bash
python -m testing.benchmark.run_benchmark \
  --profile smoke \
  --models "gemini:gemma-3-27b-it,ollama:llama3:latest"
```

## Model sets

- `baseline`: `gemini:gemma-3-27b-it,ollama:llama3:latest`
- `hf-router-chat`: `huggingface:google/gemma-4-31B-it,huggingface:Qwen/Qwen3.5-9B,huggingface:CohereLabs/c4ai-command-r7b-12-2024`
- `hf-local-lite`: `huggingface:MBZUAI/LaMini-Flan-T5-248M,huggingface:google/flan-t5-base,huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `hf-first-pass`: `huggingface:MBZUAI/LaMini-Flan-T5-248M,huggingface:google/flan-t5-base,huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0,huggingface:google/gemma-2-2b-it,huggingface:Qwen/Qwen2.5-3B-Instruct,huggingface:microsoft/Phi-3-mini-4k-instruct,huggingface:mistralai/Mistral-7B-Instruct-v0.2`
- `all`: `gemini:gemma-3-27b-it,huggingface:MBZUAI/LaMini-Flan-T5-248M,huggingface:google/flan-t5-base,huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0,huggingface:google/gemma-2-2b-it,huggingface:Qwen/Qwen2.5-3B-Instruct,huggingface:microsoft/Phi-3-mini-4k-instruct,huggingface:mistralai/Mistral-7B-Instruct-v0.2`

Hosted Hugging Face availability depends on providers enabled for your token. If a model reports `model_not_supported`, list available hosted models first:

```bash
python -m testing.benchmark.list_hf_models --limit 50
```

The `hf-first-pass` and `all` sets include legacy models that may work best with local backend or older hosted endpoints.

Use `--model-set <name>` to run a named set, or `--models` to override with an explicit list.

## Suite and gate options

- `--suite default|correctness|latency-stability`
  - `correctness`: enables strict label validation and correctness gate profile.
  - `latency-stability`: enables strict label validation, sets repeated runs (3) by default, and uses latency/stability gate profile.
- `--gate-profile none|correctness|latency-stability|all` to override gate behavior.
- `--enforce-label-quality` to validate dataset label completeness and formatting before running.
- `--expected-label-status <status>` to require dataset metadata label status (for example `approved-manual`).
- `--skip-visuals` to disable chart generation.

Write a 100-case labeling template file:

```bash
python -m testing.benchmark.run_benchmark \
  --write-template testing/benchmark/datasets/queries_full_template_100.json \
  --profile smoke
```

## Rigorous datasets

The benchmark now includes larger auto-derived datasets based on `scraped/events.json`:

- `testing/benchmark/datasets/queries_rigorous_retrieval.json` (96 cases)
- `testing/benchmark/datasets/queries_rigorous_temporal.json` (72 cases)
- `testing/benchmark/datasets/queries_rigorous_robustness.json` (72 cases)

Run correctness suite against each dataset:

```bash
python -m testing.benchmark.run_benchmark \
  --suite correctness \
  --dataset testing/benchmark/datasets/queries_rigorous_retrieval.json

python -m testing.benchmark.run_benchmark \
  --suite correctness \
  --dataset testing/benchmark/datasets/queries_rigorous_temporal.json

python -m testing.benchmark.run_benchmark \
  --suite correctness \
  --dataset testing/benchmark/datasets/queries_rigorous_robustness.json
```

These datasets use `metadata.label_status=heuristic-auto`. They are designed for stress testing and rapid iteration; manually review or relabel subsets before using them for final model selection decisions.

## Environment variables

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `OLLAMA_BASE_URL` (optional, default `http://localhost:11434`)
- `HUGGINGFACE_API_TOKEN` or `HF_TOKEN`
- `HUGGINGFACE_BASE_URL` (optional, default `https://router.huggingface.co/v1/chat/completions`)
- `HUGGINGFACE_BACKEND` (optional, `api` or `local`; default `api`)
- `HUGGINGFACE_LOCAL_DEVICE` (optional, `auto`/`cpu`/`cuda`/`mps`; default `auto`)
- `HUGGINGFACE_LOCAL_DTYPE` (optional, `auto`/`float16`/`bfloat16`/`float32`/`none`; default `auto`)
- `HUGGINGFACE_LOCAL_TRUST_REMOTE_CODE` (optional, `true`/`false`; default `false`)

Gemini and Hugging Face keys can be defined in project environment files (`.env`, `.env.local`, `backend/.env`, or `api/.env`).

If a shell variable exists but is empty (for example, `HF_TOKEN=""`), the benchmark loader will now replace it with the non-empty value from these `.env` files.

Some Hugging Face models may be gated and require accepted model terms on your Hugging Face account.

Legacy text-generation endpoint (`https://api-inference.huggingface.co/models`) can still be used by setting `HUGGINGFACE_BASE_URL` explicitly, but hosted model coverage is now primarily via the router chat-completions API.

When using `HUGGINGFACE_BACKEND=local`, model weights run on your machine via `transformers` + `torch` and no Hugging Face API token is required for public models.

## Local setup (Windows)

1. Create and activate a Python virtual environment in the repo root.
2. Install local Hugging Face runtime dependencies:

```bash
pip install -r testing/benchmark/requirements-hf-local.txt
```

3. Choose backend and hardware (PowerShell examples):

```powershell
$env:HUGGINGFACE_BACKEND = "local"
$env:HUGGINGFACE_LOCAL_DEVICE = "auto"
$env:HUGGINGFACE_LOCAL_DTYPE = "auto"
```

4. Run a lightweight local pass first:

```bash
python -m testing.benchmark.run_benchmark --profile smoke --model-set hf-local-lite --huggingface-backend local
```

5. Run full first-pass set locally (higher RAM/VRAM required):

```bash
python -m testing.benchmark.run_benchmark --profile smoke --model-set hf-first-pass --huggingface-backend local
```

6. For one-model debugging:

```bash
python -m testing.benchmark.run_benchmark --profile smoke --models "huggingface:MBZUAI/LaMini-Flan-T5-248M" --huggingface-backend local
```

If an API key is missing, the run still completes and records that model call as an error.

## Output files

Each run creates a timestamped folder in `testing/benchmark/reports/`:

- `summary.json` - machine-readable metrics per model
- `per_query.csv` - detailed output for every query/model run
- `summary.md` - quick comparison table
- `gate_results.json` - pass/fail checks by model for active gate profile
- `gate_results.md` - human-readable gate summary
- `visuals/*.png` - static comparison charts (when matplotlib is available)

For `--profile full` without a custom dataset, a `generated_full_template.json` is also written into the run directory for manual labeling.

## Labeling notes

The smoke dataset is only a seed. For decision-grade comparisons:

1. Generate a 100-case template with `--write-template`.
2. Fill `expected` fields manually (`keywords`, `date_from/date_to`, `time_from/time_to`, `relevant_event_urls`).
3. Re-run using `--dataset <your labeled file>`.

## Fairness guidance

- Keep `temperature=0.0` across models for repeatability.
- Use the same timeout budget and run count across all models.
- Include failures/timeouts in denominators when comparing stability.
