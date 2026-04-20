#!/bin/bash

DATASET="testing/benchmark/datasets/queries_rigorous_temporal.json"
COMMON_ARGS="--dataset $DATASET --gate-profile none --timeout-s 120"

echo "Starting sequential benchmark runs..."
echo ""

# Run Gemini
echo "[1/3] Running Gemini gemma-3-27b-it..."
python -m testing.benchmark.run_benchmark --models "gemini:gemma-3-27b-it" $COMMON_ARGS
echo "✓ Gemini complete"
echo ""

# Run Qwen 0.5B
echo "[2/3] Running Qwen2.5-0.5B (GPU)..."
python -m testing.benchmark.run_benchmark \
  --models "huggingface:Qwen/Qwen2.5-0.5B-Instruct" \
  --huggingface-backend local \
  --huggingface-local-device cuda \
  --huggingface-local-dtype float16 \
  $COMMON_ARGS
echo "✓ Qwen 0.5B complete"
echo ""

# Run TinyLlama  
echo "[3/3] Running TinyLlama-1.1B (GPU)..."
python -m testing.benchmark.run_benchmark \
  --models "huggingface:TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --huggingface-backend local \
  --huggingface-local-device cuda \
  --huggingface-local-dtype float16 \
  $COMMON_ARGS
echo "✓ TinyLlama complete"
echo ""

echo "All benchmarks complete!"
echo "Figures updated in: figures/"
ls -lh figures/
