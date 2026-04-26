import os
import sys
import time
import statistics

# Ensure the repository root is on the import path when running this file directly.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Warning: matplotlib is not installed. Plot generation will be skipped.")

from api.search import base_terms, search


# Calculate top-k accuracy: fraction of cases where ground truth is in top-k predictions.
def top_k_accuracy(all_predictions, ground_truths, k):
    assert len(all_predictions) == len(ground_truths)
    hits = 0

    for preds, gt in zip(all_predictions, ground_truths):
        if gt in preds[:k]:
            hits += 1

    return hits / len(ground_truths) if ground_truths else 0.0

# Compute metrics for each model based on predictions, ground truths, response times, and parameter counts.
def compute_metrics(models_outputs):
    results = {}

    for name, data in models_outputs.items():
        preds = data.get("predictions", [])
        gts = data.get("ground_truths", [])
        times = data.get("response_times", [])
        params = data.get("param_count", 0)

        top1 = top_k_accuracy(preds, gts, 1)
        top3 = top_k_accuracy(preds, gts, 3)
        error_gap = top3 - top1
        avg_time = statistics.mean(times) if times else 0.0

        results[name] = {
            "top1": top1,
            "top3": top3,
            "error_gap": error_gap,
            "avg_response_time": avg_time,
            "param_count": params,
        }

    return results

# Plot top-1 and top-3 accuracy for each model.
def plot_topk(results):
    if plt is None:
        print("Skipping top-k accuracy plot: matplotlib is not installed.")
        return

    names = list(results.keys())
    top1 = [results[n]["top1"] * 100 for n in names]
    top3 = [results[n]["top3"] * 100 for n in names]

    x = range(len(names))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], top1, width=width, label="Top-1")
    plt.bar([i + width / 2 for i in x], top3, width=width, label="Top-3")

    plt.xticks(list(x), names)
    plt.ylabel("Accuracy (%)")
    plt.title("Top-1 vs Top-3 Accuracy")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("testing/plots/topk_accuracy.png")
    plt.show()

# Plot accuracy vs average response time for each model.
def plot_accuracy_vs_speed(results):
    if plt is None:
        print("Skipping accuracy vs speed plot: matplotlib is not installed.")
        return

    names = list(results.keys())
    times = [results[n]["avg_response_time"] for n in names]
    accuracies = [results[n]["top3"] * 100 for n in names]

    plt.figure()
    plt.scatter(times, accuracies)

    for i, name in enumerate(names):
        plt.text(times[i], accuracies[i], name)

    plt.xlabel("Average Response Time (seconds)")
    plt.ylabel("Top-3 Accuracy (%)")
    plt.title("Accuracy vs Speed")
    plt.grid()
    plt.savefig("testing/plots/accuracy_vs_speed.png")
    plt.show()

# Helper function to create a mock event for testing.
def make_event(title, url="u", start="2026-04-10T10:00:00"):
    return {
        "title": title,
        "group": "",
        "description": "",
        "location": "",
        "audience": "",
        "url": url,
        "start": start,
    }

# Measure the runtime of the search function with varying input sizes.
def measure_runtime():
    input_sizes = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
    runtimes = []

    terms = base_terms("music")

    for n in input_sizes:
        events = [make_event("music", url=f"u{i}") for i in range(n)]

        repeats = 5
        total_time = 0

        for _ in range(repeats):
            t0 = time.perf_counter()
            search(events, terms, top_n=3)
            total_time += time.perf_counter() - t0

        avg_time = total_time / repeats
        runtimes.append(avg_time)
        print(f"n={n}, time={avg_time:.6f}s")

    return input_sizes, runtimes

# Plot runtime vs input size.
def plot_runtime(input_sizes, runtimes):
    if plt is None:
        print("Skipping runtime plot: matplotlib is not installed.")
        return

    plt.figure()
    plt.plot(input_sizes, runtimes, marker="o")
    plt.xlabel("Input Size (# of events)")
    plt.ylabel("Running Time (seconds)")
    plt.title("Search Running Time vs Input Size")
    plt.grid()
    plt.savefig("testing/plots/runtime_vs_input.png")
    plt.show()

# Main function
if __name__ == "__main__":
    # Ground truths for 4 test cases
    gts = ["b", "x", "n", "z"]

    models_outputs = {
        "Gemma-27B": {
            "predictions": [["a", "b", "c"], ["x", "y", "z"], ["m", "n", "o"], ["p", "q", "r"]],
            "ground_truths": gts,
            "response_times": [7.5, 8.1, 7.9, 7.7], # High latency
            "param_count": 27.0,
        },
        "Llama3-8B": {
            "predictions": [["b", "a", "c"], ["x", "z", "y"], ["n", "m", "o"], ["z", "p", "q"]],
            "ground_truths": gts,
            "response_times": [3.2, 3.5, 3.1, 3.4],
            "param_count": 8.0,
        },
        "Qwen-1.5B": {
            "predictions": [["a", "c", "b"], ["y", "x", "z"], ["m", "o", "p"], ["q", "r", "z"]],
            "ground_truths": gts,
            "response_times": [1.2, 1.4, 1.3, 1.5],
            "param_count": 1.5,
        },
        "TinyLlama-1.1B": {
            "predictions": [["a", "e", "f"], ["y", "g", "h"], ["m", "o", "i"], ["q", "r", "s"]],
            "ground_truths": gts, # Will result in lower accuracy
            "response_times": [0.5, 0.6, 0.5, 0.7],
            "param_count": 1.1,
        },
        "Qwen-0.5B": {
            "predictions": [["e", "f", "g"], ["h", "i", "j"], ["k", "l", "m"], ["n", "o", "p"]],
            "ground_truths": gts,
            "response_times": [0.2, 0.3, 0.2, 0.3], # Very fast
            "param_count": 0.5,
        }
    }

    # 1. Compute metrics for all models
    metrics = compute_metrics(models_outputs)
    
    print(f"{'Model':<15} | {'Top-1':<7} | {'Top-3':<7} | {'Avg Time':<10}")
    print("-" * 50)
    for name, values in metrics.items():
        print(f"{name:<15} | {values['top1']:.2f}    | {values['top3']:.2f}    | {values['avg_response_time']:.3f}s")

    # 2. Run scalability benchmark
    sizes, times = measure_runtime()
    
    # 3. Generate all visualizations
    plot_runtime(sizes, times)
    plot_topk(metrics)
    plot_accuracy_vs_speed(metrics)