import json

# Load both runs
with open('testing/benchmark/reports/run-20260429-025255/summary.json') as f:
    robustness = json.load(f)

with open('testing/benchmark/reports/run-20260429-192525/summary.json') as f:
    temporal = json.load(f)

print("=== ROBUSTNESS DATASET (run-20260429-025255) ===")
print("Dataset case count:", robustness['run_meta']['case_count'])
for s in robustness['summaries']:
    print(f"\n{s['model_name']}:")
    print(f"  parse_success: {s['parse_success_rate']:.4f}")
    print(f"  date_exact: {s['date_exact_rate']:.4f}")
    print(f"  time_exact: {s['time_exact_rate']:.4f}")
    print(f"  keyword_f1: {s['keyword_f1']:.4f}")
    print(f"  latency_mean_ms: {s['latency_mean_ms']:.0f}")
    print(f"  latency_p95_ms: {s['latency_p95_ms']:.0f}")

print("\n\n=== TEMPORAL DATASET (run-20260429-192525) ===")
print("Dataset case count:", temporal['run_meta']['case_count'])
for s in temporal['summaries']:
    print(f"\n{s['model_name']}:")
    print(f"  parse_success: {s['parse_success_rate']:.4f}")
    print(f"  date_exact: {s['date_exact_rate']:.4f}")
    print(f"  time_exact: {s['time_exact_rate']:.4f}")
    print(f"  keyword_f1: {s['keyword_f1']:.4f}")
    print(f"  latency_mean_ms: {s['latency_mean_ms']:.0f}")
    print(f"  latency_p95_ms: {s['latency_p95_ms']:.0f}")
