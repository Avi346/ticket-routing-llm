import json

import pandas as pd

import config


rows = []

for path in sorted(config.BENCHMARK_DIR.glob("*.json")):
    result = json.loads(path.read_text())

    rows.append({
        "dataset": result["dataset"],
        "system": result["system"],
        "budget": result["budget"],
        "batch_size": result["batch_size"],
        "p50_ms": result["latency_p50_ms"],
        "p95_ms": result["latency_p95_ms"],
        "requests_per_second": result["throughput_rps"],
        "input_tokens": result["mean_input_tokens"],
        "output_tokens": result["mean_output_tokens"],
        "peak_gpu_gb": result["peak_allocated_gpu_gb"],
    })

if not rows:
    raise SystemExit("No benchmark JSON files found. Pull the results first.")

summary = pd.DataFrame(rows).sort_values(
    ["dataset", "system", "budget"]
)

summary.to_csv(config.BENCHMARK_DIR / "summary.csv", index=False)
print(summary.round(3).to_string(index=False))