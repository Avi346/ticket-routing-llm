import json
import math

import pandas as pd

import config
import data


def load_benchmark(system, name, budget):
    path = config.BENCHMARK_DIR / f"{system}_{name}_b{budget}_s0.json"
    return json.loads(path.read_text())


def accuracy(name, system, budget):
    test = data.load_test(name, config.HEADLINE_TEST_SIZE)
    scores = []

    for path in sorted(config.RESULTS_DIR.glob(f"{system}_{name}_b{budget}_s*.csv")):
        predictions = pd.read_csv(path).head(len(test))
        assert predictions["idx"].tolist() == test.index.tolist(), path.name
        assert predictions["true"].tolist() == test["label"].tolist(), path.name
        scores.append(float(predictions["pred"].eq(predictions["true"]).mean()))

    assert scores, f"No predictions for {system}, {name}"
    return sum(scores) / len(scores), len(scores)


def main():
    comparisons, overview = [], []

    for name in config.DATASETS:
        budget = max(config.DATASETS[name]["budgets"])
        retrieval = load_benchmark("B", name, budget)
        lora = load_benchmark("C", name, budget)

        for key in ["dataset", "budget", "seed", "batch_size", "n", "warmup", "gpu", "dev_checksum"]:
            assert retrieval[key] == lora[key], f"Mismatched {key}: {name}"
        for package, expected in lora["versions"].items():
            assert retrieval["versions"][package] == expected, package
        assert lora["dev_checksum"] == data.checksum(
            config.SPLITS_DIR / name / "dev_subsample.csv"
        ), f"Dev split changed: {name}"

        training = lora["training_seconds"]
        assert math.isfinite(training) and training > 0
        for result in [retrieval, lora]:
            assert math.isfinite(result["throughput_rps"]) and result["throughput_rps"] > 0

        retrieval_seconds = 1 / retrieval["throughput_rps"]
        lora_seconds = 1 / lora["throughput_rps"]
        saving = retrieval_seconds - lora_seconds
        requests = math.ceil(training / saving) if saving > 0 else None

        comparisons.append({
            "dataset": name,
            "budget_per_class": budget,
            "batch_size": lora["batch_size"],
            "lora_training_seconds": training,
            "B_seconds_per_1000": retrieval_seconds * 1000,
            "C_seconds_per_1000": lora_seconds * 1000,
            "break_even_requests": requests,
            "status": "finite" if requests is not None else "no serving savings",
        })

        for system in ["B", "C", "deberta", "logreg"]:
            result = load_benchmark(system, name, budget)
            score, seeds = accuracy(name, system, budget)
            overview.append({
                "dataset": name,
                "system": system,
                "budget_per_class": budget,
                "mean_test_accuracy_pct": score * 100,
                "accuracy_seeds": seeds,
                "timing_seed": result["seed"],
                "device": result.get("gpu") or result["device"],
                "batch_size": result["batch_size"],
                "p50_ms": result["latency_p50_ms"],
                "p95_ms": result["latency_p95_ms"],
                "requests_per_second": result["throughput_rps"],
            })

    summary = pd.DataFrame(comparisons)
    summary["break_even_requests"] = summary["break_even_requests"].astype("Int64")
    summary.to_csv(config.BENCHMARK_DIR / "break_even.csv", index=False)
    pd.DataFrame(overview).to_csv(config.BENCHMARK_DIR / "accuracy_speed.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
