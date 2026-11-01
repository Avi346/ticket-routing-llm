import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

import baselines
import config
import data


def benchmark(name):
    seed = config.BENCHMARK_SEED
    budget = max(config.DATASETS[name]["budgets"])
    path = config.BENCHMARK_DIR / f"logreg_{name}_b{budget}_s{seed}.json"

    if path.exists():
        print("Already measured:", path.name)
        return

    x, y = baselines.get_train(name, seed, budget)
    classifier = LogisticRegression(
        C=config.LOGREG_BEST_C[name],
        max_iter=1000,
    ).fit(x, y)

    embedder = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    texts = data.load_dev_subsample(name)["text"].head(
        config.BENCHMARK_N
    ).tolist()

    def predict(batch):
        vectors = embedder.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return classifier.predict(vectors)

    for i in range(config.BENCHMARK_WARMUP):
        predict([texts[i % len(texts)]])

    latencies = []

    for text in texts:
        start = time.perf_counter()
        predict([text])
        latencies.append(time.perf_counter() - start)

    batch_size = 8

    for _ in range(config.BENCHMARK_WARMUP):
        predict(texts[:batch_size])

    start = time.perf_counter()

    for i in range(0, len(texts), batch_size):
        predict(texts[i:i + batch_size])

    elapsed = time.perf_counter() - start

    result = {
        "dataset": name,
        "system": "logreg",
        "budget": budget,
        "seed": seed,
        "device": "cpu",
        "cpu": platform.processor(),
        "n": len(texts),
        "batch_size": batch_size,
        "warmup": config.BENCHMARK_WARMUP,
        "latency_p50_ms": float(np.percentile(latencies, 50) * 1000),
        "latency_p95_ms": float(np.percentile(latencies, 95) * 1000),
        "throughput_rps": len(texts) / elapsed,
        "throughput_seconds": elapsed,
        "latency_seconds": latencies,
        "mean_input_tokens": None,
        "mean_output_tokens": None,
        "peak_allocated_gpu_gb": None,
        "includes_query_embedding": True,
        "dev_checksum": data.checksum(
            config.SPLITS_DIR / name / "dev_subsample.csv"
        ),
        "versions": {
            package: version(package)
            for package in [
                "torch", "sentence-transformers", "scikit-learn"
            ]
        },
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)

    print(name, {
        key: round(result[key], 3)
        for key in ["latency_p50_ms", "latency_p95_ms", "throughput_rps"]
    })


if __name__ == "__main__":
    import sys

    benchmark(sys.argv[1])