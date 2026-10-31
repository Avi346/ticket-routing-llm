import json
import time
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np

import config


def measure(predict, texts, batch_size):
    import torch

    latencies = []
    input_tokens, output_tokens = [], []

    for i in range(config.BENCHMARK_WARMUP):
        predict([texts[i % len(texts)]])

    for text in texts:
        torch.cuda.synchronize()
        start = time.perf_counter()

        _, n_in, n_out = predict([text])

        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
        input_tokens.extend(n_in)
        output_tokens.extend(n_out)

    if batch_size == 1:
        elapsed = sum(latencies)
    else:
        for _ in range(config.BENCHMARK_WARMUP):
            predict(texts[:batch_size])

        input_tokens, output_tokens = [], []
        torch.cuda.synchronize()
        start = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            _, n_in, n_out = predict(texts[i:i + batch_size])
            input_tokens.extend(n_in)
            output_tokens.extend(n_out)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return {
        "n": len(texts),
        "batch_size": batch_size,
        "latency_p50_ms": float(np.percentile(latencies, 50) * 1000),
        "latency_p95_ms": float(np.percentile(latencies, 95) * 1000),
        "throughput_rps": len(texts) / elapsed,
        "throughput_seconds": elapsed,
        "mean_input_tokens": float(np.mean(input_tokens)),
        "mean_output_tokens": float(np.mean(output_tokens)),
        "latency_seconds": latencies,
    }


def benchmark_lora(name):
    from unsloth import FastLanguageModel
    import torch

    import data
    import finetune

    assert torch.cuda.is_available(), "Use a T4 GPU runtime"
    assert "T4" in torch.cuda.get_device_name(0), "Use T4 for comparable timings"

    seed = config.BENCHMARK_SEED
    budget = max(config.DATASETS[name]["budgets"])
    folder = config.ADAPTER_DIR / name / f"b{budget}_s{seed}"
    path = config.BENCHMARK_DIR / f"C_{name}_b{budget}_s{seed}.json"

    if path.exists():
        print("Already measured:", path.name)
        return

    stats = json.loads((folder / "training_stats.json").read_text())

    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(folder),
        max_seq_length=config.MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    tok = getattr(tok, "tokenizer", tok)

    texts = data.load_dev_subsample(name)["text"].head(
        config.BENCHMARK_N
    ).tolist()

    torch.cuda.reset_peak_memory_stats()
    result = measure(
        lambda batch: finetune.predict(model, tok, batch),
        texts,
        config.BATCH_SIZE["zero"],
    )

    result.update({
        "system": "C",
        "dataset": name,
        "budget": budget,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(0),
        "warmup": config.BENCHMARK_WARMUP,
        "training_seconds": stats["seconds"],
        "peak_allocated_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        "dev_checksum": data.checksum(
            config.SPLITS_DIR / name / "dev_subsample.csv"
        ),
        "versions": {
            package: version(package)
            for package in ["torch", "transformers", "unsloth", "trl"]
        },
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)

    print(name, {
        key: round(result[key], 3)
        for key in ["latency_p50_ms", "latency_p95_ms", "throughput_rps"]
    })
def benchmark_prompting(name, system, budget=None):
    import prompting
    import torch

    import baselines
    import data
    import prompts

    kinds = {"A_zero": "zero", "A_fewshot": "fewshot", "B": "retrieval"}
    kind = kinds[system]
    seed = config.BENCHMARK_SEED

    if budget is None:
        if system == "A_zero":
            budget = 0
        elif system == "A_fewshot":
            budget = min(config.FEWSHOT_BUDGETS[name])
        else:
            budget = max(config.DATASETS[name]["budgets"])

    path = config.BENCHMARK_DIR / f"{system}_{name}_b{budget}_s{seed}.json"

    if path.exists():
        print("Already measured:", path.name)
        return

    assert torch.cuda.is_available(), "Use a T4 GPU runtime"
    assert "T4" in torch.cuda.get_device_name(0), "Use T4 for comparable timings"

    prompting.load_model()
    labels = data.load_labels(name)
    pool_pairs = []

    if budget:
        pool = data.load_train(name, seed, budget)
        pool_pairs = list(zip(pool["text"], pool["label"]))

    if system == "B":
        from sentence_transformers import SentenceTransformer

        cached = config.EMBED_DIR / name / f"train_seed{seed}.npy"
        assert cached.exists(), "Pull the saved embeddings first"

        pool_x, _ = baselines.get_train(name, seed, budget)
        embedder = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")

    prefix = prompts.ANSWER_PREFIX if config.USE_ANSWER_PREFIX else ""

    def predict_batch(texts):
        selected = [pool_pairs] * len(texts)

        if system == "B":
            query = embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            top = np.argsort(
                -(query @ pool_x.T), axis=1, kind="stable"
            )[:, :config.RETRIEVAL_K]
            selected = [
                [pool_pairs[j] for j in row[::-1]]
                for row in top
            ]

        users = [
            prompts.build_prompt(text, labels, shots)
            for text, shots in zip(texts, selected)
        ]
        chats = [
            prompting.to_chat(user, config.USE_ANSWER_PREFIX)
            for user in users
        ]
        outputs = prompting.generate_batch(chats)

        return (
            [prefix + row[0] for row in outputs],
            [row[1] for row in outputs],
            [row[2] for row in outputs],
        )

    texts = data.load_dev_subsample(name)["text"].head(
        config.BENCHMARK_N
    ).tolist()

    torch.cuda.reset_peak_memory_stats()
    result = measure(predict_batch, texts, config.BATCH_SIZE[kind])

    packages = ["torch", "transformers", "unsloth", "trl"]
    if system == "B":
        packages.append("sentence-transformers")

    result.update({
        "system": system,
        "dataset": name,
        "budget": budget,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(0),
        "warmup": config.BENCHMARK_WARMUP,
        "training_seconds": 0,
        "includes_query_embedding": system == "B",
        "query_embedding_device": "cpu" if system == "B" else None,
        "peak_allocated_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        "dev_checksum": data.checksum(
            config.SPLITS_DIR / name / "dev_subsample.csv"
        ),
        "versions": {package: version(package) for package in packages},
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)

    print(system, name, {
        key: round(result[key], 3)
        for key in ["latency_p50_ms", "latency_p95_ms", "throughput_rps"]
    })

if __name__ == "__main__":
    import sys

    name = sys.argv[1]
    system = sys.argv[2] if len(sys.argv) > 2 else "C"

    if system == "C":
        benchmark_lora(name)
    else:
        budget = int(sys.argv[3]) if len(sys.argv) > 3 else None
        benchmark_prompting(name, system, budget)