import json
from datetime import datetime, timezone
from importlib.metadata import version

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
import data
from benchmark import measure


def benchmark(name):
    assert torch.cuda.is_available(), "Use a T4 GPU runtime"
    assert "T4" in torch.cuda.get_device_name(0), "Use T4 for comparable timings"

    seed = config.BENCHMARK_SEED
    budget = max(config.DATASETS[name]["budgets"])
    folder = config.ROOT / "adapters" / "deberta" / name
    path = config.BENCHMARK_DIR / f"deberta_{name}_b{budget}_s{seed}.json"

    if path.exists():
        print("Already measured:", path.name)
        return

    stats = json.loads((folder / "training_stats.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(
        folder,
        dtype=torch.float32,
    ).to("cuda").eval()

    def predict(texts):
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.DEBERTA_MAX_LEN,
            return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            predictions = model(**inputs).logits.argmax(dim=-1)

        return predictions.cpu().tolist(), [], []

    texts = data.load_dev_subsample(name)["text"].head(
        config.BENCHMARK_N
    ).tolist()

    torch.cuda.reset_peak_memory_stats()
    result = measure(predict, texts, batch_size=8)

    result.update({
        "dataset": name,
        "system": "deberta",
        "budget": budget,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(0),
        "warmup": config.BENCHMARK_WARMUP,
        "precision": "fp32 weights with fp16 autocast",
        "training_seconds": stats["seconds"],
        "peak_allocated_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        "dev_checksum": data.checksum(
            config.SPLITS_DIR / name / "dev_subsample.csv"
        ),
        "versions": {
            package: version(package)
            for package in ["torch", "transformers"]
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


if __name__ == "__main__":
    import sys

    benchmark(sys.argv[1])