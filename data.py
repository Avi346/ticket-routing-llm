import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_raw(name):
    from datasets import load_dataset
    cfg = config.DATASETS[name]
    ds = load_dataset(cfg["hf_name"], cfg["hf_config"])
    names = ds["train"].features["label"].names
    raw = {}
    for split in ds:
        df = ds[split].to_pandas()
        df["label"] = df["label"].map(lambda i: names[i])
        raw[split] = df[["text", "label"]]
    return raw, names


def clean(text):
    return " ".join(str(text).lower().split())


def spread_evenly(df, seed):
    # shuffle so the first 500 rows have the same class mix as all 1000
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    pos = df.groupby("label").cumcount()
    size = df.groupby("label")["label"].transform("size")
    noise = np.random.default_rng(seed).random(len(df))
    return df.loc[((pos + noise) / size).sort_values().index].reset_index(drop=True)


def make_splits(name, raw):
    cfg = config.DATASETS[name]
    train = raw["train"].reset_index(drop=True)
    test_all = raw["test"].reset_index(drop=True)

    # remove train rows that also appear in test
    test_texts = set(test_all["text"].map(clean))
    train = train[~train["text"].map(clean).isin(test_texts)].reset_index(drop=True)

    if cfg["dev_source"] == "train":
        dev = train.groupby("label").sample(n=cfg["dev_per_class"], random_state=config.SPLIT_SEED)
        pool = train.drop(dev.index)
    else:
        dev = raw[cfg["dev_source"]]
        pool = train
    dev = dev.reset_index(drop=True)

    # remove pool rows that also appear in dev
    pool = pool[~pool["text"].map(clean).isin(set(dev["text"].map(clean)))].reset_index(drop=True)

    test, _ = train_test_split(test_all, train_size=config.TEST_SIZE,
                               stratify=test_all["label"], random_state=config.SPLIT_SEED)
    dev_sub, _ = train_test_split(dev, train_size=config.DEV_SUBSAMPLE_SIZE,
                                  stratify=dev["label"], random_state=config.SPLIT_SEED)

    splits = {"dev": dev, "dev_subsample": dev_sub.reset_index(drop=True),
              "test": spread_evenly(test, config.SPLIT_SEED)}

    # a budget of k per class = rows with rank_in_class < k, so smaller budgets are nested in bigger ones
    for seed in config.SEEDS:
        shuffled = pool.sample(frac=1, random_state=seed)
        shuffled["rank_in_class"] = shuffled.groupby("label").cumcount()
        keep = shuffled[shuffled["rank_in_class"] < max(cfg["budgets"])]
        splits[f"train_seed{seed}"] = keep.reset_index(drop=True)
    return splits


def find_near_duplicates(train_df, test_df):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    a = model.encode(train_df["text"].tolist(), normalize_embeddings=True)
    b = model.encode(test_df["text"].tolist(), normalize_embeddings=True)
    sim = b @ a.T
    close = sim.max(axis=1) >= config.NEAR_DUPLICATE_THRESHOLD
    return pd.DataFrame({
        "test_text": test_df["text"].values[close],
        "train_text": train_df["text"].values[sim.argmax(axis=1)[close]],
        "similarity": sim.max(axis=1)[close],
    })


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_splits(name, splits, labels, near_dups):
    folder = config.SPLITS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    for key, df in splits.items():
        df.to_csv(folder / f"{key}.csv", index=False)
    near_dups.to_csv(folder / "near_duplicates.csv", index=False)
    (folder / "labels.json").write_text(json.dumps(labels))

    info = {"class_counts": splits["train_seed0"]["label"].value_counts().to_dict()}
    try:
        from huggingface_hub import HfApi
        hf = HfApi().dataset_info(config.DATASETS[name]["hf_name"])
        info["revision"] = hf.sha
        info["license"] = hf.card_data.license if hf.card_data else None
    except Exception as e:
        info["revision"] = f"not found: {e}"
    (folder / "info.json").write_text(json.dumps(info, indent=2))

    sums = {f.name: checksum(f) for f in folder.iterdir()}
    (folder / "checksums.json").write_text(json.dumps(sums, indent=2))


def check_checksums(name):
    folder = config.SPLITS_DIR / name
    sums = json.loads((folder / "checksums.json").read_text())
    for filename, old in sums.items():
        if checksum(folder / filename) != old:
            raise RuntimeError(f"{filename} was changed after the split was built")


def read_split(name, filename):
    check_checksums(name)
    return pd.read_csv(config.SPLITS_DIR / name / filename)


def load_labels(name):
    check_checksums(name)
    return json.loads((config.SPLITS_DIR / name / "labels.json").read_text())


def load_test(name, n=None):
    test = read_split(name, "test.csv")
    return test if n is None else test.head(n)


def load_dev(name):
    return read_split(name, "dev.csv")


def load_dev_subsample(name):
    return read_split(name, "dev_subsample.csv")


def load_train(name, seed, budget):
    df = read_split(name, f"train_seed{seed}.csv")
    return df[df["rank_in_class"] < budget].reset_index(drop=True)


def build(name):
    if (config.SPLITS_DIR / name / "checksums.json").exists():
        print(name, "already built, skipping")
        return
    raw, labels = load_raw(name)
    splits = make_splits(name, raw)
    train_all = pd.concat([df for k, df in splits.items() if k.startswith("train")])
    near_dups = find_near_duplicates(train_all.drop_duplicates("text"), splits["test"])
    save_splits(name, splits, labels, near_dups)
    print(name, {k: len(v) for k, v in splits.items()}, "near duplicates:", len(near_dups))


if __name__ == "__main__":
    for n in sys.argv[1:] or config.DATASETS:
        build(n)