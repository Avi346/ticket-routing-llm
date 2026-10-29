import numpy as np
import pandas as pd
import config
import data

_model = None


def embed(name, key):
    # key is a split file name without .csv, e.g. "train_seed0", "dev", "test"
    path = config.EMBED_DIR / name / f"{key}.npy"
    if path.exists():
        return np.load(path)

    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)

    texts = data.read_split(name, f"{key}.csv")["text"].tolist()
    vecs = _model.encode(texts, normalize_embeddings=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs

from sklearn.linear_model import LogisticRegression

import evaluate


def get_train(name, seed, budget):
    df = data.read_split(name, f"train_seed{seed}.csv")
    x = embed(name, f"train_seed{seed}")
    keep = (df["rank_in_class"] < budget).values
    return x[keep], df["label"][keep].tolist()


def tune_logreg(name):
    x, y = get_train(name, config.TUNE_SEED, config.TUNE_BUDGET[name])
    dev = data.load_dev(name)
    dev_x = embed(name, "dev")
    best_c, best_acc = None, -1
    for c in config.LOGREG_C:
        clf = LogisticRegression(C=c, max_iter=1000).fit(x, y)
        acc = evaluate.accuracy(dev["label"].tolist(), clf.predict(dev_x))
        print(name, "C =", c, "dev acc =", round(acc, 4))
        if acc > best_acc:
            best_c, best_acc = c, acc
    return best_c

def results_path(system, name, seed, budget):
    return config.RESULTS_DIR / f"{system}_{name}_b{budget}_s{seed}.csv"


def save_preds(path, true, pred):
    path.parent.mkdir(exist_ok=True)
    pd.DataFrame({"true": true, "pred": pred}).to_csv(path, index_label="idx")


def run_logreg(name):
    test = data.load_test(name)
    test_x = embed(name, "test")
    for seed in config.SEEDS:
        for budget in config.DATASETS[name]["budgets"]:
            path = results_path("logreg", name, seed, budget)
            if path.exists():
                continue
            x, y = get_train(name, seed, budget)
            clf = LogisticRegression(C=config.LOGREG_BEST_C[name], max_iter=1000).fit(x, y)
            save_preds(path, test["label"], clf.predict(test_x))
            
import math 

def train_deberta(name, seed, budget, lr, eval_texts, min_steps=None):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)

    set_seed(seed)                      # BEFORE the model is created

    labels = data.load_labels(name)
    l2i = {l: i for i, l in enumerate(labels)}
    df = data.load_train(name, seed, budget)

    tok = AutoTokenizer.from_pretrained(config.ENCODER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.ENCODER_MODEL, num_labels=len(labels))

    def encode(texts):
        return tok(texts, truncation=True, max_length=config.DEBERTA_MAX_LEN)

    train_ds = Dataset.from_dict({**encode(df["text"].tolist()),
                                  "labels": [l2i[l] for l in df["label"]]})
    floor = min_steps if min_steps is not None else config.DEBERTA_MIN_STEPS
    steps = max(config.DEBERTA_EPOCHS * math.ceil(len(df) / config.DEBERTA_BATCH), floor)
    args = TrainingArguments(output_dir="tmp_deberta", max_steps=steps, learning_rate=lr,
                             per_device_train_batch_size=config.DEBERTA_BATCH,
                             per_device_eval_batch_size=64, fp16=True,
                             seed=seed, data_seed=seed, warmup_ratio=0.1,
                             save_strategy="no", report_to="none",
                             logging_strategy="steps", logging_steps=50)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, processing_class=tok)
    trainer.train()

    out = trainer.predict(Dataset.from_dict(encode(eval_texts))).predictions.argmax(-1)
    del trainer, model
    torch.cuda.empty_cache()
    return [labels[i] for i in out]


def tune_deberta(name):
    dev = data.load_dev(name)
    best_lr, best_acc = None, -1
    for lr in config.DEBERTA_LRS:
        pred = train_deberta(name, config.TUNE_SEED, config.TUNE_BUDGET[name], lr,
                             dev["text"].tolist(),min_steps=config.DEBERTA_MIN_STEPS)
        acc = evaluate.accuracy(dev["label"].tolist(), pred)
        print(name, "lr =", lr, "dev acc =", round(acc, 4))
        if acc > best_acc:
            best_lr, best_acc = lr, acc
    return best_lr


def run_deberta(name):
    test = data.load_test(name)
    for seed in config.SEEDS:
        for budget in config.DATASETS[name]["budgets"]:
            path = results_path("deberta", name, seed, budget)
            if path.exists():
                continue
            lr = config.DEBERTA_BEST_LR[name]
            pred = train_deberta(name, seed, budget, lr, test["text"].tolist(),min_steps=config.DEBERTA_MIN_STEPS)
            save_preds(path, test["label"], pred)
            print("saved", path.name)