import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import baselines
import config
import data
import evaluate

SYSTEMS = ["logreg", "deberta"]


def load_correct(system, name, budget):
    rows = []
    for seed in config.SEEDS:
        path = baselines.results_path(system, name, seed, budget)
        df = pd.read_csv(path).head(config.HEADLINE_TEST_SIZE)
        rows.append(evaluate.correct_array(df["true"], df["pred"]))
    return np.array(rows)


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, name in zip(axes, config.DATASETS):
    n_classes = len(data.load_labels(name))
    budgets = config.DATASETS[name]["budgets"]
    xs = [b * n_classes for b in budgets]
    for system in SYSTEMS:
        stats = [evaluate.bootstrap_ci(load_correct(system, name, b)) for b in budgets]
        mean, low, high = zip(*stats)
        ax.plot(xs, mean, marker="o", label=system)
        ax.fill_between(xs, low, high, alpha=0.2)
        print(name, system, [round(m, 3) for m in mean])
    if name == "emotion":
        test = data.load_test(name, config.HEADLINE_TEST_SIZE)
        ax.axhline(evaluate.majority_baseline(test["label"], test["label"]),
                   ls="--", color="gray", label="majority class")
    ax.set_xscale("log")
    ax.set_title(name)
    ax.set_xlabel("labeled examples (total)")
    ax.set_ylabel("accuracy (first 500 test items)")
    ax.legend()

plt.tight_layout()
plt.savefig("learning_curves_baselines.png", dpi=150)