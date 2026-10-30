import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

import baselines
import config
import data
import evaluate

SYSTEMS = {
    "logreg": ("LogReg", "#0072B2"),
    "deberta": ("DeBERTa", "#D55E00"),
    "B": ("Retrieval (B)", "#009E73"),
}


def load_correct(system, name, budget):
    seeds = config.SEEDS
    if system in ["A_fewshot", "B"]:
        seeds = config.PROMPT_SEEDS
    if system == "A_zero" or (system, name, budget) == ("A_fewshot", "emotion", 30):
        seeds = [0]  # Emotion 30/class, seed 1: T4 out of memory

    test = data.load_test(name, config.HEADLINE_TEST_SIZE)
    rows = []
    for seed in seeds:
        path = baselines.results_path(system, name, seed, budget)
        df = pd.read_csv(path).head(config.HEADLINE_TEST_SIZE)
        assert df["idx"].tolist() == test.index.tolist(), f"test order mismatch: {path.name}"
        assert df["true"].tolist() == test["label"].tolist(), f"label mismatch: {path.name}"
        rows.append(evaluate.correct_array(df["true"], df["pred"]))
    return np.array(rows)


def plot_curve(ax, system, name, budgets, color, show_ci, marker="o", linestyle="-"):
    n_classes = len(data.load_labels(name))
    xs = [b * n_classes for b in budgets]
    stats = [evaluate.bootstrap_ci(load_correct(system, name, b)) for b in budgets]
    mean, low, high = np.array(stats).T
    ax.plot(xs, mean, marker=marker, ls=linestyle, color=color, lw=2,
            markersize=6, markeredgecolor="white", markeredgewidth=0.6)
    if show_ci:
        ax.errorbar(xs, mean, yerr=[mean - low, high - mean], fmt="none",
                    ecolor=color, capsize=3, alpha=0.5, lw=1)
    return xs, mean


def label_line(ax, x, y, label, color, height):
    ax.annotate(f"{label}  {y:.1%}", xy=(x, y), xytext=(1.04, height),
                textcoords=ax.get_yaxis_transform(), va="center", fontsize=10,
                color=color, annotation_clip=False,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8})


def make_figure(show_ci=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, name in zip(axes, config.DATASETS):
        n_classes = len(data.load_labels(name))
        budgets = config.DATASETS[name]["budgets"]
        xs = [b * n_classes for b in budgets]
        heights = {"logreg": 0.95, "deberta": 0.83, "B": 0.89} if name == "banking77" else {
            "logreg": 0.56, "deberta": 0.914, "B": 0.70}
        for system, (label, color) in SYSTEMS.items():
            _, means = plot_curve(ax, system, name, budgets, color, show_ci)
            label_line(ax, xs[-1], means[-1], label, color, heights[system])

        mean, low, high = evaluate.bootstrap_ci(load_correct("A_zero", name, 0))
        ax.axhline(mean, ls="--", color="#777777", lw=1.4)
        label_line(ax, xs[-1], mean, "Zero-shot (A)", "#777777",
                   mean if name == "banking77" else 0.62)
        if show_ci:
            ax.errorbar([xs[-1]], [mean], yerr=[[mean - low], [high - mean]],
                        fmt="none", ecolor="#777777", capsize=3, alpha=0.5)

        few_x, few_mean = plot_curve(ax, "A_fewshot", name, config.FEWSHOT_BUDGETS[name],
                                     "#A64D79", show_ci, marker="D", linestyle="--")
        few_label = "Few-shot (A)" + ("*" if name == "emotion" else "")
        ax.annotate(f"{few_label}  {few_mean[-1]:.1%}",
                    (few_x[-1], few_mean[-1]), xytext=(-8, 14),
                    textcoords="offset points", ha="right" if name == "emotion" else "left",
                    fontsize=10, color="#A64D79")
        if name == "emotion":
            test = data.load_test(name, config.HEADLINE_TEST_SIZE)
            dev = data.load_dev(name)
            majority = evaluate.majority_baseline(dev["label"], test["label"])
            ax.axhline(majority, ls=":", color="#BBBBBB", lw=1)
            ax.text(xs[0], majority + 0.012, f"Majority class: {majority:.1%}",
                    color="#888888", fontsize=9)

        ax.set_xscale("log")
        ax.set_xticks(xs, labels=[str(x) for x in xs])
        ax.minorticks_off()
        ax.set_xlim(xs[0] * 0.9, xs[-1] * 1.08)
        ax.set_ylim(0.3, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(f"{'Banking77' if name == 'banking77' else 'Emotion'} · {n_classes} classes",
                     loc="left", fontsize=14, pad=16)
        ax.set_xlabel("Total labeled examples (log scale)", labelpad=10)
        ax.grid(axis="y", color="#EEEEEE")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CCCCCC")
        ax.tick_params(axis="both", length=0, pad=8)
    axes[0].set_ylabel("Accuracy", labelpad=10)
    fig.suptitle("Accuracy vs. labeled data" + (" — 95% confidence intervals" if show_ci else ""),
                 x=0.06, ha="left", fontsize=18, y=0.98)
    fig.text(0.06, 0.90, f"Same {config.HEADLINE_TEST_SIZE} test items per dataset",
             fontsize=11, color="#666666")
    fig.text(0.06, 0.03, "LogReg / DeBERTa: 3 seeds; retrieval / few-shot: 2 seeds. "
             "* Emotion at 180 labels: seed 0 only (T4 memory limit).", fontsize=9, color="#666666")
    if show_ci:
        fig.text(0.06, 0.07, "Error bars: 95% bootstrap intervals over test items.",
                 fontsize=9, color="#666666")
    fig.subplots_adjust(left=0.06, right=0.85, bottom=0.18, top=0.78, wspace=0.75)
    filename = "learning_curves_with_ci.png" if show_ci else "learning_curves.png"
    path = config.ROOT / filename
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved", path.name)


if __name__ == "__main__":
    make_figure()
    make_figure(show_ci=True)
