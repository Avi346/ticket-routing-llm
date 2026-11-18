import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

import config


SYSTEMS = {
    "logreg": ("BGE + LogReg (CPU)", "#0072B2"),
    "deberta": ("DeBERTa (T4)", "#D55E00"),
    "B": ("Retrieval B (T4)", "#009E73"),
    "C": ("LoRA C (T4)", "#6F4E99"),
}
TITLES = {"banking77": "Banking77", "emotion": "Emotion"}
OUTPUT = config.BENCHMARK_DIR / "plots"


def save(fig, name, note):
    fig.text(0.02, 0.025, note, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    fig.savefig(OUTPUT / name, dpi=200)
    plt.close(fig)


def comparison(table, column, title, xlabel, filename, note):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=15)

    for ax, name in zip(axes, config.DATASETS):
        subset = table[table["dataset"] == name].set_index("system")
        values = subset.loc[list(SYSTEMS), column]
        positions = np.arange(len(SYSTEMS))
        ax.barh(positions, values, color=[v[1] for v in SYSTEMS.values()], height=0.6)
        ax.set_yticks(positions, [v[0] for v in SYSTEMS.values()])
        ax.invert_yaxis()
        ax.set_title(f"{TITLES[name]} · {int(subset.iloc[0]['budget_per_class'])}/class")
        ax.set_xlabel(xlabel)

        if column == "mean_test_accuracy_pct":
            ax.set_xlim(0, 110)
            ax.set_xticks([0, 25, 50, 75, 100])
        else:
            ax.set_xlim(0, values.max() * 1.23)

        for position, value in zip(positions, values):
            label = f"{value:.1f}%" if column == "mean_test_accuracy_pct" else f"{value:,.1f}"
            ax.annotate(label, (value, position), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=10)

        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.15)

    save(fig, filename, note)


def break_even(table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("When serving-time savings recover LoRA training time", fontsize=15)

    for ax, row in zip(axes, table.itertuples()):
        assert row.status == "finite", "No finite intersection to plot"
        requests = np.linspace(0, row.break_even_requests * 2, 200)
        retrieval = requests * row.B_seconds_per_1000 / 1000 / 60
        lora = (row.lora_training_seconds + requests * row.C_seconds_per_1000 / 1000) / 60

        ax.plot(requests, retrieval, color=SYSTEMS["B"][1], lw=2, label="B: serving")
        ax.plot(requests, lora, color=SYSTEMS["C"][1], lw=2, label="C: training + serving")
        ax.axvline(row.break_even_requests, color="#777777", ls=":", lw=1)
        ax.set_title(f"{TITLES[row.dataset]} · about {row.break_even_requests:,} requests")
        ax.set_xlabel("Requests")
        ax.set_ylabel("Allocated compute time (minutes)")
        ax.set_xlim(0, requests[-1])
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=9)

    save(fig, "compute_break_even.png",
         "Same label budget; batch 8. Assumes sustained measured throughput and proportional compute allocation.\n"
         "Training one adapter; excludes idle time and setup. Estimated compute amortization, not financial savings.")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(config.BENCHMARK_DIR / "accuracy_speed.csv")
    comparison(table, "mean_test_accuracy_pct", "Accuracy at the largest label budget",
               "Test accuracy (%)", "accuracy.png",
               "First 500 test messages; mean across 2 seeds for B and 3 for other systems. Descriptive means, not significance tests.")
    comparison(table, "p50_ms", "Typical response time at the largest label budget",
               "Median single-request latency (ms) · lower is better", "latency.png",
               "200 dev messages; seed 0 timing. Includes tokenization/embedding and prediction. CPU and T4 are different deployment options.")
    break_even(pd.read_csv(config.BENCHMARK_DIR / "break_even.csv"))
    print("Saved 3 plots to", OUTPUT)


if __name__ == "__main__":
    main()
