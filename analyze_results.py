import numpy as np
import pandas as pd

import config
import data
import evaluate


def load_runs(name, system, budget):
    seeds = config.PROMPT_SEEDS if system in ["B", "A_fewshot"] else config.SEEDS
    if system == "A_zero" or (name, system, budget) == ("emotion", "A_fewshot", 30):
        seeds = [0]

    test = data.load_test(name, config.HEADLINE_TEST_SIZE)
    labels = data.load_labels(name)
    correct, f1, validity = [], [], []

    for seed in seeds:
        path = config.RESULTS_DIR / f"{system}_{name}_b{budget}_s{seed}.csv"
        frame = pd.read_csv(path).head(len(test))
        assert frame["idx"].tolist() == test.index.tolist(), path.name
        assert frame["true"].tolist() == test["label"].tolist(), path.name
        predictions = frame["pred"].where(frame["pred"].notna(), None).tolist()
        correct.append(evaluate.correct_array(frame["true"], predictions))

        if name == "emotion":
            f1.append(evaluate.macro_f1(frame["true"], predictions, labels))
        if "raw" in frame:
            parsed = [evaluate.parse_output(raw, labels) for raw in frame["raw"]]
            assert predictions == parsed, f"Parser mismatch: {path.name}"
            validity.append(sum(p is not None for p in parsed) / len(parsed))

    values = np.asarray(correct)
    mean, low, high = evaluate.bootstrap_ci(values)
    row = {
        "dataset": name,
        "system": system,
        "budget_per_class": budget,
        "total_labels": budget * len(labels),
        "seeds": len(seeds),
        "test_items": len(test),
        "accuracy": mean,
        "accuracy_ci_low": low,
        "accuracy_ci_high": high,
        "seed_accuracy_sd": values.mean(axis=1).std(ddof=1) if len(seeds) > 1 else None,
        "macro_f1": np.mean(f1) if f1 else None,
        "valid_category_json_rate": np.mean(validity) if validity else None,
    }
    return values, row


def main():
    metrics, comparisons, crossovers = [], [], []

    for name, settings in config.DATASETS.items():
        runs, rows = {}, {}
        configurations = [("A_zero", 0)]
        configurations += [("A_fewshot", b) for b in config.FEWSHOT_BUDGETS[name]]
        configurations += [
            (system, budget)
            for system in ["B", "C", "deberta", "logreg"]
            for budget in settings["budgets"]
        ]

        for system, budget in configurations:
            values, row = load_runs(name, system, budget)
            runs[system, budget] = values
            rows[system, budget] = row
            metrics.append(row)

        for system in ["C", "deberta"]:
            selected = None

            for budget in settings["budgets"]:
                candidates = [("A_zero", 0), ("B", budget)]
                if budget in config.FEWSHOT_BUDGETS[name]:
                    candidates.append(("A_fewshot", budget))
                best = max(candidates, key=lambda key: rows[key]["accuracy"])
                fine = rows[system, budget]
                prompted = rows[best]
                difference, low, high = evaluate.paired_bootstrap(
                    runs[system, budget], runs[best]
                )
                passes = fine["accuracy_ci_low"] > prompted["accuracy"]

                comparisons.append({
                    "dataset": name,
                    "system": system,
                    "budget_per_class": budget,
                    "total_labels": fine["total_labels"],
                    "best_prompting": best[0],
                    "prompting_budget": best[1],
                    "prompting_seeds": prompted["seeds"],
                    "fine_tuned_accuracy": fine["accuracy"],
                    "fine_tuned_ci_low": fine["accuracy_ci_low"],
                    "prompting_accuracy": prompted["accuracy"],
                    "accuracy_difference": difference,
                    "paired_ci_low": low,
                    "paired_ci_high": high,
                    "paired_ci_above_zero": low > 0,
                    "passes_roadmap_rule": passes,
                })

                if passes and selected is None:
                    selected = fine

            crossovers.append({
                "dataset": name,
                "system": system,
                "budget_per_class": selected["budget_per_class"] if selected else None,
                "total_labels": selected["total_labels"] if selected else None,
                "status": "observed" if selected else "not observed in tested budgets",
            })

    output = config.ROOT / "analysis"
    output.mkdir(exist_ok=True)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False)
    summary = pd.DataFrame(crossovers)
    summary[["budget_per_class", "total_labels"]] = summary[
        ["budget_per_class", "total_labels"]
    ].astype("Int64")
    summary.to_csv(output / "crossovers.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
