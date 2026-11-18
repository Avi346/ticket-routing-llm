from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import config
import data
import evaluate


OUTPUT = config.ROOT / "analysis"
SYSTEMS = ["B", "C", "deberta", "logreg"]


def load_predictions(name, system, budget, seed, test, labels):
    path = config.RESULTS_DIR / f"{system}_{name}_b{budget}_s{seed}.csv"
    frame = pd.read_csv(path).head(len(test))
    assert frame["idx"].tolist() == test.index.tolist(), path.name
    assert frame["true"].tolist() == test["label"].tolist(), path.name
    if "raw" in frame:
        parsed = [evaluate.parse_output(raw, labels) for raw in frame["raw"]]
        assert frame["pred"].where(frame["pred"].notna(), None).tolist() == parsed, path.name
    return frame["pred"].fillna("<invalid>").to_numpy()


def main():
    classes, confusions, overlaps, examples, majority = [], [], [], [], []

    for name, settings in config.DATASETS.items():
        budget = max(settings["budgets"])
        test = data.load_test(name, config.HEADLINE_TEST_SIZE)
        labels = data.load_labels(name)
        true = test["label"].to_numpy()
        predictions = {}

        majority_label = data.load_dev(name)["label"].value_counts().idxmax()
        majority.append({
            "dataset": name,
            "dev_majority_label": majority_label,
            "test_accuracy": float((true == majority_label).mean()),
        })

        for system in SYSTEMS:
            seeds = config.PROMPT_SEEDS if system == "B" else config.SEEDS
            scores, mistakes = [], Counter()

            for seed in seeds:
                pred = load_predictions(name, system, budget, seed, test, labels)
                predictions[system, seed] = pred
                precision, recall, f1, support = precision_recall_fscore_support(
                    true, pred, labels=labels, zero_division=0
                )
                scores.append(np.array([precision, recall, f1]))
                mistakes.update(zip(true[true != pred], pred[true != pred]))

            precision, recall, f1 = np.mean(scores, axis=0)
            for i, label in enumerate(labels):
                classes.append({
                    "dataset": name,
                    "system": system,
                    "class": label,
                    "test_support": int(support[i]),
                    "seeds": len(seeds),
                    "mean_precision": precision[i],
                    "mean_recall": recall[i],
                    "mean_f1": f1[i],
                })

            for (actual, predicted), count in mistakes.most_common():
                confusions.append({
                    "dataset": name,
                    "system": system,
                    "true": actual,
                    "predicted": predicted,
                    "seeds": len(seeds),
                    "errors_across_runs": count,
                    "mean_errors_per_seed": count / len(seeds),
                })

        for left, right in [("B", "C"), ("C", "deberta")]:
            seeds = config.PROMPT_SEEDS if left == "B" else config.SEEDS
            for seed in seeds:
                a = predictions[left, seed] == true
                b = predictions[right, seed] == true
                overlaps.append({
                    "dataset": name,
                    "left": left,
                    "right": right,
                    "seed": seed,
                    "test_items": len(test),
                    "both_right": int((a & b).sum()),
                    "only_left_right": int((a & ~b).sum()),
                    "only_right_right": int((~a & b).sum()),
                    "both_wrong": int((~a & ~b).sum()),
                })

        review = test.rename(columns={"label": "true"}).copy()
        review.insert(0, "idx", test.index)
        review.insert(0, "dataset", name)
        for system in SYSTEMS:
            review[system] = predictions[system, 0]
        errors = review[(review["B"] != review["true"]) | (review["C"] != review["true"])]
        examples.append(errors.sample(n=min(25, len(errors)), random_state=0).sort_values("idx"))

    OUTPUT.mkdir(exist_ok=True)
    per_class = pd.DataFrame(classes).sort_values(["dataset", "system", "mean_f1", "class"])
    per_class.to_csv(OUTPUT / "per_class.csv", index=False)
    per_class.groupby(["dataset", "system"]).head(10).to_csv(OUTPUT / "worst_classes.csv", index=False)
    pd.DataFrame(confusions).groupby(["dataset", "system"]).head(10).to_csv(
        OUTPUT / "top_confusions.csv", index=False
    )
    pd.DataFrame(overlaps).to_csv(OUTPUT / "error_overlap.csv", index=False)
    pd.DataFrame(majority).to_csv(OUTPUT / "majority_baseline.csv", index=False)

    review = pd.concat(examples, ignore_index=True)
    review.to_csv(OUTPUT / "review_candidates.csv", index=False)
    annotations = OUTPUT / "error_annotations.csv"
    if annotations.exists():
        notes = pd.read_csv(annotations)
        assert set(zip(review.dataset, review.idx)) == set(zip(notes.dataset, notes.idx))
        assert notes[["error_type", "note"]].notna().all().all()
        review = review.merge(notes, on=["dataset", "idx"], validate="one_to_one")
        review.to_csv(OUTPUT / "reviewed_errors.csv", index=False)
        counts = review.groupby(["dataset", "error_type"]).size().reset_index(name="reviewed_messages")
        counts.to_csv(OUTPUT / "review_categories.csv", index=False)

    print("Saved error analysis to", OUTPUT)


if __name__ == "__main__":
    main()
