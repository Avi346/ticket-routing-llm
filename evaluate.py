import json

import numpy as np
from sklearn.metrics import f1_score


def parse_output(text, labels):
    # returns the label if the output is valid json like {"category": "<label>"}, else None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or list(data.keys()) != ["category"]:
        return None
    if data["category"] not in labels:
        return None
    return data["category"]


def json_validity(outputs, labels):
    valid = [parse_output(o, labels) is not None for o in outputs]
    return sum(valid) / len(valid)


def correct_array(true_labels, predicted):
    # 1 if right, 0 if wrong. None (invalid output) is always wrong
    return np.array([1 if t == p else 0 for t, p in zip(true_labels, predicted)])


def accuracy(true_labels, predicted):
    return correct_array(true_labels, predicted).mean()


def macro_f1(true_labels, predicted, labels):
    # invalid outputs (None) are turned into a fake label so sklearn accepts them
    predicted = [p if p is not None else "invalid" for p in predicted]
    return f1_score(true_labels, predicted, labels=labels, average="macro", zero_division=0)


def majority_baseline(train_labels, test_labels):
    # accuracy of always guessing the most common class in train
    biggest = max(set(train_labels), key=list(train_labels).count)
    return float(np.mean([t == biggest for t in test_labels]))


def bootstrap_ci(correct, n_boot=2000, seed=0):
    # correct: 1D array (one run) or 2D array (seeds x test items)
    # resample test items with replacement, take the mean each time, return the 95% range
    correct = np.atleast_2d(correct)
    rng = np.random.default_rng(seed)
    n_items = correct.shape[1]
    means = []
    for _ in range(n_boot):
        picked = rng.integers(0, n_items, n_items)
        means.append(correct[:, picked].mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return correct.mean(), low, high


def paired_bootstrap(correct_a, correct_b, n_boot=2000, seed=0):
    # difference in accuracy (a - b) on the same items, with a 95% range
    # the same resampled items are used for both, which is what makes it paired
    correct_a = np.atleast_2d(correct_a)
    correct_b = np.atleast_2d(correct_b)
    rng = np.random.default_rng(seed)
    n_items = correct_a.shape[1]
    diffs = []
    for _ in range(n_boot):
        picked = rng.integers(0, n_items, n_items)
        diffs.append(correct_a[:, picked].mean() - correct_b[:, picked].mean())
    low, high = np.percentile(diffs, [2.5, 97.5])
    return correct_a.mean() - correct_b.mean(), low, high