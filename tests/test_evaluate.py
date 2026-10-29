import numpy as np
import pytest

import config
import data
import evaluate

LABELS = ["a", "b", "c"]


# ---- scoring tests ----

def test_parse_valid():
    assert evaluate.parse_output('{"category": "a"}', LABELS) == "a"


def test_parse_invalid():
    bad_outputs = [
        '{"category": "z"}',             # label does not exist
        '{"category": "a", "x": 1}',     # extra key
        '{"label": "a"}',                # wrong key
        'not json',
        '{"category": "a"} extra text',
        '[1, 2]',
        None,
    ]
    for output in bad_outputs:
        assert evaluate.parse_output(output, LABELS) is None


def test_accuracy_invalid_counts_as_wrong():
    true = ["a", "b", "c", "a"]
    predicted = ["a", "b", None, "b"]
    assert evaluate.accuracy(true, predicted) == 0.5


def test_json_validity():
    outputs = ['{"category": "a"}', 'junk', '{"category": "b"}', '{"category": "q"}']
    assert evaluate.json_validity(outputs, LABELS) == 0.5


def test_majority_baseline():
    assert evaluate.majority_baseline(["a", "a", "b"], ["a", "b", "a", "a"]) == 0.75


def test_macro_f1_perfect_and_wrong():
    true = ["a", "b", "c"]
    assert evaluate.macro_f1(true, ["a", "b", "c"], LABELS) == 1.0
    assert evaluate.macro_f1(true, [None, None, None], LABELS) == 0.0


def test_bootstrap_range_contains_mean():
    correct = np.array([1] * 80 + [0] * 20)
    mean, low, high = evaluate.bootstrap_ci(correct)
    assert mean == 0.8
    assert low < mean < high


def test_paired_bootstrap_same_system_is_zero():
    correct = np.array([1] * 80 + [0] * 20)
    diff, low, high = evaluate.paired_bootstrap(correct, correct)
    assert diff == 0 and low == 0 and high == 0


def test_paired_bootstrap_clear_winner():
    a = np.array([1] * 90 + [0] * 10)
    b = np.array([1] * 50 + [0] * 50)
    diff, low, high = evaluate.paired_bootstrap(a, b)
    assert low > 0


# ---- leakage tests (use the real saved splits, skipped if not built yet) ----

@pytest.mark.parametrize("name", list(config.DATASETS))
def test_no_leakage(name):
    if not (config.SPLITS_DIR / name / "checksums.json").exists():
        pytest.skip("splits not built yet, run python data.py")

    test_texts = set(data.load_test(name)["text"].map(data.clean))
    dev_texts = set(data.load_dev(name)["text"].map(data.clean))
    assert not test_texts & dev_texts

    biggest = max(config.DATASETS[name]["budgets"])
    for seed in config.SEEDS:
        train_texts = set(data.load_train(name, seed, biggest)["text"].map(data.clean))
        assert not train_texts & test_texts, f"test item in training set (seed {seed})"
        assert not train_texts & dev_texts, f"dev item in training set (seed {seed})"


@pytest.mark.parametrize("name", list(config.DATASETS))
def test_budgets_are_nested(name):
    if not (config.SPLITS_DIR / name / "checksums.json").exists():
        pytest.skip("splits not built yet, run python data.py")

    budgets = config.DATASETS[name]["budgets"]
    for seed in config.SEEDS:
        for small, big in zip(budgets, budgets[1:]):
            small_set = set(data.load_train(name, seed, small)["text"])
            big_set = set(data.load_train(name, seed, big)["text"])
            assert small_set <= big_set