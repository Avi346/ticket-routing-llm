import config
import data
import retrieval


def test_no_test_item_in_any_pool():
    for name in config.DATASETS:
        test = set(data.load_test(name)["text"].map(data.clean))
        n_classes = len(data.load_labels(name))
        for seed in config.SEEDS:
            for budget in config.DATASETS[name]["budgets"]:
                p, _ = retrieval.pool(name, seed, budget)
                assert len(p) == budget * n_classes
                assert not test & set(p["text"].map(data.clean))


def test_neighbours_shape_and_no_leak():
    for name in config.DATASETS:
        test = set(data.load_test(name)["text"].map(data.clean))
        shots = retrieval.neighbours(name, 0, config.DATASETS[name]["budgets"][0], n=50)
        assert len(shots) == 50
        assert all(len(s) == config.RETRIEVAL_K for s in shots)
        assert not test & {data.clean(t) for s in shots for t, _ in s}