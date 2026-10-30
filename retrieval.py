import numpy as np

import baselines
import config
import data


def pool(name, seed, budget):
    df = data.read_split(name, f"train_seed{seed}.csv")
    keep = (df["rank_in_class"] < budget).values
    return df[keep].reset_index(drop=True), keep


def neighbours(name, seed, budget, k=config.RETRIEVAL_K, n=config.HEADLINE_TEST_SIZE):
    p, keep = pool(name, seed, budget)
    pool_x = baselines.embed(name, f"train_seed{seed}")[keep]
    test_x = baselines.embed(name, "test")[:n]
    sim = test_x @ pool_x.T          # embeddings are normalized, so this is cosine similarity
    top = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    # most similar example goes last, right before the question
    return [[(p["text"][j], p["label"][j]) for j in row[::-1]] for row in top]