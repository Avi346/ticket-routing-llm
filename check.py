import baselines
for n in ["banking77", "emotion"]:
    for key in ["test", "train_seed0", "train_seed1"]:
        baselines.embed(n, key)