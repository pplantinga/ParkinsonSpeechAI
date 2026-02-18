import json, pathlib
import pandas as pd
import numpy as np

seeds = [3011, 3012, 3013, 3014, 3015]

if __name__ == "__main__":
    predictions = []
    for seed in seeds:
        filepath = pathlib.Path(f"results/wavlm_base_plus_ecapa_tdnn/seed_{seed}/predictions.json")

        with open(filepath) as f:
            predictions.extend([{"pid": k, "seed": seed, "label": v["label"], "score": v["score"]} for k, v in json.load(f).items()])

    df = pd.DataFrame.from_records(predictions)
    print(df)

    # Generate a list of unique pids, to make one prediction per pid
    pids = df.pid.unique().sort_values(by="pid")
    labels = df[df.seed == 3011].sort_values(by="pid").label
    print(pids)

    # For 1000 trials, randomly permute the labels, as well as choose one random answer per seed
    rng = np.random.default_rng(seed=2847284)
    for k in range(1000):
        seeds = rng.choice(seeds, size=len(pids))
        for pid, seed in zip(pids, seeds):
            row = df.loc[df.seed == seed & df.pid == pid]


