import json, pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

seeds = [3011, 3012, 3013, 3014, 3015]


def permutation_test_f1(df, n_permutations=1000, seed=42):
    """
    Test whether male and female samples differ in F1
    using a permutation test.

    df must contain: pid, label, score, sex, seed
    """
    rng = np.random.default_rng(seed)

    # Get unique subjects
    subjects = df[["pid", "label", "sex"]].drop_duplicates()

    def sample_predictions():
        """Randomly pick one seed's prediction for each subject."""
        preds = []
        for pid in subjects.pid.values:
            # restrict to this subject's rows (different seeds)
            rows = df[df.pid == pid]
            chosen_row = rows.sample(n=1, random_state=rng.integers(1e9))
            preds.append((pid, chosen_row.label.values[0],
                          chosen_row.sex.values[0],
                          int(chosen_row.score.values[0] >= 0.5)))
        return pd.DataFrame(preds, columns=["pid", "label", "sex", "pred"])

    def f1_by_sex(pred_df):
        """Compute F1 separately for males and females."""
        results = {}
        for sex in ["M", "F"]:
            subset = pred_df[pred_df.sex == sex]
            if subset.label.nunique() < 2:
                # If only one class present, F1 is undefined -> set to nan
                results[sex] = np.nan
            else:
                results[sex] = f1_score(subset.label, subset.pred)
        return results

    # Observed statistic
    obs_preds = sample_predictions()
    obs_f1 = f1_by_sex(obs_preds)
    observed_diff = obs_f1["M"] - obs_f1["F"]

    # Permutation null distribution
    null_diffs = []
    for _ in range(n_permutations):
        preds = sample_predictions()

        # Shuffle sex labels across subjects
        shuffled_sex = rng.permutation(preds["sex"].values)
        preds_perm = preds.copy()
        preds_perm["sex"] = shuffled_sex

        f1s = f1_by_sex(preds_perm)
        diff = f1s["M"] - f1s["F"]
        null_diffs.append(diff)

    null_diffs = np.array(null_diffs)

    # Two-sided p-value
    p_value = np.mean(np.abs(null_diffs) >= abs(observed_diff))

    return observed_diff, p_value, null_diffs

if __name__ == "__main__":
    predictions = []
    for seed in seeds:
        filepath = pathlib.Path(f"results/wavlm_base_plus_ecapa_tdnn/seed_{seed}/predictions.json")
        with open(filepath) as f:
            predictions.extend([
                {"pid": k, "seed": seed, "label": v["label"], "score": v["score"], "sex": v["sex"]}
                for k, v in json.load(f).items()
            ])

    df = pd.DataFrame.from_records(predictions)
    observed_diff, p_value, null_distribution = permutation_test_f1(df)
    print("Observed accuracy difference (M - F):", observed_diff)
    print("Permutation test p-value:", p_value)
