import argparse
import json
import numpy as np
from sklearn.metrics import f1_score

def parse_test_file(test_file):
    with open(test_file) as f:
        f.readline() # Clear "Overall:"
        overall = json.loads(f.readline())
        f.readline() # Clear "Male:"
        male = json.loads(f.readline())
        f.readline() # Clear "Female:"
        female = json.loads(f.readline())
    return overall, male, female

def extract_predictions(scores):
    y_true, y_pred = [], []
    y_true += [1] * int(scores["TP"])
    y_pred += [1] * int(scores["TP"])
    y_true += [0] * int(scores["TN"])
    y_pred += [0] * int(scores["TN"])
    y_true += [0] * int(scores["FP"])
    y_pred += [1] * int(scores["FP"])
    y_true += [1] * int(scores["FN"])
    y_pred += [0] * int(scores["FN"])
    return np.array(y_true), np.array(y_pred)

def bootstrap_ci(y_true, y_pred, trials=2000, rng_seed=897234, average='binary'):
    rng = np.random.RandomState(rng_seed)
    N = len(y_true)
    boot_scores = np.empty(trials)
    for trial in range(trials):
        idx = rng.randint(0, N, size=N)   # sample with replacement
        boot_scores[trial] = f1_score(y_true[idx], y_pred[idx], average=average)
    lower, upper = np.percentile(boot_scores, [2.5, 97.5])
    return lower, upper, boot_scores


def permutation_test(y_true, y_pred, trials=2000, rng_seed=897234, average='binary'):
    t_obs = f1_score(y_true, y_pred, average=average)
    rng = np.random.RandomState(rng_seed)
    t_null = []
    for trial in range(trials):
        y_perm = rng.permutation(y_true)
        t_null.append(f1_score(y_perm, y_pred, average=average))

    p = (1 + np.sum(np.array(t_null) >= t_obs)) / (1 + trials)

    return t_obs, p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file")
    args = parser.parse_args()

    overall, male, female = parse_test_file(args.test_file)
    y_true_male, y_pred_male = extract_predictions(male)
    y_true_female, y_pred_female = extract_predictions(female)
    f1_low, f1_high, scores = bootstrap_ci(y_true_male, y_pred_male)

    print(f"95% Confidence interval: {f1_low:.1%}-{f1_high:.1%}")

    t_obs, p = permutation_test(y_true_male, y_pred_male)

    print(f"Observed f1: {t_obs:.1%}, permutation test p-value: {p:.2g}")
