import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
import sys
from tqdm import tqdm

def extract_experiment_data(base_path):
  predictions = []
  for root, _, results in os.walk(base_path):
    if not "train" in root:
      continue
    for result in results:
      if not (result.startswith("pd") or result.startswith("ad")):
        continue
      with open(os.path.join(root, result)) as f:
        file_content = f.read()
        first_line = file_content.splitlines()[0]

        last_line = file_content.splitlines()[-1]
        json_part = last_line.split("Combined stats: ")[1]
        data = json.loads(json_part)
        threshold = data["overall"]["threshold"]

        predictions.extend([
            {"disease": result.split("_")[0], "experiment": root.split("/")[-2], "seed": root.split("/")[-1], "pid": k, "label": v["label"], "score": v["combined"], "threshold": threshold}
            for k, v in json.loads(first_line).items()
        ])

  df = pd.DataFrame.from_records(predictions)
  return df

def compute_metric(y_true, y_pred, metric="accuracy", threshold=0.5):
    """
    Compute a performance metric given true labels and continuous predictions.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred : array-like
        Continuous model predictions (e.g., probabilities)
    metric : str
        One of {"accuracy", "roc_auc", "f1"}
    threshold : float
        Threshold to binarize predictions for accuracy/F1

    Returns
    -------
    float
        Metric value
    """
    if metric == "roc_auc":
        return roc_auc_score(y_true, y_pred)

    y_pred_bin = (y_pred >= threshold).astype(int)

    if metric == "accuracy":
        return accuracy_score(y_true, y_pred_bin)
    elif metric == "f1":
        return f1_score(y_true, y_pred_bin)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def permutation_test(df, metric="accuracy", n_permutations=2000, random_state=None):
    """
    Perform a permutation test to evaluate whether a model performs
    significantly better than chance.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing true labels and continuous predictions
    metric : str
        Metric to evaluate {"accuracy", "roc_auc", "f1"}
    n_permutations : int
        Number of permutations
    random_state : int or None
        Optional random seed for reproducibility

    Returns
    -------
    observed_score : float
        The actual metric on true labels
    p_value : float
        Permutation test p-value
    perm_scores : np.ndarray
        Distribution of scores from permuted labels
    """
    rng = np.random.default_rng(random_state)

    y_true = df.label.values
    y_pred = df.score.values

    observed_score = compute_metric(y_true, y_pred, metric=metric, threshold=df.threshold.values[0])
    perm_scores = np.zeros(n_permutations)

    for i in tqdm(range(n_permutations), desc=f"Running permutation test ({metric})"):
        y_perm = rng.permutation(y_true) # shuffle labels
        perm_scores[i] = compute_metric(y_perm, y_pred, metric=metric, threshold=df.threshold.values[0]) # compute metric with predictions and shuffled labels
        # do this n_permutations times

    p_value = np.mean(perm_scores >= observed_score)
    return observed_score, p_value, perm_scores


if __name__ == '__main__':    
    path = sys.argv[1]
    df = extract_experiment_data(path)

    experiments = df.experiment.unique()
    diseases = df.disease.unique()

    results = []

    # loop through experiments and diseases
    # trying to see if for experiment X, AD performed better than chance, then if PD performed better than chance
    for exp in experiments:
        for disease in diseases:
            subset_df = df[(df.experiment == exp) & (df.disease == disease)]
            if not subset_df.empty:
                print(f"Running permutation test for Experiment: {exp}, Disease: {disease}")
                observed_auc, p_value_auc, perm_scores_auc = permutation_test(subset_df, metric="roc_auc")
                observed_f1, p_value_f1, perm_scores_f1 = permutation_test(subset_df, metric="f1")
                results.append({
                    "experiment": exp,
                    "disease": disease,
                    "observed_auc": observed_auc,
                    "p_value_auc": p_value_auc,
                    "observed_f1": observed_f1,
                    "p_value_f1": p_value_f1,
                    "perm_scores_auc": perm_scores_auc,
                    "perm_scores_f1": perm_scores_f1
                })

    results_df = pd.DataFrame(results)
    
    # Test significance with no correction
    alpha = 0.05
    results_df['significant_auc'] = results_df['p_value_auc'] < alpha
    results_df['significant_f1'] = results_df['p_value_f1'] < alpha
    print("Permutation test results (no correction):")
    print(results_df)

    # Apply bonferroni correction
    n_tests = len(results_df)
    corrected_alpha = alpha / 2
    results_df['corrected_significant_auc'] = results_df['p_value_auc'] < corrected_alpha
    results_df['corrected_significant_f1'] = results_df['p_value_f1'] < corrected_alpha
    print("After Bonferroni correction:")
    print(results_df)

    # Save results
    save_folder = "./permutation_test_results.csv" if len(sys.argv) < 3 else os.path.join(sys.argv[2], "permutation_test_results.csv")
    results_df.to_csv(save_folder, index=False)
