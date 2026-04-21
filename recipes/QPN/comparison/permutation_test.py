#!/usr/bin/env python3
"""
Permutation test for PD and AD detection for a single experiment.

Walks $HOME/scratch/comparison/${experiment_name}/seed_*/ and collects
per-participant combined predictions from pd_test_metrics.json and
ad_test_metrics.json, then runs a label-permutation test (AUC and F1)
to check whether PD / AD detection is better than chance.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm


BASE_PATH = Path(os.environ["HOME"]) / "scratch" / "comparison"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(experiment_path):
    """
    Walk experiment_path/seed_*/ and collect per-participant combined
    predictions from pd_test_metrics.json / ad_test_metrics.json.

    Each file has:
      line 1  : JSON dict {pid: {"scores": [...], "label": int, "combined": float}, ...}
      last ln : "Combined stats: {...}" containing the chosen threshold.
    """
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {experiment_path}")

    predictions = []

    for root, _, files in os.walk(experiment_path):
        root_path = Path(root)
        for fname in files:
            if not (fname.startswith("pd") or fname.startswith("ad")):
                continue
            if not fname.endswith("_test_metrics.json"):
                continue

            fpath = root_path / fname
            with open(fpath) as f:
                lines = f.read().splitlines()

            if len(lines) < 2:
                print(f"[warn] skipping {fpath}: fewer than 2 lines")
                continue

            first_line = lines[0]
            last_line = lines[-1]

            # Threshold from "Combined stats: {...}"
            try:
                combined_json = last_line.split("Combined stats: ")[1]
                threshold = json.loads(combined_json)["overall"]["threshold"]
            except (IndexError, KeyError, json.JSONDecodeError) as e:
                print(f"[warn] could not parse combined stats in {fpath}: {e}")
                continue

            # Per-participant predictions
            try:
                per_pid = json.loads(first_line)
            except json.JSONDecodeError as e:
                print(f"[warn] could not parse predictions in {fpath}: {e}")
                continue

            seed_dir = root_path.name        # e.g. "seed_2028"
            disease = fname.split("_")[0]    # "pd" or "ad"

            for pid, v in per_pid.items():
                predictions.append({
                    "disease": disease,
                    "seed": seed_dir,
                    "pid": pid,
                    "label": v["label"],
                    "score": v["combined"],
                    "threshold": threshold,
                })

    df = pd.DataFrame.from_records(predictions)
    if df.empty:
        raise RuntimeError(f"No pd/ad test metrics found under {experiment_path}")
    return df


# ---------------------------------------------------------------------------
# Metrics and permutation test
# ---------------------------------------------------------------------------

def compute_metric(y_true, y_pred, metric="accuracy", threshold=0.5):
    """Compute accuracy / roc_auc / f1 from true labels and continuous scores."""
    if metric == "roc_auc":
        if len(np.unique(y_true)) < 2:  # degenerate after permutation
            return np.nan
        return roc_auc_score(y_true, y_pred)

    y_pred_bin = (np.asarray(y_pred) >= threshold).astype(int)

    if metric == "accuracy":
        return accuracy_score(y_true, y_pred_bin)
    if metric == "f1":
        return f1_score(y_true, y_pred_bin, zero_division=0)
    raise ValueError(f"Unsupported metric: {metric}")


def permutation_test(df, metric="accuracy", n_permutations=2000, random_state=42):
    """
    Label-permutation test: H0 = predictions carry no information about labels.
    Returns (observed_score, p_value, perm_scores).
    """
    rng = np.random.default_rng(random_state)

    y_true = df["label"].values
    y_pred = df["score"].values
    threshold = float(df["threshold"].values[0])

    observed_score = compute_metric(y_true, y_pred, metric=metric, threshold=threshold)
    perm_scores = np.zeros(n_permutations)

    for i in tqdm(range(n_permutations), desc=f"Permutation test ({metric})", leave=False):
        y_perm = rng.permutation(y_true)
        perm_scores[i] = compute_metric(y_perm, y_pred, metric=metric, threshold=threshold)

    valid = ~np.isnan(perm_scores)
    if valid.sum() == 0:
        p_value = np.nan
    else:
        p_value = float(np.mean(perm_scores[valid] >= observed_score))

    return observed_score, p_value, perm_scores


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_all(df, n_permutations=2000, random_state=42):
    """Run permutation tests per disease (PD, AD), pooling across seeds."""
    results = []
    for disease, subset in df.groupby("disease"):
        print(f"\nPermutation test for disease={disease}  (n={len(subset)}, "
              f"seeds={subset['seed'].nunique()})")

        obs_auc, p_auc, perm_auc = permutation_test(
            subset, metric="roc_auc",
            n_permutations=n_permutations, random_state=random_state,
        )
        obs_f1, p_f1, perm_f1 = permutation_test(
            subset, metric="f1",
            n_permutations=n_permutations, random_state=random_state,
        )

        results.append({
            "disease": disease,
            "n": len(subset),
            "n_seeds": subset["seed"].nunique(),
            "observed_auc": obs_auc,
            "p_value_auc": p_auc,
            "observed_f1": obs_f1,
            "p_value_f1": p_f1,
            "perm_scores_auc": perm_auc,
            "perm_scores_f1": perm_f1,
        })

    return pd.DataFrame(results)


def report(results_df, experiment_name, alpha=0.05):
    """Print raw and Bonferroni-corrected significance for each disease."""
    if results_df.empty:
        print("No results to report.")
        return results_df

    results_df = results_df.copy()
    results_df["significant_auc"] = results_df["p_value_auc"] < alpha
    results_df["significant_f1"] = results_df["p_value_f1"] < alpha

    # Bonferroni across disease × {AUC, F1}
    num_tests = len(results_df) * 2
    results_df["p_value_auc_corrected"] = np.minimum(results_df["p_value_auc"] * num_tests, 1.0)
    results_df["p_value_f1_corrected"] = np.minimum(results_df["p_value_f1"] * num_tests, 1.0)
    results_df["significant_auc_corrected"] = results_df["p_value_auc_corrected"] < alpha
    results_df["significant_f1_corrected"] = results_df["p_value_f1_corrected"] < alpha

    def _fmt(row, corrected=False):
        auc_sig = row["significant_auc_corrected" if corrected else "significant_auc"]
        f1_sig = row["significant_f1_corrected" if corrected else "significant_f1"]
        p_auc = row["p_value_auc_corrected" if corrected else "p_value_auc"]
        p_f1 = row["p_value_f1_corrected" if corrected else "p_value_f1"]
        auc_obs = row["observed_auc"]
        f1_obs = row["observed_f1"]

        if auc_sig and f1_sig:
            s = f"Significant in BOTH   AUC={auc_obs:.3f} (p={p_auc:.4f})  F1={f1_obs:.3f} (p={p_f1:.4f})"
        elif auc_sig:
            s = f"Significant in AUC    AUC={auc_obs:.3f} (p={p_auc:.4f})  F1={f1_obs:.3f} (p={p_f1:.4f})"
        elif f1_sig:
            s = f"Significant in F1     AUC={auc_obs:.3f} (p={p_auc:.4f})  F1={f1_obs:.3f} (p={p_f1:.4f})"
        else:
            s = f"Not significant       AUC={auc_obs:.3f} (p={p_auc:.4f})  F1={f1_obs:.3f} (p={p_f1:.4f})"
        return s

    print("\n" + "=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Raw results (alpha = {alpha})")
    print("=" * 80)
    for _, row in results_df.iterrows():
        print(f"[{row['disease'].upper()}]\n  {_fmt(row, corrected=False)}")

    print("\n" + "=" * 80)
    print(f"Bonferroni-corrected results (alpha = {alpha}, {num_tests} tests)")
    print("=" * 80)
    for _, row in results_df.iterrows():
        print(f"[{row['disease'].upper()}]\n  {_fmt(row, corrected=True)}")

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_name",
                        help=f"Experiment folder name under {BASE_PATH}")
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    experiment_path = BASE_PATH / args.experiment_name
    print(f"Loading data from: {experiment_path}")
    df = load_data(experiment_path)
    print(f"Loaded {len(df)} prediction rows across "
          f"{df['seed'].nunique()} seeds, "
          f"diseases: {sorted(df['disease'].unique())}")

    results_df = run_all(
        df,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )
    results_df = report(results_df, experiment_name=args.experiment_name, alpha=args.alpha)

    output_csv = f"{args.experiment_name}_output.csv"
    to_save = results_df.drop(columns=["perm_scores_auc", "perm_scores_f1"], errors="ignore")
    to_save.to_csv(output_csv, index=False)
    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    main()
