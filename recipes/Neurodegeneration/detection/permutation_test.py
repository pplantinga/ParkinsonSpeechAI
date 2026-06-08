#!/usr/bin/env python3
"""
Permutation test for dementia detection for a single experiment.

Walks $HOME/scratch/results/${experiment_name}/seed_*/ and collects
per-participant combined predictions from test_metrics.json, averages
scores across seeds per participant, then runs a label-permutation test
(AUC and F1) to check whether dementia detection is better than chance —
both overall and per dataset (qpn, pitt, delaware).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm


BASE_PATH = Path(os.environ["HOME"]) / "scratch" / "dementiaTraining"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(experiment_path):
    """
    Walk experiment_path/seed_*/ and collect per-participant combined
    predictions from test_metrics.json.

    File format (one object per line):
      line 1: JSON dict {utt_id: {"scores": [...], "label": int,
                                   "combined": float, "dataset": str, ...}}
      line 2: "Chunk stats: {...}"
      line 3: "Combined stats: {...}"   <- threshold lives here
      line 4: "Per-dataset stats: {...}"
    """
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {experiment_path}")

    predictions = []

    for seed_dir in sorted(experiment_path.glob("seed_*")):
        fpath = seed_dir / "test_metrics.json"
        if not fpath.exists():
            print(f"[warn] no test_metrics.json found in {seed_dir}, skipping")
            continue

        with open(fpath) as f:
            lines = f.read().splitlines()

        if len(lines) < 2:
            print(f"[warn] skipping {fpath}: fewer than 2 lines")
            continue

        # Find the "Combined stats: {...}" line by prefix, not index, so the
        # parse survives changes to how many stat blocks get written.
        combined_line = next(
            (ln for ln in lines if ln.startswith("Combined stats: ")), None
        )
        if combined_line is None:
            print(f"[warn] no 'Combined stats:' line in {fpath}")
            continue
        try:
            threshold = json.loads(
                combined_line.split("Combined stats: ")[1]
            )["threshold"]
        except (KeyError, json.JSONDecodeError) as e:
            print(f"[warn] could not parse combined stats in {fpath}: {e}")
            continue

        # Per-participant predictions are still the first line
        try:
            per_utt = json.loads(lines[0])
        except json.JSONDecodeError as e:
            print(f"[warn] could not parse predictions in {fpath}: {e}")
            continue

        for utt_id, v in per_utt.items():
            predictions.append({
                "seed":      seed_dir.name,
                "utt_id":   utt_id,
                "pid":      v.get("pid"),
                "dataset":  v.get("dataset"),
                "ptype":    v.get("ptype"),
                "label":    v["label"],
                "score":    v["combined"],
                "threshold": threshold,
            })

    df = pd.DataFrame.from_records(predictions)
    if df.empty:
        raise RuntimeError(f"No test metrics found under {experiment_path}")
    return df


def aggregate_across_seeds(df):
    """
    Average each participant's score across seeds, yielding one row per
    participant. Label and metadata are constant across seeds per participant;
    threshold is averaged (should be identical across seeds but we're safe).

    This ensures the permutation test's independence assumption holds —
    the same participant appearing in multiple seeds is not treated as
    multiple independent observations.
    """
    n_seeds = df["seed"].nunique()
    participants_per_seed = df.groupby("seed")["utt_id"].nunique()

    # Warn if any participant is missing from a seed — could indicate a
    # mismatched checkpoint or a crashed run
    expected = df["utt_id"].nunique()
    missing = participants_per_seed[participants_per_seed != expected]
    if not missing.empty:
        print(f"[warn] some seeds have a different participant count than expected ({expected}):")
        for seed, count in missing.items():
            print(f"       {seed}: {count} participants")

    agg = (
        df.groupby("utt_id")
        .agg(
            pid=("pid", "first"),
            dataset=("dataset", "first"),
            ptype=("ptype", "first"),
            label=("label", "first"),
            score=("score", "mean"),       # average across seeds
            threshold=("threshold", "mean"),
            n_seeds_seen=("seed", "nunique"),
        )
        .reset_index()
    )

    print(f"\nAggregated {len(df)} rows ({n_seeds} seed(s)) "
          f"-> {len(agg)} unique participants")
    if (agg["n_seeds_seen"] < n_seeds).any():
        n_partial = (agg["n_seeds_seen"] < n_seeds).sum()
        print(f"[warn] {n_partial} participant(s) appear in fewer than "
              f"{n_seeds} seeds — their averaged scores are based on partial data")

    return agg


# ---------------------------------------------------------------------------
# Metrics and permutation test
# ---------------------------------------------------------------------------

def compute_metric(y_true, y_pred, metric, threshold=0.5):
    """Compute roc_auc or f1 from true labels and continuous scores."""
    if metric == "roc_auc":
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_pred)
    if metric == "f1":
        y_pred_bin = (np.asarray(y_pred) >= threshold).astype(int)
        return f1_score(y_true, y_pred_bin, zero_division=0)
    raise ValueError(f"Unsupported metric: {metric}")


def permutation_test(df, metric="roc_auc", n_permutations=2000, random_state=42):
    """
    Label-permutation test: H0 = predictions carry no information about labels.
    Returns (observed_score, p_value, perm_scores).
    """
    rng = np.random.default_rng(random_state)

    y_true = df["label"].values
    y_pred = df["score"].values
    threshold = float(df["threshold"].mean())

    observed_score = compute_metric(y_true, y_pred, metric=metric, threshold=threshold)
    perm_scores = np.zeros(n_permutations)

    for i in tqdm(range(n_permutations), desc=f"  {metric}", leave=False):
        y_perm = rng.permutation(y_true)
        perm_scores[i] = compute_metric(y_perm, y_pred, metric=metric, threshold=threshold)

    valid = ~np.isnan(perm_scores)
    p_value = float(np.mean(perm_scores[valid] >= observed_score)) if valid.sum() > 0 else np.nan

    return observed_score, p_value, perm_scores


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_all(df, n_permutations=2000, random_state=42):
    """
    Run permutation tests overall and per dataset (qpn, pitt, delaware).
    df should already be aggregated across seeds (one row per participant).
    """
    results = []

    slices = {"overall": df}
    for dataset, subset in df.groupby("dataset"):
        slices[dataset] = subset

    for slice_name, subset in slices.items():
        n_pos = subset["label"].sum()
        n_neg = len(subset) - n_pos
        print(f"\n[{slice_name.upper()}]  n={len(subset)}  "
              f"Disease={n_pos}  Control={n_neg}")

        obs_auc, p_auc, perm_auc = permutation_test(
            subset, metric="roc_auc",
            n_permutations=n_permutations, random_state=random_state,
        )
        obs_f1, p_f1, perm_f1 = permutation_test(
            subset, metric="f1",
            n_permutations=n_permutations, random_state=random_state,
        )

        results.append({
            "slice":           slice_name,
            "n":               len(subset),
            "n_disease":       int(n_pos),
            "n_control":       int(n_neg),
            "observed_auc":    obs_auc,
            "p_value_auc":     p_auc,
            "observed_f1":     obs_f1,
            "p_value_f1":      p_f1,
            "perm_scores_auc": perm_auc,
            "perm_scores_f1":  perm_f1,
        })

    return pd.DataFrame(results)


def report(results_df, experiment_name, alpha=0.05):
    """Print raw and Bonferroni-corrected significance for each slice."""
    if results_df.empty:
        print("No results to report.")
        return results_df

    results_df = results_df.copy()

    # Bonferroni across slices × {AUC, F1}
    num_tests = len(results_df) * 2
    results_df["p_value_auc_corrected"] = np.minimum(results_df["p_value_auc"] * num_tests, 1.0)
    results_df["p_value_f1_corrected"]  = np.minimum(results_df["p_value_f1"]  * num_tests, 1.0)

    def _fmt(row, corrected=False):
        suffix = "_corrected" if corrected else ""
        p_auc   = row[f"p_value_auc{suffix}"]
        p_f1    = row[f"p_value_f1{suffix}"]
        auc_sig = p_auc < alpha
        f1_sig  = p_f1  < alpha

        if auc_sig and f1_sig:
            tag = "Significant in BOTH    "
        elif auc_sig:
            tag = "Significant in AUC only"
        elif f1_sig:
            tag = "Significant in F1 only "
        else:
            tag = "Not significant        "

        return (f"{tag}   "
                f"AUC={row['observed_auc']:.3f} (p={p_auc:.4f})   "
                f"F1={row['observed_f1']:.3f} (p={p_f1:.4f})")

    for corrected, label in [
        (False, "Raw"),
        (True, f"Bonferroni-corrected ({num_tests} tests)"),
    ]:
        print("\n" + "=" * 80)
        print(f"Experiment: {experiment_name}  |  {label} results (alpha={alpha})")
        print("=" * 80)
        for _, row in results_df.iterrows():
            print(f"[{row['slice'].upper():<10}]  {_fmt(row, corrected=corrected)}")

    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_name",
                        help=f"Experiment folder name under {BASE_PATH}")
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--random-state",   type=int, default=42)
    parser.add_argument("--alpha",          type=float, default=0.05)
    args = parser.parse_args()

    experiment_path = BASE_PATH / args.experiment_name
    print(f"Loading data from: {experiment_path}")
    df = load_data(experiment_path)
    print(f"Loaded {len(df)} rows across "
          f"{df['seed'].nunique()} seed(s), "
          f"datasets: {sorted(df['dataset'].dropna().unique())}")

    df = aggregate_across_seeds(df)

    results_df = run_all(df, n_permutations=args.n_permutations,
                             random_state=args.random_state)
    results_df = report(results_df, experiment_name=args.experiment_name,
                                    alpha=args.alpha)

    output_csv = f"{args.experiment_name}_permutation_results.csv"
    to_save = results_df.drop(columns=["perm_scores_auc", "perm_scores_f1"], errors="ignore")
    to_save.to_csv(output_csv, index=False)
    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    main()
