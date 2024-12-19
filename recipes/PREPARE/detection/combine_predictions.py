import sys
import csv
import numpy
import pandas
from pathlib import Path
from sklearn.metrics import log_loss

def determine_weight(valid_labels, valid_scores1, valid_scores2):

    # Try different weights and compute log loss
    min_score = 1.0
    min_weight = 0.5
    min_alpha1 = 1.0
    min_alpha2 = 1.0
    for alpha in numpy.arange(1, 3, 0.1):
        for weight in numpy.arange(0.01, 1, 0.01):
            pred = combine_preds(valid_scores1, valid_scores2, alpha, weight)
            score = log_loss(valid_labels, pred)

            if score < min_score:
                min_score = score
                min_weight = weight
                min_alpha = alpha

    print("Best score:", round(min_score, 4))
    print("Best weight:", round(min_weight, 2))
    print("Best alpha:", round(min_alpha, 2))

    return min_weight, min_alpha

def combine_preds(scores1, scores2, alpha=1, weight=0.5):
    pred = scores1 ** alpha * weight + scores2 ** alpha * (1 - weight)
    pred = pred.div(pred.sum(axis=1), axis=0)
        
    return pred


def load_scores(path):
    valid_df = pandas.read_csv(path / "valid_predictions.csv", index_col="uid")
    test_df = pandas.read_csv(path / "test_predictions.csv", index_col="uid")

    return valid_df, test_df


def load_valid_labels(path):
    valid_labels = pandas.read_json(path, orient="index")

    valid_labels = valid_labels[["diagnosis_control", "diagnosis_mci", "diagnosis_adrd"]]

    valid_labels.index.name = "uid"

    return valid_labels


if __name__ == "__main__":
    assert len(sys.argv) in [3, 4], "Expected dir1, dir2[, dir3] as arguments"

    path1 = Path(sys.argv[1])
    path2 = Path(sys.argv[2])

    valid_scores1, test_scores1 = load_scores(path1)
    valid_scores2, test_scores2 = load_scores(path2)

    valid_labels = load_valid_labels("valid.json")

    weight, alpha = determine_weight(valid_labels, valid_scores1, valid_scores2)

    valid_preds = combine_preds(valid_scores1, valid_scores2, alpha, weight)
    test_preds = combine_preds(test_scores1, test_scores2, alpha, weight)

    if len(sys.argv) == 4:
        path3 = Path(sys.argv[3])
        valid_scores3, test_scores3 = load_scores(path3)
        weight, alpha = determine_weight(valid_labels, valid_preds, valid_scores3)
        test_preds = combine_preds(test_preds, test_scores3, alpha, weight)

    test_preds.to_csv("test_predictions.csv")
