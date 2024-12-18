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
    for weight in numpy.arange(0.01, 1, 0.01):
        pred = combine_preds(valid_scores1, valid_scores2, weight)
        score = log_loss(valid_labels, pred)

        if score < min_score:
            min_score = score
            min_weight = weight

    print("Best score:", min_score)
    print("Best weight:", min_weight)

    return min_weight

def combine_preds(scores1, scores2, weight=0.5):
    pred = scores1 * weight + scores2 * (1 - weight)
    #pred /= pred.sum(dim=-1, keepdim=True)
        
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
    assert len(sys.argv) == 3, "Expected dir1, dir2 as arguments"

    path1 = Path(sys.argv[1])
    path2 = Path(sys.argv[2])

    valid_scores1, test_scores1 = load_scores(path1)
    valid_scores2, test_scores2 = load_scores(path2)

    valid_labels = load_valid_labels("valid.json")

    weight = determine_weight(valid_labels, valid_scores1, valid_scores2)

    preds = combine_preds(test_scores1, test_scores2, weight)
    print(preds)

    preds.to_csv("test_predictions.csv")
    
