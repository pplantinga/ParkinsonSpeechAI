#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from prepare_prepare import prepare_prepare


def main(datadir):
    manifests = {d: datadir / (d + ".json") for d in ["train", "valid", "test"]}
    prepare_prepare(datadir, manifests)
    train_set, valid_set, test_set, le = prepare_data(datadir, manifests)

    #xgb_model = fit_xgboost(train_set, valid_set, le)
    xgb_model = random_search(train_set, valid_set)

    predictions = predict_test(xgb_model, test_set)


def prepare_data(datadir, manifests):
    train_manifest = pd.read_json(manifests["train"], orient="index")
    valid_manifest = pd.read_json(manifests["valid"], orient="index")
    test_manifest = pd.read_json(manifests["test"], orient="index")

    # Load csv data
    train_features = pd.read_csv(datadir / "train_features.csv")
    test_features = pd.read_csv(datadir / "test_features.csv")

    # Average features across time and add age
    feature_columns = train_features.columns[3:]
    X_train_valid = train_features.groupby("uid")[feature_columns].mean()
    X_test = test_features.groupby("uid")[feature_columns].mean()
    train_valid_manifest = pd.concat((train_manifest["age"], valid_manifest["age"]))
    X_train_valid = X_train_valid.join(train_valid_manifest)
    X_test = X_test.join(test_manifest["age"])

    # Split the data
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train_valid.loc[train_manifest.index]
    X_valid = X_train_valid.loc[valid_manifest.index]

    y_train = train_manifest[["diagnosis_control", "diagnosis_mci", "diagnosis_adrd"]]
    y_valid = valid_manifest[["diagnosis_control", "diagnosis_mci", "diagnosis_adrd"]]

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = X_train #scaler.fit_transform(X_train)
    X_valid_scaled = X_valid #scaler.transform(X_valid)
    X_test_scaled = X_test #scaler.transform(X_test)

    # Convert labels to numbers
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train.idxmax(axis=1))
    y_valid_encoded = le.transform(y_valid.idxmax(axis=1))

    return (
        (X_train_scaled, y_train_encoded),
        (X_valid_scaled, y_valid_encoded),
        X_test_scaled,
        le,
    )


def fit_xgboost(train_set, valid_set, le):
    """Fit an xgboost model"""
    X_train, y_train = train_set
    X_valid, y_valid = valid_set

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )

    # Create sample weights array
    sample_weights = np.ones(len(y_train))
    for idx, label in enumerate(y_train):
        sample_weights[idx] = class_weights[label]

    # Train XGBoost classifier
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=42,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = xgb.predict(X_valid)

    # Calculate accuracy
    accuracy = accuracy_score(y_valid, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_valid, y_pred, target_names=le.classes_))

    # Get feature importance
    #feature_importance = pd.DataFrame(
    #    {"feature": feature_columns, "importance": xgb.feature_importances_}
    #)
    #feature_importance = feature_importance.sort_values("importance", ascending=False)

    #print("\nTop 10 Most Important Features:")
    #print(feature_importance.head(10))

    return xgb


def random_search(train_set, valid_set):

    X_train, y_train = train_set
    X_valid, y_valid = valid_set

    # Define parameter space for RandomizedSearchCV
    random_params = {
        "n_estimators": randint(20, 100),
        "max_depth": randint(20, 50),
        "learning_rate": loguniform(0.01, 0.2),
        "subsample": uniform(0.6, 0.3),
        "colsample_bytree": uniform(0.6, 0.3),
        #"min_child_weight": randint(1, 3),
        "gamma": uniform(0, 1),
        "reg_alpha": uniform(0, 0.5),
        "reg_lambda": uniform(0, 0.5),
    }

    # Initialize base model
    xgb = XGBClassifier(objective="multi:softprob", random_state=42, num_class=3)

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=random_params,
        n_iter=100,
        scoring="neg_log_loss",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    # Fit and get best parameters
    random_search.fit(X_train, y_train)

    # Get best parameters and score
    print("Best parameters:", random_search.best_params_)
    print("Best score:", random_search.best_score_)

    # Train final model with best parameters
    best_model = XGBClassifier(**random_search.best_params_)
    best_model.fit(X_train, y_train)

    # Evaluate on val set
    accuracy = best_model.score(X_valid, y_valid)
    print("Validation accuracy: %.2f%%" % (accuracy * 100.0))

    valid_preds = best_model.predict_proba(X_valid)
    log_score = log_loss(y_valid, valid_preds)
    print("Validation score: %.2f" % (log_score))

    # Print predictions for tuning average
    write_to_file(X_valid, valid_preds, "valid_predictions.csv")

    return best_model

def write_to_file(X_df, predictions, filename="predictions.csv"):
    X_df = X_df.assign(
        diagnosis_control = [p[0] for p in predictions],
        diagnosis_mci = [p[1] for p in predictions],
        diagnosis_adrd = [p[2] for p in predictions],
    )

    X_df.to_csv(
        filename,
        index_label="uid",
        columns=["diagnosis_control", "diagnosis_mci", "diagnosis_adrd"],
    )


def predict_test(model, X_test):
    # Print predictions for submitting
    y_pred_probs = model.predict_proba(X_test)
    write_to_file(X_test, y_pred_probs, "test_predictions.csv")

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Expected 1 argument: data folder"
    
    main(Path(sys.argv[1]))
