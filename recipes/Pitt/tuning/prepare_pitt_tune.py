import json
import os
import glob
import torchaudio
import numpy as np
import pathlib
import pandas


def convert_to_python(obj):
    """Recursively convert NumPy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python(item) for item in obj]
    else:
        return obj


def prepare_pitt(data_folder, train_annotation, valid_annotation, chunk_size, split):

    assert os.path.exists(data_folder), "Data folder not found"

    data_folder = pathlib.Path(data_folder)
    data_csv = read_csv(data_folder, "pitt_corpus")

    train_gt, valid_gt = stratified_patient_split(data_csv, split)

    create_json(train_annotation, train_gt, chunk_size, overlap=0)
    create_json(valid_annotation, valid_gt, chunk_size)


def stratified_patient_split(data_csv, split):
    """
    Splits recordings into two halves by patient ID, stratified by diagnosis,
    so each half has a similar Control/Disease balance.

    Sorting by patient ID before splitting ensures the assignment is fully
    deterministic — the same patients always land in the same split.

    The split argument (0 or 1) swaps which half becomes train vs. valid.

    :param data_csv: pandas DataFrame with at least 'id' and 'dx' columns
    :param split: int (0 or 1)
    :return: (train_df, valid_df) — two DataFrames
    """
    # Get one row per unique patient to determine their diagnosis
    unique_patients = data_csv.drop_duplicates(subset="id")[["id", "dx"]].copy()

    split_a_ids = []
    split_b_ids = []

    for dx, group in unique_patients.groupby("dx"):
        sorted_ids = sorted(group["id"].tolist())   # Sort for determinism
        midpoint = len(sorted_ids) // 2
        split_a_ids.extend(sorted_ids[:midpoint])
        split_b_ids.extend(sorted_ids[midpoint:])

    split_a = data_csv[data_csv["id"].isin(split_a_ids)]
    split_b = data_csv[data_csv["id"].isin(split_b_ids)]

    # Log split composition for verification
    for name, ids in [("Split A", split_a_ids), ("Split B", split_b_ids)]:
        counts = unique_patients[unique_patients["id"].isin(ids)]["dx"].value_counts().to_dict()
        print(f"{name}: {len(ids)} patients — {counts}")

    if split == 0:
        return split_a, split_b
    else:
        return split_b, split_a


def read_csv(data_folder, subset):
    df = pandas.read_csv(data_folder / (subset + ".csv"))

    expanded_rows = []

    for _, row in df.iterrows():
        subfolder = "control" if row["dx"] == "Control" else "dementia"

        base_path = data_folder / subfolder
        id_str = str(row["id"]).zfill(3)
        pattern = f"{id_str}-*.mp3"
        recording_files = sorted(glob.glob(str(base_path / pattern)))

        for rec_path in recording_files:
            new_row = row.copy()
            new_row["path"] = pathlib.Path(rec_path)
            expanded_rows.append(new_row)

    expanded_df = pandas.DataFrame(expanded_rows)
    return expanded_df


def create_json(json_file, ground_truth, chunk_size, transcripts=None, overlap=None):
    hop_size = chunk_size / 2 if overlap is None else chunk_size - overlap
    json_dict = {}

    for row in ground_truth.to_dict(orient="records"):
        audioinfo = torchaudio.info(row["path"])
        duration = audioinfo.num_frames / audioinfo.sample_rate

        ptype = "Disease" if row["dx"] == "ProbableAD" else "Control"

        max_start = max(duration - hop_size, 1)
        for i, start in enumerate(np.arange(0, max_start, hop_size)):
            chunk_duration = min(chunk_size, duration - i * hop_size)
            json_dict[f"{row['id']}_{i}"] = {
                "wav": str(row["path"]),
                "start": start,
                "duration": chunk_duration,
                "ptype": ptype,
            }

    with open(json_file, mode="w") as json_f:
        json_dict = convert_to_python(json_dict)
        json.dump(json_dict, json_f, indent=2)

