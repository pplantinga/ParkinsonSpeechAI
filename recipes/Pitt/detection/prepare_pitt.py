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

def prepare_pitt(data_folder, train_annotation, test_annotation, valid_annotation, chunk_size):

    assert os.path.exists(data_folder), "Data folder not found"

    data_folder = pathlib.Path(data_folder)
    data_csv = read_csv(data_folder, "pitt_corpus")

    # Separate out validation/test sets
    valid_ids = [59, 167, 211, 124, 182, 52, 208, 304, 122, 173, 238, 687, 10, 508, 213, 244]

    valid_gt = data_csv[data_csv["id"].isin(valid_ids)]
    train_gt = data_csv[~data_csv["id"].isin(valid_ids)]

    test_gt = data_csv[data_csv["test"] == 1]
    train_gt = data_csv[data_csv["test"] == 0]

    # Create json manifests
    create_json(train_annotation, train_gt, chunk_size, overlap=0)
    create_json(test_annotation, test_gt, chunk_size)
    create_json(valid_annotation, valid_gt, chunk_size)


def read_csv(data_folder, subset):
    df = pandas.read_csv(data_folder / (subset + ".csv"))

    # Create a list to store expanded entries
    expanded_rows = []

    for _, row in df.iterrows():
        # Determine subfolder based on diagnosis
        subfolder = "control" if row["dx"] == "Control" else "dementia"

        # Check for all recordings of this patient
        base_path = data_folder / subfolder
        # Format ID to be 3 digits with leading zeros
        id_str = str(row['id']).zfill(3)
        pattern = f"{id_str}-*.mp3"
        recording_files = sorted(glob.glob(str(base_path / pattern)))

        # If patient is in test set, only keep the first recording (-0)
        if row["test"] == 1:
            recording_files = [f for f in recording_files if f.endswith("-0.mp3")]

        # Create new entry for each recording
        for rec_path in recording_files:
            new_row = row.copy()
            new_row["path"] = pathlib.Path(rec_path)
            expanded_rows.append(new_row)

    # Create new dataframe with expanded entries
    expanded_df = pandas.DataFrame(expanded_rows)
    return expanded_df

def create_json(json_file, ground_truth, chunk_size, transcripts=None, overlap=None):
    hop_size = chunk_size / 2 if overlap is None else chunk_size - overlap
    json_dict = {}

    for row in ground_truth.to_dict(orient="records"):
        # Get duration
        audioinfo = torchaudio.info(row["path"])
        duration = audioinfo.num_frames / audioinfo.sample_rate

        ptype = "AD" if row["dx"] == "ProbableAD" else "HC"

        # Write chunks to dict
        max_start = max(duration - hop_size, 1)
        for i, start in enumerate(np.arange(0, max_start, hop_size)):
            chunk_duration = min(chunk_size, duration - i * hop_size)
            json_dict[f"{row['id']}_{i}"] = {
                "wav": str(row["path"]),
                "start": start,
                "duration": chunk_duration,
                "ptype": ptype,
            }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json_dict = convert_to_python(json_dict)
        json.dump(json_dict, json_f, indent=2)
