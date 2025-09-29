import json
import os
import torchaudio
import numpy as np
import pathlib
import pandas


def prepare_adresso(data_folder, test_annotation, chunk_size):
    assert os.path.exists(data_folder), "Data folder not found"

    # Read csv from file
    data_folder = pathlib.Path(data_folder)
    train_gt = read_csv(data_folder, "train")
    test_gt = read_csv(data_folder, "test-dist")
    test_gt = pandas.concat([train_gt, test_gt], ignore_index=True)

    # Create json manifest
    create_json(test_annotation, test_gt, chunk_size)


def read_csv(data_folder, subset):
    df = pandas.read_csv(data_folder / (subset + ".csv"))
    df["path"] = data_folder / "diagnosis" / subset / "audio" / (df["ID"] + ".wav")
    return df

def create_json(json_file, ground_truth, chunk_size, overlap=None):
    hop_size = chunk_size / 2 if overlap is None else chunk_size - overlap
    json_dict = {}

    for row in ground_truth.to_dict(orient="records"):
        # Get duration
        audioinfo = torchaudio.info(row["path"])
        duration = audioinfo.num_frames / audioinfo.sample_rate
        # Write chunks to dict
        max_start = max(duration - hop_size, 1)
        for i, start in enumerate(np.arange(0, max_start, hop_size)):
            chunk_duration = min(chunk_size, duration - i * hop_size)
            json_dict[f"{row['ID']}_{i}"] = {
                "wav": str(row["path"]),
                "start": start,
                "duration": chunk_duration,
                "ptype": row["Dx"],
            }
    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
