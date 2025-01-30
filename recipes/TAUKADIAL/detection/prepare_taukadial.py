import json
import os
import glob
import torchaudio
import numpy as np
import pathlib
import pandas


def prepare_taukadial(
    data_folder, train_annotation, test_annotation, valid_annotation, chunk_size
):
    assert os.path.exists(data_folder), "Data folder not found"
    #if os.path.exists(train_annotation):
    #    return

    # Read csv from file
    data_folder = pathlib.Path(data_folder)
    train_gt = read_csv(data_folder, "train")
    test_gt = read_csv(data_folder, "test")

    # Separate out validation set
    valid_pids = train_gt["pid"].unique()[-10:]
    valid_gt = train_gt[train_gt["pid"].isin(valid_pids)]
    train_gt = train_gt[~train_gt["pid"].isin(valid_pids)]

    # Create json manifests
    create_json(train_annotation, train_gt, chunk_size, overlap=0)
    create_json(test_annotation, test_gt, chunk_size)
    create_json(valid_annotation, valid_gt, chunk_size)


def read_csv(data_folder, subset, filename="groundtruth.csv"):
    df = pandas.read_csv(data_folder / subset / filename)
    df["pid"] = df["tkdname"].str.slice(9, 12)
    df["uttid"] = df["tkdname"].str.slice(9, 14)
    df["path"] = data_folder / subset / df["tkdname"]
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
            json_dict[f"{row['uttid']}_{i}"] = {
                "wav": str(row["path"]),
                "start": start,
                "duration": chunk_duration,
                **{k: v for k, v in row.items() if k in ["age", "sex", "dx", "pid"]}
            }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
