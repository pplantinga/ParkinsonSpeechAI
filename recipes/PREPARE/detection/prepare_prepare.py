import csv
import json
import os
import pathlib
import torch
import torchaudio


def prepare_prepare(data_folder, manifests, valid_count=90):
    assert manifests.keys() == {"train", "valid", "test"}
    try:
        os.listdir(data_folder)
    except:
        print("Data folder not found")
        return

    data_folder = pathlib.Path(data_folder)
    train_files = (data_folder / "train_audios").glob("*.flac")
    test_files = (data_folder / "test_audios").glob("*.flac")
    files = {
        file.stem: file for file in [*train_files, *test_files]
    }

    demographics = read_csv(data_folder / "metadata.csv")
    annotations = read_csv(data_folder / "train_labels.csv", type_fn=float)
    valid_ids = select_validation(valid_count, annotations)

    ids = {
        "test": demographics.keys() - annotations.keys(),
        "valid": valid_ids,
        "train": annotations.keys() - set(valid_ids),
    }

    for dataset, path in manifests.items():
        create_json(
            dataset, path, ids[dataset], files, demographics, annotations
        )

def select_validation(valid_count, annotations):
    """Select an evenly distributed validation set of `valid_count` size"""
    valid_ids = []
    counts = {"diagnosis_control": 0, "diagnosis_mci": 0, "diagnosis_adrd": 0}
    for uid, row in annotations.items():
        for label in counts:
            if row[label] > 0 and counts[label] < valid_count // 3:
                valid_ids.append(uid)
                counts[label] += 1
                break

    return valid_ids


def read_csv(filepath, type_fn=None):
    """Read a csv file into a dictionary with the first key as index"""
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        index, keys = reader.fieldnames[0], reader.fieldnames[1:]
        if type_fn:
            lines = {row[index]: {k: type_fn(row[k]) for k in keys} for row in reader}
        else:
            lines = {row[index]: {k: row[k] for k in keys} for row in reader}
    return lines


def create_json(dataset, path, ids, files, demographics, annotations):
    """Prepare a json manifest with all relevant info"""
    json_dict = {}
    
    for uid in ids:
        info = torchaudio.info(files[uid])

        json_dict[uid] = {
            "filepath": str(files[uid]),
            "duration": info.num_frames / info.sample_rate,
            **demographics[uid],
        }
        if uid in annotations:
            json_dict[uid].update(annotations[uid])

    # Writing the dictionary to the json file
    with open(path, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
