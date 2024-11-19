import csv
import json
import os
import pathlib
import torchaudio


def prepare_prepare(data_folder, manifests, valid_ratio=0.05):
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
    sorted_ids = list(sorted(annotations.keys()))
    valid_count = int(len(sorted_ids) * valid_ratio)

    ids = {
        "test": demographics.keys() - annotations.keys(),
        "valid": set(sorted_ids[:valid_count]),
        "train": set(sorted_ids[valid_count:]),
    }

    for dataset, path in manifests.items():
        create_json(
            dataset, path, ids[dataset], files, demographics, annotations
        )

def read_csv(filepath, type_fn=None):
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        index, keys = reader.fieldnames[0], reader.fieldnames[1:]
        if type_fn:
            lines = {row[index]: {k: type_fn(row[k]) for k in keys} for row in reader}
        else:
            lines = {row[index]: {k: row[k] for k in keys} for row in reader}
    return lines


def create_json(dataset, path, ids, files, demographics, annotations):

    # First load annotations
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
