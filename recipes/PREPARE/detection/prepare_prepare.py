import csv
import json
import os
import pathlib
import torchaudio


def prepare_prepare(data_folder, manifests, valid_count=90, prep_ssl=False):
    assert manifests.keys() == {"train", "valid", "test"}
    assert os.path.exists(data_folder)

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
        if os.path.exists(path):
            continue

        create_json(
            dataset=dataset,
            path=path,
            ids=ids[dataset],
            files=files,
            demographics=demographics,
            annotations=annotations,
            prep_ssl=prep_ssl,
        )

def select_validation(valid_count, annotations):
    """Select an evenly distributed validation set of `valid_count` size"""
    valid_ids = []
    counts = {"diagnosis_control": 0, "diagnosis_mci": 0, "diagnosis_adrd": 0}
    for uid, row in annotations.items():
        for label in counts:
            if row[label] > 0:# and counts[label] < valid_count // 3:
                valid_ids.append(uid)
                counts[label] += 1
                break

        if sum(counts.values()) >= valid_count:
            break

    print("Validation distribution")
    print(counts)

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


def create_json(dataset, path, ids, files, demographics, annotations, prep_ssl):
    """Prepare a json manifest with all relevant info"""
    json_dict = {}
    
    for uid in ids:
        info = torchaudio.info(files[uid])

        json_dict[uid] = make_sample(
            filepath=files[uid],
            duration=info.num_frames / info.sample_rate,
            demographics=demographics[uid],
            annotations=annotations[uid] if uid in annotations else None,
            ssl=files[uid].with_suffix(".pt") if prep_ssl else None,
        )

    # Writing the dictionary to the json file
    with open(path, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

def make_sample(filepath, duration, demographics, annotations=None, ssl=None, chunk=None):
    sample = {
        "filepath": str(filepath),
        "duration": duration,
        **demographics,
    }
    if annotations:
        sample.update(annotations)
    if ssl:
        sample["ssl_path"] = str(ssl)

    return sample
