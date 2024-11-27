import csv
import json
import os
import pathlib
import torch
import torchaudio


def prepare_prepare(
    data_folder,
    manifests,
    chunk_size=None,
    hop_size=None,
    valid_count=90,
    prep_ssl=False,
):
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
            chunk_size=chunk_size,
            hop_size=hop_size,
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


def create_json(
    dataset, path, ids, files, demographics, annotations, prep_ssl, chunk_size, hop_size
):
    """Prepare a json manifest with all relevant info"""
    json_dict = {}
    
    for uid in ids:
        info = torchaudio.info(files[uid])
        duration = info.num_frames / info.sample_rate
        sample_max = info.num_frames - int(chunk_size * info.sample_rate) + 1
        hop_samples = int(hop_size * info.sample_rate)
        ssl = files[uid].with_suffix(".pt") if prep_ssl else None
        annot = annotations[uid] if uid in annotations else None

        if not chunk_size or duration < chunk_size:
            json_dict[uid] = make_sample(
                filepath=files[uid],
                duration=duration,
                demographics=demographics[uid],
                start=0,
                annotations=annot,
                ssl=ssl,
            )
        else:
            arange = torch.arange(0, duration - chunk_size + hop_size, hop_size)
            for i, start in enumerate(arange):
                json_dict[f"{uid}_{i}"] = make_sample(
                    filepath=files[uid],
                    duration=chunk_size,
                    demographics=demographics[uid],
                    start=start.item(),
                    annotations=annot,
                    ssl=ssl,
                    chunk=i,
                )

    # Writing the dictionary to the json file
    with open(path, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

def make_sample(filepath, duration, demographics, start, annotations=None, ssl=None, chunk=None):
    sample = {
        "filepath": str(filepath),
        "duration": duration,
        "start": start,
        **demographics,
    }
    if annotations:
        sample.update(annotations)
    if ssl:
        start_frame = int(start * 50)
        dur_frames = int(duration * 50)
        feats = torch.load(ssl, map_location="cpu")
        chunk_path = ssl.with_stem(f"{ssl.stem}_{chunk}")
        feat_part = feats[start_frame:start_frame + dur_frames].clone()
        torch.save(feat_part, chunk_path)
        sample["ssl_path"] = str(chunk_path)

    return sample
