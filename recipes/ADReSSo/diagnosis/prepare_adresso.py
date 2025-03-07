import json
import os
import glob
import torchaudio
import numpy as np
import pathlib
import pandas


def prepare_adresso(
    data_folder, train_annotation, test_annotation, valid_annotation, chunk_size, transcript_folder=None
):
    assert os.path.exists(data_folder), "Data folder not found"
    #if os.path.exists(train_annotation):
    #    return

    # Read csv from file
    data_folder = pathlib.Path(data_folder)
    train_gt = read_csv(data_folder, "train")
    test_gt = read_csv(data_folder, "test-dist")

    train_transcripts = read_transcripts(transcript_folder, "train.json")
    valid_transcripts = read_transcripts(transcript_folder, "valid.json")
    test_transcripts = read_transcripts(transcript_folder, "test.json")

    # Separate out validation set
    valid_pids = ["adrso247", "adrso248", "adrso249", "adrso250", "adrso253", "adrso257", "adrso259", "adrso260", "adrso261", "adrso262"]

    valid_gt = train_gt[train_gt["ID"].isin(valid_pids)]
    train_gt = train_gt[~train_gt["ID"].isin(valid_pids)]

    # Create json manifests
    create_json(train_annotation, train_gt, chunk_size, train_transcripts, overlap=0)
    create_json(test_annotation, test_gt, chunk_size, test_transcripts)
    create_json(valid_annotation, valid_gt, chunk_size, valid_transcripts)


def read_csv(data_folder, subset):
    df = pandas.read_csv(data_folder / (subset + ".csv"))
    df["path"] = data_folder / "diagnosis" / subset / "audio" / (df["ID"] + ".wav")
    return df


def read_transcripts(transcript_folder, filename):
    if transcript_folder is None:
        return None
    with open(pathlib.Path(transcript_folder) / filename) as f:
        return json.load(f)


def create_json(json_file, ground_truth, chunk_size, transcripts=None, overlap=None):
    hop_size = chunk_size / 2 if overlap is None else chunk_size - overlap
    json_dict = {}

    for row in ground_truth.to_dict(orient="records"):
        # Get duration
        audioinfo = torchaudio.info(row["path"])
        duration = audioinfo.num_frames / audioinfo.sample_rate

        if transcripts is not None:
            key = row["path"].name
            transcript = transcripts[key] if key in transcripts else ""
            json_dict[row['ID']] = {
                "wav": str(row["path"]), "dx": row["Dx"], "transcript": transcript
            }
        else:
            # Write chunks to dict
            max_start = max(duration - hop_size, 1)
            for i, start in enumerate(np.arange(0, max_start, hop_size)):
                chunk_duration = min(chunk_size, duration - i * hop_size)
                json_dict[f"{row['ID']}_{i}"] = {
                    "wav": str(row["path"]),
                    "start": start,
                    "duration": chunk_duration,
                    "dx": row["Dx"],
                }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
