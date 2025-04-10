import json
import os
import glob
import torchaudio
import openpyxl
import numpy as np
import pathlib


def prepare_neuro(
    data_folder, train_annotation, test_annotation, valid_annotation, chunk_size, transcript_folder=None
):
    assert os.path.exists(data_folder), "Data folder not found"

    if os.path.exists(train_annotation):
        return

    path_type_dict = get_path_type_dicts(data_folder)

    train_transcripts, valid_transcripts, test_transcripts, train_translations, manual = read_transcripts(transcript_folder)
    create_json(train_annotation, path_type_dict["train"], chunk_size, train_transcripts, manual, train_translations, overlap=0)
    create_json(test_annotation, path_type_dict["test"], chunk_size, test_transcripts, manual)
    create_json(valid_annotation, path_type_dict["valid"], chunk_size, valid_transcripts, manual)


def read_transcripts(transcript_dir):
    """Read the transcripts from file."""
    if transcript_dir is None:
        return None, None, None, None, None

    d = pathlib.Path(transcript_dir)
    assert d.exists(), "Transcript dict must exist and hold train.json etc."
    with open(d / "train.json") as f:
        train_dict = json.load(f)
    with open(d / "valid.json") as f:
        valid_dict = json.load(f)
    with open(d / "test.json") as f:
        test_dict = json.load(f)
    with open(d / "train_translation.json") as f:
        train_translations = json.load(f)
    with open(d / "manual.json") as f:
        manual = json.load(f)

    return train_dict, valid_dict, test_dict, train_translations, manual


def get_path_type_dicts(data_folder):
    """
    Function that extracts the patient_type and the path for each recording.
    Takes data_folder path, returns dicts with path:patient_type for each set.

    :param data_folder: string
    :return: dicts
    """

    datasets = os.listdir(data_folder)
    batch1_excel_path = os.path.join(data_folder, "QPN_Batch1.xlsx")
    batch2_excel_path = os.path.join(data_folder, "QPN_Batch2.xlsx")
    path_type_dict = {}

    # Load the Excel files
    batch1_workbook = openpyxl.load_workbook(batch1_excel_path)
    batch1_sheet = batch1_workbook.active
    batch2_workbook = openpyxl.load_workbook(batch2_excel_path)
    batch2_sheet = batch2_workbook["Demographic"]

    for dataset in datasets:
        dataset_path = os.path.join(data_folder, dataset)

        # Skip files
        if os.path.isfile(dataset_path):
            continue
        if dataset == "noise" or dataset == "rir":
            continue

        batch1_data_path = os.path.join(dataset_path, "Batch1")
        batch2_data_path = os.path.join(dataset_path, "Batch2")

        batch1_files = glob.glob(batch1_data_path + "/*.wav")
        batch2_files = glob.glob(batch2_data_path + "/*.wav")

        batch1_patients = get_patient_traits(batch1_files, batch1_sheet, "Batch1")
        batch2_patients = get_patient_traits(batch2_files, batch2_sheet, "Batch2")

        path_type_dict[dataset] = batch1_patients | batch2_patients

    return path_type_dict


def get_patient_traits(files, sheet, batch):
    pids = [path.split("/")[-1].split("_")[1] for path in files]
    patients = {}

    for row in range(2, sheet.max_row + 1):  # Start from row 2 to skip the header
        pid = sheet.cell(row=row, column=1).value
        ptype = sheet.cell(row=row, column=2).value
        sex = sheet.cell(row=row, column=3).value
        l1 = sheet.cell(row=row, column=4).value
        updrs = sheet.cell(row=row, column=5).value
        age = sheet.cell(row=row, column=6).value

        # Check if the patient ID is in the recordings, if it is add to dict
        if pid is not None and pid.rstrip() in pids:

            # rstrip() everything
            ptype = ptype.rstrip()
            sex = sex.rstrip()
            l1 = l1.rstrip()

            # Refactor patient type
            if ptype == "CTRL" or ptype == "control":
                ptype = "HC"
            elif ptype == "PD" or ptype == "patient":
                ptype = "PD"
            else:
                print(f"Unknown key found: {ptype}")
                continue

            # Refactor language
            # TODO fix excel to have unified language identifiers
            if l1 == "FR" or "French" in l1 or "Fench" in l1:
                l1 = "French"
            elif l1 == "EN" or "English" in l1:
                l1 = "English"
            else:
                l1 = "Other"

            # TODO Convert UPDRS score to category, waiting for answer from Jen-Kai on B1 scores

            # Save to dict
            patient_traits = {
                "ptype": ptype,
                "sex": sex,
                "age": age,
                "l1": l1,
                "updrs": updrs,
            }
            patients[pid] = patient_traits

    # Change pids to paths
    updated_dict = {}
    for pid in patients:
        for path in files:
            if pid in path:
                updated_dict[path] = patients[pid]

    return updated_dict


def create_json(json_file, path_type_dict, chunk_size, transcripts=None, manual=None, translations=None, overlap=None):
    hop_size = chunk_size / 2 if overlap is None else chunk_size - overlap
    json_dict = {}

    for audiofile in path_type_dict.keys():
        # Get info dict
        info_dict = path_type_dict[audiofile].copy()

        # Skip condition
        skip = False

        # Remove 'l1' files as they are duplicates
        if "l1" in audiofile:
            continue

        # Get utterance info from the file name
        audiopath = pathlib.Path(audiofile)
        uttid = audiopath.stem + "_" + audiopath.parent.name

        # Second and third items are the PID and task, last is usually the language
        items = audiopath.stem.split("_")
        info_dict.update({"pid": items[1], "task": items[2], "lang": items[-1]})

        # Corrections
        if info_dict["task"] in ["a1", "a2", "a3", "a4"]:
            info_dict["task"] = "vowel_repeat"
        if info_dict["lang"] not in ["en", "fr"]:
            info_dict["lang"] = "other"

        # Get duration
        audioinfo = torchaudio.info(audiofile)
        duration = audioinfo.num_frames / audioinfo.sample_rate

        if transcripts is not None:
            transcript = transcripts[uttid] if uttid in transcripts else ""
            json_dict[uttid] = {
                "wav": audiofile, "info_dict": info_dict, "transcript": transcript
            }
            if translations is not None:
                translation = translations[uttid] if uttid in translations else ""
                json_dict[uttid]["translation"] = translation

            # Manual transcription
            patientid, langid = uttid.split("_")[1], uttid.split("_")[-2]
            manual_id = patientid + "_" + langid
            if manual is not None and manual_id in manual:
                json_dict[uttid]["manual"] = manual[manual_id]
            else:
                json_dict[uttid]["manual"] = ""

        else:
            max_start = max(duration - hop_size, 1)
            for i, start in enumerate(np.arange(0, max_start, hop_size)):
                chunk_duration = min(chunk_size, duration - i * hop_size)
                json_dict[f"{uttid}_{i}"] = {
                    "wav": audiofile,
                    "start": start,
                    "duration": chunk_duration,
                    "info_dict": info_dict,
                }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
