import torch
import json
import os
import glob
import torchaudio
import openpyxl


def prepare_neuro(data_folder, train_annotation, valid_annotation, test_annotation,
                  keep_short_recordings, chunk_length=30):
    try:
        os.listdir(data_folder)
    except:
        print("Data folder not found")
        return

    path_type_dict = get_path_type_dicts(data_folder)

    create_json(train_annotation, path_type_dict["train"], keep_short_recordings, chunk_length)
    create_json(valid_annotation, path_type_dict["valid"], keep_short_recordings, chunk_length)
    create_json(test_annotation, path_type_dict["test"], keep_short_recordings, chunk_length)


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
    batch2_sheet = batch2_workbook['Demographic']

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
        patient_id = sheet.cell(row=row, column=1).value

        if batch == "Batch1":
            patient_type = sheet.cell(row=row, column=2).value
            patient_gender = sheet.cell(row=row, column=3).value
            patient_age = sheet.cell(row=row, column=5).value
            patient_l1 = sheet.cell(row=row, column=4).value
        else:
            patient_type = sheet.cell(row=row, column=4).value
            patient_gender = sheet.cell(row=row, column=5).value
            patient_age = 0
            patient_l1 = sheet.cell(row=row, column=6).value

        # Check if the patient ID is in the recordings, if it is add to dict
        if patient_id is not None and patient_id.rstrip() in pids:
            print(batch)
            print(patient_id)
            print(patient_gender)

            # rstrip() everything
            patient_type = patient_type.rstrip()
            patient_gender = patient_gender.rstrip()
            patient_l1 = patient_l1.rstrip()

            # Refactor patient type
            if patient_type == "CTRL" or patient_type == "control":
                patient_type = "HC"
            elif patient_type == "PD" or patient_type == "patient":
                patient_type = "PD"
            else:
                continue

            # Refactor language
            if patient_l1 == "FR":
                patient_l1 = "French"
            elif patient_l1 == "EN":
                patient_l1 = "English"
            else:
                patient_l1 = "Other"

            # Save to dict
            patient_traits = [patient_type, patient_gender, patient_age, patient_l1]
            patients[patient_id] = patient_traits

    # Change pids to paths
    updated_dict = {}
    for pid in patients:
        for path in files:
            if pid in path:
                updated_dict[path] = patients[pid]

    return updated_dict

def create_json(json_file, path_type_dict, keep_short_recordings, chunk_length):
    json_dict = {}
    
    for audiofile in path_type_dict.keys():

        # Remove 'l1' files as they are duplicates
        if 'l1' in audiofile:
            continue

        # Keep/remove short recordings from the data (repeats/vowels)
        if not keep_short_recordings:
            if 'repeat' in audiofile:
                continue
            if 'a1' in audiofile or 'a2' in audiofile or 'a3' in audiofile or 'a4' in audiofile:
                continue

        # Get PD or HC
        patient_type = path_type_dict[audiofile][0]

        # Get uttid
        uttid = audiofile.split("/")[-1].split(".")[0] + "_" + audiofile.split("/")[-2]

        # Get duration
        audioinfo = torchaudio.info(audiofile)
        duration = audioinfo.num_frames / audioinfo.sample_rate

        # Get gender, age and l1
        patient_gender = path_type_dict[audiofile][1]
        patient_age = path_type_dict[audiofile][2]
        patient_l1 = path_type_dict[audiofile][3]

        # Get test type
        test_type = ""

        if 'repeat' in audiofile:
            test_type = "repeat"
        if 'a1' in audiofile or 'a2' in audiofile or 'a3' in audiofile or 'a4' in audiofile:
            test_type = "vowel_repeat"
        if 'recall' in audiofile:
            test_type = "recall"
        if 'read' in audiofile:
            test_type = "read_text"
        if 'dpt' in audiofile:
            test_type = "dpt"
        if 'hbd' in audiofile:
            test_type = "hbd"
        if test_type == "":
            test_type = "unk"
            print(f"Unknown test found, {audiofile}")

        # Create slow voice features
        audio, fs = torchaudio.load(audiofile)

        # Create entry or entries for this utterance
        base_dict = {
            "wav": audiofile,
            "patient_type": patient_type,
            "patient_gender": patient_gender,
            "patient_age": patient_age,
            "patient_l1": patient_l1,
            "test_type": test_type,
            "duration": duration,
        }
        if duration < chunk_length:
            json_dict[uttid] = {**base_dict, "start": 0}
        else:
            for i in range(int(duration // chunk_length)):
                json_dict[f"{uttid}_{i}"] = {**base_dict, "start": i * chunk_length}

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
