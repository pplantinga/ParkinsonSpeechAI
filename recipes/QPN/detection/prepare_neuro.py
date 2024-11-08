import json
import os
import torchaudio
import openpyxl


def prepare_neuro(data_folder, train_annotation, test_annotation, valid_annotation,
                  remove_repeats, remove_keys):
    try:
        os.listdir(data_folder)
    except:
        print("Data folder not found")
        return

    path_type_dict = get_path_type_dicts(data_folder)

    create_json(train_annotation, path_type_dict["train"], remove_repeats, remove_keys)
    create_json(test_annotation, path_type_dict["test"], remove_repeats, remove_keys)
    create_json(valid_annotation, path_type_dict["valid"], remove_repeats, remove_keys)


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

        batch1_files = get_file_paths(batch1_data_path)
        batch2_files = get_file_paths(batch2_data_path)

        batch1_patients = get_patient_traits(batch1_files, batch1_sheet, "Batch1")
        batch2_patients = get_patient_traits(batch2_files, batch2_sheet, "Batch2")

        path_type_dict[dataset] = batch1_patients | batch2_patients

    return path_type_dict


def get_file_paths(path):
    files = os.listdir(path)
    file_paths = []
    for file in files:
        file_path = os.path.join(path, file)
        file_paths.append(file_path)

    return file_paths


def get_patient_traits(files, sheet, batch):
    pids = [path.split("/")[-1].split("_")[1] for path in files]
    patients = {}

    for row in range(2, sheet.max_row + 1):  # Start from row 2 to skip the header
        pid = sheet.cell(row=row, column=1).value
        ptype = sheet.cell(row=row, column=2).value
        gender = sheet.cell(row=row, column=3).value
        l1 = sheet.cell(row=row, column=4).value
        updrs = sheet.cell(row=row, column=5).value

        if batch == "Batch1":
            age = sheet.cell(row=row, column=6).value
        else:
            age = 0

        # Check if the patient ID is in the recordings, if it is add to dict
        if pid is not None and pid.rstrip() in pids:

            # rstrip() everything
            ptype = ptype.rstrip()
            gender = gender.rstrip()
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
            if l1 == "FR":
                l1 = "French"
            elif l1 == "EN":
                l1 = "English"
            else:
                l1 = "Other"

            # TODO Convert UPDRS score to category, waiting for answer from Jen-Kai on B1 scores


            # Save to dict
            patient_traits = {
                "ptype": ptype,
                "gender": gender,
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


def create_json(json_file, path_type_dict, remove_repeats, remove_keys):
    json_dict = {}
    
    for audiofile in path_type_dict.keys():

        # Remove 'l1' files as they are duplicates
        if 'l1' in audiofile:
            continue

        # Keep/remove short recordings from the data (repeats/vowels)
        if remove_repeats:
            if 'repeat' in audiofile:
                continue
            if 'a1' in audiofile or 'a2' in audiofile or 'a3' in audiofile or 'a4' in audiofile:
                continue

        # Get uttid
        uttid = audiofile.split("/")[-1].split(".")[0] + "_" + audiofile.split("/")[-2]

        # Get duration
        audioinfo = torchaudio.info(audiofile)
        duration = audioinfo.num_frames / audioinfo.sample_rate

        # Get test type
        test = ""

        if 'repeat' in audiofile:
            test = "repeat"
        if 'a1' in audiofile or 'a2' in audiofile or 'a3' in audiofile or 'a4' in audiofile:
            test = "vowel_repeat"
        if 'recall' in audiofile:
            test = "recall"
        if 'read' in audiofile:
            test = "read_text"
        if 'dpt' in audiofile:
            test = "dpt"
        if 'hbd' in audiofile:
            test = "hbd"
        if test == "":
            test = "unk"
            print(f"Unknown test found, {audiofile}")

        # Get patient traits and add test to it
        info_dict = path_type_dict[audiofile].copy()
        info_dict["test"] = test

        # Add pid to the patient traits
        info_dict["pid"] = uttid.split("_")[1]

        # Remove keys
        for key in info_dict.keys():
            if key in remove_keys and "test" not in audiofile:
                continue

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": audiofile,
            "duration": duration,
            "info_dict": info_dict,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
