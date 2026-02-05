import json, torchaudio, pathlib, torch, tqdm
import polars as pl
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from speechbrain.lobes.features import VocalFeatures
from torch.nn.functional import pad
import matplotlib.pyplot as plt

from generate_scores_human_samples_qpn import (
    load_audio_and_resample,
    score_items,
)


data_paths = {
    "qpn": "/home/competerscience/Documents/Repositories/ParkinsonSpeechAI/recipes/QPN/interpret/results/",
    "pc_gita": "/home/competerscience/Documents/Repositories/speechbrain/recipes/PC-GITA/pd_detection/manifest.csv",
    "italian": "/home/competerscience/Documents/data/Italian_Parkinsons_Voice_and_Speech/italian_parkinson",
    "mvdr_kcl": "/home/competerscience/Documents/data/mvdr_kcl/",
}
italian_codes = {"B": "read text", "D": "pa-ta", "F": "read phrase"}


#########################
# LOADING DATA
#########################
def load_datasets():
    data = [*load_qpn(), *load_pc_gita(), *load_mvdr_kcl()]
    return pl.DataFrame(data)

def load_qpn():
    # COMBINE VALID AND TEST 
    data = []
    for manifest in ["valid.json", "test.json"]:
        with open(pathlib.Path(data_paths["qpn"]) / manifest) as f:
            data.extend([
                {
                    "uid": uid,
                    "pid": item["info_dict"]["pid"],
                    "sex": item["info_dict"]["sex"],
                    "age": item["info_dict"]["age"],
                    "path": item["wav"],
                    "start": 0,
                    "duration": 180,
                    "task": item["info_dict"]["task"],
                    "status": item["info_dict"]["ptype"],
                    "dataset": "qpn",
                }
                for uid, item in json.load(f).items()
                if uid.endswith("0") and item["info_dict"]["task"] != "vowel_repeat"
            ])

    return data

def load_pc_gita():
    csv = pl.read_csv(data_paths["pc_gita"])
    return [
        {
            "uid": item["ID"],
            "pid": item["pid"],
            "sex": item["SEX"],
            "age": item["AGE"],
            "path": item["wav"],
            "start": 0,
            "duration": 180,
            "task": item["task"],
            "status": "HC" if item["UPDRS"] is None else "PD",
            "dataset": "pc_gita",
        }
        for item in csv.iter_rows(named=True)
        if item["ID"].endswith("0") and item["task"] in ["read text", "monologue"]
    ]

def load_mvdr_kcl():
    return [
        {
            "uid": item.stem + "_" + item.parts[-3],
            "pid": item.stem,
            "path": str(item),
            "start": 0,
            "duration": 180,
            "task": item.parts[-3],
            "status": item.parts[-2],
            "dataset": "mvdr_kcl",
        }
        for item in pathlib.Path(data_paths["mvdr_kcl"]).glob("**/*.wav")
    ]

########################
# MAIN FUNCTION
########################
if __name__ == "__main__":

    # First, load four datasets
    df = load_datasets()
    
    # Add all scores to all items
    df = score_items(df, "save_scores.csv")

    # Write dataset to file for further processing
    df.write_csv("result.csv")
