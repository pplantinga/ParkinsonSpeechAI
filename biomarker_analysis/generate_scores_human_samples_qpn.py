"""
Author: Peter Plantinga
Date: January 2026

Generate a few different interpretable scores and corresponding predictions for our human-rated samples.

Total list of scores, by section follows

Voice
-----
* GNE (Glottal-to-Noise Excitation)

Articulation
------------
* SFSB (Spectral Flux at Speech Boundaries)
* VOT (Voice Onset Delay)

Prosody
-------
* DPI (Duration of Pause Intervals)
* UPR (Unfilled Pause Ratio, from Dominique)
* F0SD (F0 standard deviation, aka monotone)

The following publication finds VOT and F0SD and DPI as the separable features:
https://onlinelibrary.wiley.com/doi/epdf/10.1002/ana.26085

Another publication finds only DPI (and a similar RST) to be separable:
www.nature.com/articles/s41598-017-00047-5#Fig3

A third publication finds F0SD and IntSD (Intensty SD) and RFA (Resonant Frequency Attenuation):
https://onlinelibrary.wiley.com/doi/epdf/10.1111/ene.15099

"""

import json, pathlib, torch, tqdm, speechbrain, torchaudio
import polars as pl
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from speechbrain.lobes.features import VocalFeatures
from torch.nn.functional import pad
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import *
from sklearn.metrics import accuracy_score


data_paths = {
    "qpn": "/home/competerscience/Documents/Repositories/ParkinsonSpeechAI/recipes/QPN/interpret/results/",
}
SCORES = ["SFSB", "F0SD", "DPI", "UPR", "VOT", "GNE"]

def load_qpn(filename):
    with open(pathlib.Path(data_paths["qpn"]) / filename) as f:
        data = [
            {
                "uid": uid.strip("_"),
                "pid": item["info_dict"]["pid"],
                "sex": item["info_dict"]["sex"],
                "age": item["info_dict"]["age"],
                "path": item["wav"],
                "start": item["start"],
                "duration": item["duration"],
                "task": item["info_dict"]["task"],
                "updrs": item["info_dict"]["updrs"],
                "status": item["info_dict"]["ptype"],
                "age_range": get_age_range(item["info_dict"]["age"]),
                "severity": get_severity(item["info_dict"]["updrs"]),
                "first_lang": get_first_lang(item["info_dict"]),
            }
            for uid, item in json.load(f).items()
            if item["info_dict"]["task"] != "vowel_repeat" and item["duration"] > 5
        ]

    return pl.DataFrame(data)

def load_audio_and_resample(path, start, duration, sample_rate=16000, device="cuda"):
    frame_offset = int(start * sample_rate)
    num_frames = int(duration * sample_rate)
    audio, sr = speechbrain.dataio.audio_io.load(path, frame_offset=frame_offset, num_frames=num_frames, always_2d=False)
    #assert sr == sample_rate
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=sample_rate)


    # Normalize every few seconds independently to maintain an even volume as spectral flux can be influenced by volume
    chunk_size = 5 * sample_rate
    if len(audio) >= chunk_size:
        audio = audio.unfold(dimension=0, size=chunk_size, step=chunk_size)
    else:
        audio = audio.unsqueeze(0)

    # Create speech mask and normalize by the volume of speech
    speech_mask = audio.abs() > 0.1 * audio.abs().mean()
    speech_means = (audio * speech_mask).abs().sum(dim=1, keepdim=True) / (speech_mask.sum(dim=1, keepdim=True) + 1)
    audio = audio / speech_means / 100

    return audio.view(-1).to(device)

def get_severity(updrs):
    if not updrs:
        return "unknown"
    elif updrs < 33:
        return "mild"
    elif updrs < 59:
        return "moderate"
    else:
        return "severe"

def get_age_range(age):
    if not age:
        return "unknown"
    elif 52 < age < 63:
        return "53-62"
    elif 62 < age < 73:
        return "63-72"
    elif 72 < age < 83:
        return "73-82"
    else:
        return "unknown"

def get_first_lang(item):
    """French should match fr and English should match en"""
    return item["l1"].lower().startswith(item["lang"])

#########################
# SCORE ITEMS
#########################
def score_items(dataset, save_csv, device="cuda"):

    # Load from cache if it exists
    if pathlib.Path(save_csv).exists():
        return pl.read_csv(save_csv)

    scores = {s: [] for s in SCORES}
    vocal_features = VocalFeatures(step_size=0.02, log_scores=False)

    score_fns = {
        "SFSB": compute_sfb,
        "F0SD": compute_f0sd,
        "DPI": compute_dpi,
        "UPR": compute_upr,
        "VOT": compute_vot,
        "GNE": compute_gne,
    }
        
    for item in tqdm.tqdm(list(dataset.iter_rows(named=True))):
        # Compute audio and features, used in several scores.
        audio = load_audio_and_resample(
            item["path"], item["start"], item["duration"], sample_rate=16000, device=device
        )
        feats = vocal_features(audio.unsqueeze(0)).squeeze(0).transpose(0, 1)
        sq_audio = audio.squeeze().square()
        energy = sq_audio.unfold(dimension=0, size=640, step=320).sum(dim=-1).squeeze()
        energy = match_len(energy, feats.size(1))

        # Use simple energy-based vad to determine speech for all tasks
        # And use HNR < 0.5 to determine voicing
        speech_segments = extract_segments(energy, 0.01, invert=False)
        voiced_segments = extract_segments(feats[1], 0.5, invert=True) * speech_segments

        # Compute all scores
        for key, compute_score in score_fns.items():
            scores[key].append(compute_score(feats, speech_segments, voiced_segments))

    # Convert all to numpy and extend dataframe
    df = dataset.with_columns(**{k: np.array(v) for k, v in scores.items()})

    df.write_csv(save_csv)
    df = pl.read_csv(save_csv)
    
    return df

def match_len(tensor, size):
    """Length should be last dimension"""
    if tensor.size(-1) == size:
        return tensor
    elif tensor.size(-1) < size:
        return pad(tensor, (0, size - tensor.size(-1)))
    else:
        return tensor[..., :size]

def extract_segments(signal, threshold, w=9, invert=False):
    """Divide into segments based on thresholds, avoiding short segments"""
    signal = pad(signal, (w // 2, w // 2 + 1))
    medians = signal.unfold(0, w, 1).median(dim=-1).values
    if invert:
        return (medians < threshold).int()
    else:
        return (medians > threshold).int()
        

def compute_sfb(feats, speech_segments, voiced_segments):
    """Articulation score is mean spec. flux at word boundaries."""
    del voiced_segments # not used
    
    diff = speech_segments.diff().abs()
    frames = match_len(diff, feats.size(1))
    score = (feats[12] * frames).sum() / frames.sum()

    return score.cpu().detach()


def compute_f0sd(feats, speech_segments, voiced_segments):
    """Measure monotone, std of f0 in voiced segments"""
    del speech_segments # not used
    
    segments = match_len(voiced_segments, len(feats[0])).bool()
    
    if segments.sum() > 2:
        return feats[0][segments].std().cpu().detach()
    else:
        return float('nan')


def compute_dpi(feats, speech_segments, voiced_segments):
    """From first onset to last offset, avg pause length"""
    del feats # not used
    del voiced_segments # not used
    
    _, speech_pause_counts = speech_segments.unique_consecutive(return_counts=True)

    # Skip silence before/after speech
    pause_counts = speech_pause_counts[2:-2:2].float()

    # Sample rate is 50 Hz, convert to seconds
    return pause_counts.mean().cpu().detach() / 50

    
def compute_upr(feats, speech_segments, voiced_segments):
    """From first onset to last offset, unfilled pause ratio"""
    del feats # not used
    del voiced_segments # not used

    _, speech_pause_counts = speech_segments.unique_consecutive(return_counts=True)

    # Skip silence before/after speech
    pause_counts = speech_pause_counts[2:-2:2]

    # Compute number of unfilled pauses longer than 0.5 seconds (50 Hz * 0.5s = 25 frames)
    unfilled_pause_count = (pause_counts > 25).sum()

    # Ratio of unfilled pauses to total pauses
    return unfilled_pause_count.cpu().detach() / (len(pause_counts) + 1)


def compute_gne(feats, speech_segments, voiced_segments):
    """5th index of feats (#4) is GNE, used for voice disorders."""
    del speech_segments # not used
    
    segments = match_len(voiced_segments, len(feats[0])).bool()
    return (feats[3] * segments).mean().cpu().detach()


def compute_vot(feats, speech_segments, voiced_segments):
    """Find the mean time from speech onset to voice onset"""
    del feats # not used
    
    values, counts = (speech_segments + voiced_segments).unique_consecutive(return_counts=True)
    if len(values) < 3:
        return 0
    
    # Find onset scenarios (0 -> 1 -> 2)
    pattern = torch.tensor([0, 1, 2], device=values.device)
    onsets = values.unfold(dimension=0, size=3, step=1).eq(pattern).all(dim=1)

    # Find zero-delay onsets (0 -> 2)
    pattern = torch.tensor([0, 2], device=values.device)
    zero_onsets = values.unfold(dimension=0, size=2, step=1).eq(pattern).all(dim=1)

    # Count onsets and add one to handle no onsets
    onset_count = onsets.sum() + zero_onsets.sum() + 1

    # Take the frame counts at the "1" position (out of 0 -> 1 -> 2)
    delays = counts[pad(onsets, (1, 1))].float().sum()

    # Return average length, in seconds (50 Hz sample rate)
    return (delays / onset_count).cpu().detach() / 50


def generate_scores(train_df, test_df):

    # Expecting only a single subject in the test df
    sex = test_df[0]["sex"]
    train_sub = train_df.filter(train_df["sex"] == sex)
    test_sub = test_df.filter(test_df["sex"] == sex)
    
    for metric in SCORES:
        
        clf = LogisticRegression(random_state=0, class_weight="balanced", solver="liblinear")
        x_train = np.array(train_sub[metric + "_znorm"])[:, None]
        x_test = np.array(test_sub[metric + "_znorm"])[:, None]

        clf.fit(x_train, train_sub["status"])
        yhat = clf.predict(x_test)
        corr = (test_sub["status"].to_numpy() == yhat).astype(int)
        test_sub = test_sub.with_columns(**{metric + "_corr": corr})
    
    return test_sub

    
def category_perf(test_df):
    scores = {s: {} for s in SCORES}
    cat_lists = {
        "task": ["read", "recall", "dpt", "repeat"],
        "severity": ["mild", "moderate", "severe"],
        "age_range": ["53-62", "63-72", "73-82"],
        "sex": ["M", "F"],
    }
    title_mapping = {
        "recall": "Memory", "dpt": "Picture", "M": "Male", "F": "Female",
        "task": "Spoken Task", "age_range": "Age Range",
    }
    def title(cat):
        return title_mapping[cat] if cat in title_mapping else cat.title()
        
    for category, cat_list in cat_lists.items():
        for cat in cat_list:
            df_ = test_df.filter(test_df[category] == cat)
            for s in scores:
                scores[s][title(cat)] = df_[f"{s}_corr"].mean()

    # Format for export
    perf = pl.DataFrame(list(scores.values()))
    perf = perf.transpose(include_header=True, header_name="Value", column_names=list(scores.keys()))
    perf = perf.with_columns(
        *[pl.col(n).round(4) for n in scores],
        Category=pl.Series([title(c) for c, v in cat_lists.items() for _ in v])
    )
    return perf.select(["Category", "Value"] + list(scores.keys()))


def standardize(df, column_name):
    """Apply z-norm to a column on two dataframes"""
    col_mean = df[column_name].mean()
    col_std = df[column_name].std()
    norm_fn = ((pl.col(column_name) - col_mean) / col_std).alias(f"{column_name}_znorm")
    return df.with_columns(norm_fn)


########################
# MAIN FUNCTION
########################
if __name__ == "__main__":

    # First, load datasets
    train_df = load_qpn("train.json")
    valid_df = load_qpn("valid.json")
    test_df = load_qpn("test.json")

    # Add all scores to all items
    train_df = score_items(train_df, "train_scores.csv")
    print(len(train_df) - len(train_df.drop_nans()), "nans in train")
    test_df = score_items(test_df, "test_scores.csv")
    valid_df = score_items(valid_df, "valid_scores.csv")

    # Normalize columns for two-feature logistic regression
    df = train_df.vstack(valid_df).vstack(test_df).drop_nans()
    for feature in SCORES:
        df = standardize(df, feature)

    # leave-one-subject-out evaluation
    predictions = []
    for pid in df["pid"].unique():
        left_out_pid_samples = df.filter(df["pid"] == pid)
        left_in_pids_samples = df.filter(df["pid"] != pid)
    
        # Create predictions for test samples
        predictions.append(generate_scores(left_in_pids_samples, left_out_pid_samples))

    # Aggregate subjects and pull out human rated samples
    machine_predictions = pl.concat(predictions)
    
    # Write predictions for analysis
    machine_predictions.write_csv("machine_predictions.csv")

    # Just for funsies, check performance across categories
    subset_scores = category_perf(machine_predictions)
    with pl.Config(tbl_rows=len(subset_scores), tbl_cols=20):
        print(subset_scores)
