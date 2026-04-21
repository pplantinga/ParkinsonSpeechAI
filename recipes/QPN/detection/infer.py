"""
Inference script for the Parkinson's detection model trained with train.py.

Usage:
    python infer.py hparams.yaml --audio_path path/to/file.wav [--chunk_size 3.0] [--threshold 0.5]
    python infer.py hparams.yaml --audio_dir path/to/dir/ [--output_csv results.csv]

The script reuses your hparams.yaml to reconstruct the exact same module
architecture, then loads the best checkpoint saved by SpeechBrain.
"""

import math
import sys
import argparse
import json
import csv
from pathlib import Path

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from torch.nn.functional import sigmoid

import speechbrain as sb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_brain(hparams_file: str, overrides: str = ""):
    """Load modules + best checkpoint from a SpeechBrain hparams file."""
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # hparams["modules"] is already an OrderedDict of nn.Module — use directly
    modules = hparams["modules"]

    # Recover the best checkpoint (same logic as on_evaluate_start)
    checkpointer = hparams.get("checkpointer")
    avg_threshold = hparams.get("threshold", 0.5)  # fallback default

    if checkpointer is not None:
        checkpoint = checkpointer.recover_if_possible(
            max_key=hparams.get("error_metric"),
        )
        if checkpoint is not None and "comb_avg_threshold" in checkpoint.meta:
            avg_threshold = checkpoint.meta["comb_avg_threshold"]
            print(f"[info] Loaded checkpoint threshold: {avg_threshold:.3f}")
        else:
            print("[warn] No checkpoint threshold found; using fallback.")
    else:
        print("[warn] No checkpointer in hparams; modules must already be loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for mod in modules.values():
        mod.to(device).eval()

    return hparams, avg_threshold, device


def load_audio_chunks(wav_path: str, chunk_size: float, sample_rate: int):
    """
    Load a wav file and split it into non-overlapping chunks of `chunk_size`
    seconds, exactly mirroring the chunking done in prepare_neuro / dataio_prep.

    Returns
    -------
    chunks : list of 1-D float tensors, each of length chunk_size * sample_rate
    """
    sig, fs = torchaudio.load(wav_path)

    # Resample if needed
    if fs != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=sample_rate)
        sig = resampler(sig)

    # Collapse to mono 1-D tensor — handles mono (1, T) and stereo (2, T)
    if sig.dim() > 1:
        sig = sig.mean(dim=0)   # mix channels → (T,)

    chunk_samples = int(chunk_size * sample_rate)

    chunks = []
    start = 0
    while start + chunk_samples <= sig.shape[0]:
        chunks.append(sig[start : start + chunk_samples])
        start += chunk_samples

    # Keep a partial tail chunk if it's at least half a full chunk
    remainder = sig[start:]
    if len(remainder) >= chunk_samples // 2:
        # Pad to full length so feature extraction sees consistent input
        pad = torch.zeros(chunk_samples - len(remainder))
        chunks.append(torch.cat([remainder, pad]))

    if not chunks:
        # File is shorter than half a chunk — use as-is with padding
        pad = torch.zeros(chunk_samples - len(sig))
        chunks.append(torch.cat([sig, pad]))

    return chunks


@torch.no_grad()
def infer_file(wav_path: str, hparams: dict, threshold: float, device: torch.device, batch_size: int = 8):
    """
    Run inference on a single audio file.

    Chunks are processed in mini-batches of `batch_size` to balance
    GPU utilisation against memory usage.

    Returns
    -------
    dict with keys:
        file        : path
        score       : averaged sigmoid probability across chunks  (float, 0-1)
        prediction  : "PD" or "HC"
        chunk_scores: list of per-chunk probabilities
        n_chunks    : int
    """
    chunk_size  = hparams["chunk_size"]
    sample_rate = hparams["sample_rate"]

    chunks = load_audio_chunks(wav_path, chunk_size, sample_rate)
    n_chunks = len(chunks)
    print(f"[info] {Path(wav_path).name}: {n_chunks} chunk(s) → {math.ceil(n_chunks / batch_size)} batch(es) of {batch_size}")

    chunk_scores = []
    for batch_start in range(0, n_chunks, batch_size):
        batch_chunks = chunks[batch_start : batch_start + batch_size]

        # (B, T) — all chunks are the same length so stacking is safe
        batch = torch.stack(batch_chunks).to(device)
        lens  = torch.ones(batch.shape[0], device=device)

        feats      = hparams["modules"]["compute_features"](batch, lens)
        embeddings = hparams["modules"]["embedding_model"](feats)
        logits     = hparams["modules"]["classifier"](embeddings)
        probs      = sigmoid(logits.view(-1))

        chunk_scores.extend(round(p.item(), 4) for p in probs)

        del batch, lens, feats, embeddings, logits, probs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    avg_score  = round(sum(chunk_scores) / len(chunk_scores), 4)
    prediction = "PD" if avg_score >= threshold else "HC"

    # Compute start/end timestamps for each chunk
    chunk_timings = [
        {
            "start_s": round(i * chunk_size, 2),
            "end_s":   round((i + 1) * chunk_size, 2),
        }
        for i in range(n_chunks)
    ]

    return {
        "file":          str(wav_path),
        "score":         avg_score,
        "prediction":    prediction,
        "chunk_scores":  chunk_scores,
        "chunk_timings": chunk_timings,
        "n_chunks":      n_chunks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parkinson's detection inference script.",
        epilog=(
            "Any additional --key value (or --key=value) arguments are forwarded "
            "directly to hyperpyyaml as hparam overrides. "
            "Example: --experiment_name whisper-small --storage_folder ~/scratch/"
        ),
    )
    parser.add_argument("hparams_file", help="Path to hparams.yaml used during training.")

    # Input: single file or directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio_path", help="Path to a single audio file.")
    group.add_argument("--audio_dir",  help="Directory of audio files (searched recursively).")

    parser.add_argument(
        "--chunk_size",
        type=float,
        default=None,
        help="Override chunk size in seconds (default: taken from hparams).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (default: loaded from best checkpoint).",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional path to write results as a CSV file.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to write full results (with chunk scores) as JSON.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of chunks to process in one forward pass (default: 8). Lower if OOM.",
    )

    # Parse known args; anything unrecognised is treated as a hparam override
    args, unknown = parser.parse_known_args()

    # Convert ['--key', 'value', '--flag=val'] → ['key: value', 'flag: val']
    # hyperpyyaml expects a YAML string, not key=value pairs
    overrides = []
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if token.startswith("--"):
            token = token.lstrip("-")
            if "=" in token:
                # --key=value
                k, v = token.split("=", 1)
                overrides.append(f"{k}: {v}")
            elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                # --key value
                overrides.append(f"{token}: {unknown[i + 1]}")
                i += 1
            else:
                # bare flag — treat as key: True
                overrides.append(f"{token}: True")
        else:
            print(f"[warn] Ignoring unrecognised token: {token}")
        i += 1

    args.overrides = overrides
    return args


def fmt_time(seconds: float) -> str:
    """Format seconds as m:ss (e.g. 185.0 → '3:05')."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def main():
    args = parse_args()

    # Build overrides YAML string (newline-separated "key: value" entries)
    overrides_lines = list(args.overrides)
    if args.chunk_size is not None:
        overrides_lines.append(f"chunk_size: {args.chunk_size}")
    overrides_str = "\n".join(overrides_lines)
    if overrides_str:
        print(f"[info] Hparam overrides:\n{overrides_str}")

    print(f"[info] Loading hparams from {args.hparams_file}")
    hparams, threshold, device = load_brain(args.hparams_file, overrides_str)
    print(f"[info] Running on device: {device}")

    # Allow CLI threshold to override checkpoint threshold
    if args.threshold is not None:
        threshold = args.threshold
        print(f"[info] Using CLI threshold: {threshold:.3f}")

    # Collect audio files (wav, mp3, flac, ogg, …)
    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
    if args.audio_path:
        wav_files = [Path(args.audio_path)]
    else:
        wav_files = sorted(
            p for p in Path(args.audio_dir).rglob("*") if p.suffix.lower() in AUDIO_EXTS
        )
        print(f"[info] Found {len(wav_files)} audio file(s) in {args.audio_dir}")

    if not wav_files:
        print("[error] No wav files found. Exiting.")
        sys.exit(1)

    # Run inference
    all_results = []
    all_chunk_rows = []  # flat list for per-chunk CSV
    for wav_path in wav_files:
        result = infer_file(str(wav_path), hparams, threshold, device, batch_size=args.batch_size)

        # Add per-chunk predictions using the same threshold
        result["chunk_predictions"] = [
            "PD" if s >= threshold else "HC" for s in result["chunk_scores"]
        ]
        all_results.append(result)

        # Print file-level summary
        status = "✓" if result["prediction"] == "PD" else "·"
        print(
            f"  [{status}] {wav_path.name:<40}  "
            f"score={result['score']:.4f}  "
            f"prediction={result['prediction']}  "
            f"({result['n_chunks']} chunks)"
        )
        # Print per-chunk detail
        for i, (score, pred, timing) in enumerate(zip(result["chunk_scores"], result["chunk_predictions"], result["chunk_timings"])):
            marker = "✓" if pred == "PD" else "·"
            t_start = fmt_time(timing["start_s"])
            t_end   = fmt_time(timing["end_s"])
            print(f"         [{marker}] chunk {i+1:>3}  {t_start} → {t_end}  score={score:.4f}  {pred}")
            all_chunk_rows.append({
                "file":             str(wav_path),
                "chunk":            i + 1,
                "start":            t_start,
                "end":              t_end,
                "start_s":          timing["start_s"],
                "end_s":            timing["end_s"],
                "chunk_score":      score,
                "chunk_prediction": pred,
                "file_score":       result["score"],
                "file_prediction":  result["prediction"],
            })

    # Summary
    n_pd = sum(1 for r in all_results if r["prediction"] == "PD")
    n_hc = len(all_results) - n_pd
    print(f"\n[summary] {len(all_results)} file(s) → PD: {n_pd}, HC: {n_hc}")

    # Write CSV — one row per chunk
    if args.output_csv:
        csv_fields = ["file", "chunk", "start", "end", "start_s", "end_s", "chunk_score", "chunk_prediction", "file_score", "file_prediction"]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(all_chunk_rows)
        print(f"[info] CSV saved to {args.output_csv}")

    # Write JSON — file-level with chunk scores and predictions nested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[info] JSON saved to {args.output_json}")


if __name__ == "__main__":
    main()
