"""
Benchmark DataLoader throughput in isolation (no model, no gradients).

Usage:
    python benchmark_dataloader.py hparams/wavlm_mlp_dementia.yaml \
        --storage_folder /tmp/nd \
        --qpn_data_path /data/qpn \
        --pitt_data_path /data/pitt \
        --delaware_data_path /data/delaware \
        [--batches 100] [--workers 8]

Prints batches/sec and samples/sec for the train DataLoader.
Compare against your observed training it/s — if they are similar, I/O
is the bottleneck. If DataLoader is much faster, the bottleneck is compute.
"""

import sys
import time
import argparse
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.sampler import BalancingDataSampler
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("hparams_file")
    p.add_argument("--batches", type=int, default=100,
                   help="Number of batches to time (default 100)")
    p.add_argument("--workers", type=int, default=None,
                   help="Override num_workers (default: use hparams value)")
    # Accept arbitrary SpeechBrain overrides (key=value pairs)
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


def main():
    args = parse_args()
    # hyperpyyaml expects "key: value\nkey: value", not "key=value key=value"
    overrides_str = "\n".join(o.replace("=", ": ", 1) for o in args.overrides)

    with open(args.hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides_str)

    from prepare_nd import prepare_nd
    prepare_nd(
        qpn_data_path=hparams["qpn_data_path"],
        pitt_data_path=hparams["pitt_data_path"],
        delaware_data_path=hparams["delaware_data_path"],
        train_annotation=hparams["train_annotation"],
        test_annotation=hparams["test_annotation"],
        valid_annotation=hparams["valid_annotation"],
        chunk_size=hparams["chunk_size"],
    )

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(2)
    label_encoder.enforce_label("Disease", 1)
    label_encoder.enforce_label("Control", 0)

    @sb.utils.data_pipeline.takes("wav", "duration", "start")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, duration, start):
        sig, fs = torchaudio.load(
            wav,
            num_frames=int(duration * hparams["sample_rate"]),
            frame_offset=int(start * hparams["sample_rate"]),
        )
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        return sig.squeeze(0)

    @sb.utils.data_pipeline.takes("info_dict")
    @sb.utils.data_pipeline.provides(
        "patient_type", "patient_type_encoded", "dataset", "dataset_ptype"
    )
    def label_pipeline(info_dict):
        yield info_dict["ptype"]
        yield label_encoder.encode_label_torch(info_dict["ptype"])
        yield info_dict["dataset"]
        yield f"{info_dict['dataset']}_{info_dict['ptype']}"

    out_keys = ["id", "sig", "patient_type_encoded", "info_dict", "dataset", "dataset_ptype"]
    train_set = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        dynamic_items=[audio_pipeline, label_pipeline],
        output_keys=out_keys,
    )

    num_workers = args.workers if args.workers is not None else hparams["train_dataloader_options"]["num_workers"]
    batch_size = hparams["train_dataloader_options"]["batch_size"]

    sampler = BalancingDataSampler(
        dataset=train_set,
        key="dataset_ptype",
        num_samples=hparams.get("samples_per_epoch", len(train_set)),
        replacement=True,
    )

    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=sb.dataio.batch.PaddedBatch,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\nBenchmarking DataLoader: {args.batches} batches, "
          f"batch_size={batch_size}, num_workers={num_workers}")
    print("Warming up (5 batches)...")

    loader_iter = iter(loader)
    for _ in range(5):
        next(loader_iter)

    print("Timing...")
    t0 = time.perf_counter()
    for i, _ in enumerate(loader_iter):
        if i + 1 >= args.batches:
            break
    elapsed = time.perf_counter() - t0

    batches_per_sec = args.batches / elapsed
    samples_per_sec = args.batches * batch_size / elapsed

    print(f"\n  batches : {args.batches}")
    print(f"  elapsed : {elapsed:.1f}s")
    print(f"  batches/s: {batches_per_sec:.2f}")
    print(f"  samples/s: {samples_per_sec:.1f}")
    print(f"\nCompare batches/s here against training it/s.")
    print("If they are close  -> I/O is the bottleneck.")
    print("If DataLoader is much faster -> compute is the bottleneck.")

    # Sweep num_workers to find the sweet spot
    print(f"\n--- num_workers sweep (10 batches each) ---")
    for nw in [0, 1, 2, 4, 8]:
        if nw > 16:
            continue
        ldr = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=nw,
            collate_fn=sb.dataio.batch.PaddedBatch,
            pin_memory=torch.cuda.is_available(),
        )
        it = iter(ldr)
        for _ in range(3):   # warmup
            next(it)
        t0 = time.perf_counter()
        for i, _ in enumerate(it):
            if i + 1 >= 10:
                break
        t = time.perf_counter() - t0
        print(f"  workers={nw:2d}  {10/t:.2f} batches/s")


if __name__ == "__main__":
    main()
