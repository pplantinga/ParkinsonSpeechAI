# !/usr/bin/python3
"""Recipe for training a detector on the McGill Neuro Parkinson's dataset.
We employ an encoder followed by a classifier.

To run this recipe, use the following command:
> python train_nd.py {hyperparameter_file} {overrides}

Using your own hyperparameter file or one of the following:
    hparams/wavlm_ecapa.yaml (for wavlm + ecapa)

Author
    * Briac Cordelle 2026
"""

import os
import random
import sys
import csv
import json
import logging
import pprint
import tempfile
import collections

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.sampler import BalancingDataSampler, ReproducibleWeightedRandomSampler

from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm
import opensmile

from sklearn.metrics import roc_auc_score

logger = sb.utils.logger.get_logger("train.py")

class NdBrain(sb.core.Brain):
    """Class for speaker embedding training"""

    def compute_forward(self, batch, stage):
        """
        Computation pipeline based on a encoder + speaker classifier for parkinson's detection.
        Data augmentation and environmental corruption are applied to the
        input speech if present.
        """

        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Augmentations, if specified
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Compute features
        feats = self.modules.compute_features(wavs, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        # Outputs
        return outputs, lens

    def compute_objectives(self, outputs, batch, stage):
        """Computes the loss using patient-type as label."""

        # Get predictions and labels
        labels, _ = batch.patient_type_encoded
        outputs, lens = outputs

        # Concatenate labels in case of wav_augment
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            labels = self.hparams.wav_augment.replicate_labels(labels)

        # Compute loss
        if stage == sb.Stage.TRAIN:
            loss = self.hparams.bce_loss(outputs, labels)

        # Validation / Test
        else:
            probs = torch.sigmoid(outputs.view(-1))
            self.error_metrics.append(batch.id, probs, labels.view(-1))
            self.error_metrics.info_dicts.extend(batch.info_dict)

            # Use unweighted, unsmoothed score for comparable results across hparams
            loss = binary_cross_entropy(probs, labels.view(-1).float())

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

            # Add this list so we can store the info dict
            self.error_metrics.info_dicts = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            # Combine chunks
            combined_avg = self.combine_chunks(how="avg")
            self._log_score_distribution(combined_avg, stage, epoch)

            # Build a metrics object over the combined (utt-level) scores
            avg_threshold = None if stage == sb.Stage.VALID else self.avg_threshold
            comb_metrics = self.hparams.error_stats()
            for utt_id, entry in combined_avg.items():
                comb_metrics.ids.append(utt_id)
                comb_metrics.scores.append(torch.tensor(entry["combined"]))
                comb_metrics.labels.append(torch.tensor(entry["label"]))

            comb_avg_stats = self.summarize_metrics(comb_metrics, threshold=avg_threshold)
            stage_stats.update({f"comb_avg_{k}": v for k, v in comb_avg_stats.items()})

            # Dump metrics to file only on test
            if stage == sb.Stage.TEST:
                with open(self.metrics_json, "w") as f:
                     json.dump(combined_avg, f)
                     f.write("\nCombined stats: ")
                     json.dump(comb_avg_stats, f)

                     # Per-dataset breakdown
                     dataset_breakdown = self.metrics_by_category(
                         combined_avg,
                         target_category="dataset",
                         threshold=self.avg_threshold,
                     )

                     f.write("\nPer-dataset stats: ")
                     json.dump(dataset_breakdown, f)

                logger.info(f"Results stored {self.metrics_json}")
                logger.info(f"Per-dataset breakdown: {pprint.pformat(dataset_breakdown)}")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            if hasattr(self.hparams, "lr_scheduler"):
               lr = self.lr_scheduler.get_last_lr()
            else:
               lr = self.optimizer.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                 stats_meta={"epoch": epoch, "lr": lr},
                 train_stats=self.train_stats,
                 valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats,
                max_keys=self.hparams.error_metric,
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Recover best checkpoint for evaluation, keeping track of threshold"""
        if self.checkpointer is not None:
            checkpoint = self.checkpointer.recover_if_possible(
                min_key=min_key, max_key=max_key
            )
            self.avg_threshold = checkpoint.meta["comb_avg_threshold"]

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """

        all_params = self.modules.parameters()

        if self.opt_class is not None:
            self.optimizer = self.opt_class(all_params)
            self.optimizers_dict = {"opt_class": self.optimizer}

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            if hasattr(self.hparams, "lr_scheduler"):
                self.lr_scheduler = self.hparams.lr_scheduler(self.optimizer)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Update scheduler if an update was made."""
        if should_step and hasattr(self.hparams, "lr_scheduler"):
            self.lr_scheduler.step()

    def _log_score_distribution(self, combined_avg, stage, epoch):
        """Log mean predicted score ± std per (dataset, ptype) cell."""
        buckets = collections.defaultdict(list)
        for entry in combined_avg.values():
            key = (entry.get("dataset", "?"), entry.get("ptype", "?"))
            buckets[key].append(entry["combined"])

        stage_name = "valid" if stage == sb.Stage.VALID else "test"
        epoch_str = f" epoch {epoch}" if epoch is not None else ""
        lines = [f"[{stage_name}{epoch_str}] score distribution (Disease=1, Control=0)"]

        for ds in sorted({ds for ds, _ in buckets}):
            for pt in ("Control", "Disease"):
                scores = buckets.get((ds, pt), [])
                if not scores:
                    continue
                mean = sum(scores) / len(scores)
                std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
                lines.append(
                    f"  {str(ds):<10} {pt:<8}  mean={mean:.3f}  std={std:.3f}  n={len(scores)}"
                )

        logger.info("\n".join(lines))

    def combine_chunks(self, how="avg"):
        """Aggregates predictions made on all individual chunks"""
        ids = self.error_metrics.ids
        scores = self.error_metrics.scores
        labels = self.error_metrics.labels
        info_dicts = self.error_metrics.info_dicts

        combined_scores = {}
        for i, score, label, info_dict in zip(ids, scores, labels, info_dicts):
            utt_id, chunk = i.rsplit("_", 1)

            if utt_id not in combined_scores:
                combined_scores[utt_id] = {
                    "scores": [round(score.item(), 3)],
                    "label": label.item(),
                    **info_dict,
                }
            else:
                combined_scores[utt_id]["scores"].append(round(score.item(), 3))

        # For now just take the average or max. Perhaps do something fancier later
        for utt_id in combined_scores:
            scores = combined_scores[utt_id]["scores"]
            if how == "avg":
                combined_scores[utt_id]["combined"] = round(
                    sum(scores) / len(scores), 3
                )
            elif how == "max":
                combined_scores[utt_id]["combined"] = round(max(scores), 3)
            else:
                raise ValueError("Expected 'avg' or 'max'")

        return combined_scores

    def metrics_by_category(
        self, combined_scores, target_category=None, threshold=None
    ):
        """Divides metrics by a given category."""

        # Collect available elements in the target_category, or "overall"
        options = {"overall"}
        if target_category:
            options = set(t[target_category] for t in combined_scores.values())

        # Separate the scores into individual metrics objects
        metrics = {option: self.hparams.error_stats() for option in options}
        for utt_id, categories in combined_scores.items():
            option = categories[target_category] if target_category else "overall"
            metrics[option].ids.append(utt_id)
            metrics[option].scores.append(torch.tensor(categories["combined"]))
            metrics[option].labels.append(torch.tensor(categories["label"]))

        # Summarize scores
        breakdown = {
            option: self.summarize_metrics(metrics[option], threshold=threshold)
            for option in options
        }

        return breakdown

    def summarize_metrics(self, metrics, threshold):
        """Simplify metrics to round(100 * (P, R, F1, AUC)), x-ent, and threshold"""
        all_metrics = metrics.summarize(threshold=threshold)
        target_metrics = ["precision", "recall", "F-score"]
        summary = {k: round(100 * all_metrics[k], 2) for k in target_metrics}
        summary["threshold"] = round(all_metrics["threshold"], 3)

        cross_ent = binary_cross_entropy(metrics.scores, metrics.labels.float())
        summary["bce"] = round(cross_ent.item(), 3)

        try:
            auc = roc_auc_score(
                metrics.labels.float().cpu().numpy(),
                metrics.scores.float().cpu().numpy(),
            )
            summary["auc"] = round(100 * auc, 2)
        except ValueError:
            # roc_auc_score throws if only one class is present in the subset
            # This can happen in small per-dataset slices
            summary["auc"] = None

        return summary

def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'hc': 0, 'pd': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(2)
    label_encoder.enforce_label("Disease", 1)
    label_encoder.enforce_label("Control", 0)

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "duration", "start")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, duration, start):
        sig, fs = torchaudio.load(
            wav,
            num_frames=int(duration * hparams["sample_rate"]),
            frame_offset=int(start * hparams["sample_rate"]),
        )

        # Convert stereo to mono by averaging channels, then squeeze to 1D
        # torchaudio.load returns [channels, samples]; Whisper expects [samples]
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)

        return sig.squeeze(0)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("info_dict")
    @sb.utils.data_pipeline.provides(
        "patient_type", "patient_type_encoded", "dataset", "dataset_ptype"
    )
    def label_pipeline(info_dict):
        """Defines the pipeline to process the patient type labels.
        Note that we have to assign a different integer to each class
        through the label encoder.
        """
        yield info_dict["ptype"]
        patient_type_encoded = label_encoder.encode_label_torch(info_dict["ptype"])
        yield patient_type_encoded
        yield info_dict["dataset"]
        yield f"{info_dict['dataset']}_{info_dict['ptype']}"

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    train_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    out_keys = ["id", "sig", "patient_type_encoded", "info_dict", "dataset", "dataset_ptype"]
    for dataset in train_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=train_info[dataset],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=out_keys,
        )

    hparams["train_dataloader_options"]["sampler"] = BalancingDataSampler(
        dataset=datasets["train"],
        key="dataset_ptype",
        num_samples=hparams["samples_per_epoch"],
        replacement=True,
    )

    return datasets


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    from prepare_nd import prepare_nd

    sb.utils.distributed.run_on_main(
        prepare_nd,
        kwargs={
            "qpn_data_path": hparams["qpn_data_path"],
            "pitt_data_path": hparams["pitt_data_path"],
            "delaware_data_path": hparams["delaware_data_path"],
            "train_annotation": hparams["train_annotation"],
            "test_annotation": hparams["test_annotation"],
            "valid_annotation": hparams["valid_annotation"],
            "chunk_size": hparams["chunk_size"],
        },
    )
    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    nd_brain = NdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    nd_brain.fit(
        nd_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )

    # Run validation and test set to get the predictions
    logger.info("Final validation result:")
    nd_brain.metrics_json = hparams["valid_metrics_json"]
    nd_brain.evaluate(
        test_set=datasets["valid"],
        max_key=hparams["error_metric"],
        test_loader_kwargs=hparams["test_dataloader_options"],
    )

    logger.info("Final test result:")
    nd_brain.metrics_json = hparams["test_metrics_json"]
    nd_brain.evaluate(
        test_set=datasets["test"],
        max_key=hparams["error_metric"],
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
