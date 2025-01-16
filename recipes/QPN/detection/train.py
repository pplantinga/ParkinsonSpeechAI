# !/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors, ecapa-tdnn) on the McGill Neuro Parkinson's dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_ecapa_tdnn.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hparams/wavlm_ecapa.yaml (for wavlm + ecapa)
    hparams/fbank_ecapa.yaml (for fbank + ecapa)

Author
    * Briac Cordelle 2024
    * Peter Plantinga 2024
"""

import os
import random
import sys
import csv
import json
import logging

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.sampler import BalancingDataSampler

from torch.utils.data import DataLoader
from tqdm import tqdm

logger = sb.utils.logger.get_logger(__name__)


class ParkinsonBrain(sb.core.Brain):
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
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

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
        if self.hparams.loss == "aam":
            # Squeeze and ensure targets are one hot encoded (for AAM)
            preds = outputs.squeeze(1)
            targets = labels.squeeze(1)
            targets = F.one_hot(targets.long(), preds.shape[1]).float()

            # Compute loss with weights
            preds = self.hparams.AAM_loss(preds, targets)

            # Pass through log softmax
            preds = F.log_softmax(preds, dim=1)

            # Pass through KLDiv Loss, apply weight and average
            # KLDLoss = torch.nn.KLDivLoss(reduction="none")
            loss = KLDLoss(preds, targets) * weights
            loss = loss.sum() / targets.sum()

        elif self.hparams.loss == "focal":
            loss = self.hparams.focal_loss(outputs, labels)
        elif self.hparams.loss == "nll":
            preds = self.hparams.log_softmax(outputs)
            loss = self.hparams.nll_loss(preds, labels, weight=self.weight)
        else:
            raise ValueError("Unknown loss specified, expected 'focal', 'aam' or 'nll'")

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            probs = self.hparams.softmax(outputs).squeeze(1)
            self.error_metrics.append(batch.id, probs[:, 1], labels.squeeze(1))
            self.error_metrics.info_dicts.extend(batch.info_dict)

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
            for metric in ["F-score", "precision", "recall"]:
                stage_stats[metric] = 100 * self.error_metrics.summarize(
                    field=metric, threshold=self.hparams.threshold
                )
            combined = self.combine_chunks()

            # Log results split by given categories
            for category in self.hparams.result_categories:
                result_breakdown = self.results_by_category(combined, category)
                logger.info(f"Breakdown by {category}:")
                logger.info(json.dumps(result_breakdown, indent=2))

            # Dump results to file only on test
            if stage == sb.Stage.TEST:
                with open(self.results_json, "w") as f:
                    json.dump(combined, f)
                logger.info(f"Results stored {self.results_json}")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats,
                max_keys=[self.hparams.error_metric],
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def combine_chunks(self):
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
                    **info_dict
                }
            else:
                combined_scores[utt_id]["scores"].append(round(score.item(), 3))

        # For now just take the average. Perhaps do something fancier later
        for utt_id in combined_scores:
            scores = combined_scores[utt_id]["scores"]
            combined_scores[utt_id]["combined"] = round(sum(scores) / len(scores), 3)

        return combined_scores

    def results_by_category(self, results, target_category):
        """Divides results by a given category."""

        # Separate the scores into individual metrics objects
        options = set(t[target_category] for t in results.values())
        metrics = {option: self.hparams.error_stats() for option in options}
        for utt_id, categories in results.items():
            option = categories[target_category]
            metrics[option].ids.append(utt_id)
            metrics[option].scores.append(torch.tensor(categories["combined"]))
            metrics[option].labels.append(torch.tensor(categories["label"]))

        # Iterate all options for this category and summarize metrics
        def summarize(metric, option):
            threshold = self.hparams.threshold
            score = metrics[option].summarize(metric, threshold=threshold)
            return round(100 * score, 2)

        breakdown = {}
        for option in options:
            breakdown[option] = {
                metric: summarize(metric, option)
                for metric in ["F-score", "precision", "recall"]
            }

        return breakdown


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'hc': 0, 'pd': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(hparams["out_neurons"])
    label_encoder.enforce_label("PD", 1)
    label_encoder.enforce_label("HC", 0)

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "duration", "start")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, duration, start):
        sig, fs = torchaudio.load(
            wav,
            num_frames=int(duration * hparams["sample_rate"]),
            frame_offset=int(start * hparams["sample_rate"]),
        )

        return sig.squeeze(0)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("info_dict")
    @sb.utils.data_pipeline.provides(
        "patient_type", "patient_type_encoded", "ptype_sex"
    )
    def label_pipeline(info_dict):
        """Defines the pipeline to process the patient type labels.
        Note that we have to assign a different integer to each class
        through the label encoder.
        """
        yield info_dict["ptype"]
        patient_type_encoded = label_encoder.encode_label_torch(info_dict["ptype"])
        yield patient_type_encoded

        # For re-balancing, let's use both patient type and patient sex
        yield info_dict["ptype"] + "_" + info_dict["sex"]

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    train_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    for dataset in train_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=train_info[dataset],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "patient_type_encoded", "info_dict", "ptype_sex"],
        )
        repeat_filtered = datasets[dataset].filtered_sorted(key_test={"info_dict": lambda x: x["task"] == "repeat"})
        print(f"Found {len(repeat_filtered)} 'repeat' samples in {dataset} set")

    # Define sampler based on sex and patient type, shuffle must be None
    # Unfortunately I think we need replacement here since HC_M is so small
    hparams["train_dataloader_options"]["sampler"] = BalancingDataSampler(
        dataset=datasets["train"],
        key="ptype_sex",
        num_samples=hparams["samples_per_epoch"],
        replacement=True,
    )

    # Remove keys from training data for e.g. training only on men
    for key, value in hparams["remove_keys"]:
        datasets["train"] = datasets["train"].filtered_sorted(
            key_test=lambda x: x["info_dict"][key] != value,
        )

    return datasets


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset prep (parsing KCL MDVR and annotation into json files)
    from prepare_neuro import prepare_neuro

    sb.utils.distributed.run_on_main(
        prepare_neuro,
        kwargs={
            "data_folder": hparams["data_folder"],
            "train_annotation": hparams["train_annotation"],
            "test_annotation": hparams["test_annotation"],
            "valid_annotation": hparams["valid_annotation"],
            "chunk_size": hparams["chunk_size"],
        },
    )
    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])
    sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    parkinson_brain = ParkinsonBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    parkinson_brain.weight = torch.tensor(
        [[hparams["weight_hc"], hparams["weight_pd"]]],
        device=parkinson_brain.device,
        dtype=torch.float32,
    )

    # Training
    parkinson_brain.fit(
        parkinson_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )

    # Run validation and test set to get the predictions
    logger.info("Final validation result:")
    parkinson_brain.results_json = hparams["valid_results_json"]
    parkinson_brain.evaluate(
        test_set=datasets["valid"],
        max_key=hparams["error_metric"],
        test_loader_kwargs=hparams["test_dataloader_options"],
    )

    logger.info("Final test result:")
    parkinson_brain.results_json = hparams["test_results_json"]
    parkinson_brain.evaluate(
        test_set=datasets["test"],
        max_key=hparams["error_metric"], 
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
