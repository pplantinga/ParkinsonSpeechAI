# !/usr/bin/python3
"""Recipe for training a detector on the McGill Neuro Parkinson's dataset.
We employ an encoder followed by a classifier.

To run this recipe, use the following command:
> python train_kfold.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    sex_differences_balanced_svp.yaml (for wavlm + ecapa)

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
import pprint
import tempfile
import collections

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb

from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm
import opensmile
from sklearn.model_selection import StratifiedKFold

logger = sb.utils.logger.get_logger("train.py")


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

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

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        # Outputs
        return outputs, lens

    def compute_objectives(self, outputs, batch, stage):
        """Computes the loss using patient-type as label."""

        # Get predictions and labels
        labels, _ = batch.cohort_encoded
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
            # Combine chunks using two strategies
            combined_avg = self.combine_chunks(how="avg")

            # Generate overall metrics, using stored threshold for test set
            avg_threshold = None if stage == sb.Stage.VALID else self.avg_threshold
            metrics_comb_summary, metrics_comb = self.metrics_by_category(
                combined_avg, target_category=None, threshold=avg_threshold
            )
            if stage == sb.Stage.TEST:
                m = metrics_comb["overall"]
                threshold_delta = 0.5 - metrics_comb_summary["overall"]["threshold"]
                self.overall_metrics.append(m.ids, m.scores + threshold_delta, m.labels)
                self.overall_metrics.info_dicts.extend([combined_avg[i] for i in m.ids])

            # Log overall metrics
            chunk_stats = self.summarize_metrics(
                self.error_metrics, self.hparams.threshold
            )
            stage_stats.update({f"chunk_{k}": v for k, v in chunk_stats.items()})
            stage_stats.update(
                {f"comb_avg_{k}": v for k, v in metrics_comb_summary["overall"].items()}
            )

            # Log metrics split by given categories
            for category in self.hparams.metric_categories:
                if category == "reason" and stage != sb.Stage.TEST:
                    continue
                threshold = metrics_comb_summary["overall"]["threshold"]
                cat_metrics_summary, cat_metrics = self.metrics_by_category(
                    combined_scores=combined_avg, target_category=category, threshold=threshold
                )
                logger.info(f"Comb avg breakdown by {category}")
                logger.info(pprint.pformat(cat_metrics_summary, indent=2, compact=True, width=300))

                if stage == sb.Stage.TEST and category == "sex":
                    self.male_metrics.append(cat_metrics["M"].ids, cat_metrics["M"].scores + threshold_delta, cat_metrics["M"].labels)
                    self.female_metrics.append(cat_metrics["F"].ids, cat_metrics["F"].scores + threshold_delta, cat_metrics["F"].labels)



        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            lr = self.lr_scheduler.get_last_lr()

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats,
                max_keys=[self.hparams.error_metric],
                min_keys=["loss"],
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
                max_key=max_key, min_key=min_key
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

            self.lr_scheduler = self.hparams.lr_scheduler(self.optimizer)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Update scheduler if an update was made."""
        if should_step:
            self.lr_scheduler.step()

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

        return breakdown, metrics

    def summarize_metrics(self, metrics, threshold):
        """Simplify metrics to round(100 * (P, R, F1)), x-ent, and threshold"""
        all_metrics = metrics.summarize(threshold=threshold)
        target_metrics = ["precision", "recall", "F-score"]
        summary = {k: round(100 * all_metrics[k], 2) for k in target_metrics}
        summary["threshold"] = round(all_metrics["threshold"], 3)
        cross_ent = binary_cross_entropy(metrics.scores, metrics.labels.float())
        summary["bce"] = round(cross_ent.item(), 3)
        summary["count"] = len(metrics.ids)
        return summary


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'hc': 0, 'pd': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(2)
    label_encoder.enforce_label("PD", 1)
    label_encoder.enforce_label("HC", 0)

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def opensmile_pipeline(wav):
        feats = smile.process_file(wav)

        return torch.tensor(feats.to_numpy())

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig, fs = torchaudio.load(wav)

        return sig.squeeze(0)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("cohort")
    @sb.utils.data_pipeline.provides("cohort_encoded")
    def label_pipeline(cohort):
        """Defines the pipeline to process the patient type labels.
        Note that we have to assign a different integer to each class
        through the label encoder.
        """
        yield label_encoder.encode_label_torch(cohort)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("sex", "age", "updrs")
    @sb.utils.data_pipeline.provides("info_dict")
    def demographics_pipeline(sex, age, updrs):
        """Defines the pipeline to process the patient type labels.
        Note that we have to assign a different integer to each class
        through the label encoder.
        """
        info_dict = {"sex": sex, "age": age, "updrs": updrs}

        # Add a computed category to the info_dict, disease severity
        if info_dict["updrs"] is None:
            info_dict["severity"] = "No Info"
        elif info_dict["updrs"] > 58:
            info_dict["severity"] = "Severe"
        elif info_dict["updrs"] > 32:
            info_dict["severity"] = "Moderate"
        elif info_dict["updrs"] > 0:
            info_dict["severity"] = "Mild"
        else:
            info_dict["severity"] = "No Info"

        # Add a computed category to the info_dict, age category
        if 53 <= info_dict["age"] < 63:
            info_dict["age_range"] = "53-62"
        elif 63 <= info_dict["age"] < 73:
            info_dict["age_range"] = "63-72"
        elif 73 <= info_dict["age"] < 83:
            info_dict["age_range"] = "73-82"
        else:
            info_dict["age_range"] = "unknown"

        yield info_dict

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.

    overall_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["manifest_file"],
        dynamic_items=[audio_pipeline, label_pipeline, demographics_pipeline],
        #dynamic_items=[opensmile_pipeline, label_pipeline],
        output_keys=["id", "sig", "cohort_encoded", "info_dict", "subject_id", "duration"],
    )

    sampler = sb.dataio.sampler.DynamicBatchSampler(overall_dataset, 45, 10)
    hparams["train_dataloader_options"]["sampler"] = sampler


    # Select stratified folds by sex and cohort
    # first, generate a mapping from participant ids to their cohort and sex
    with overall_dataset.output_keys_as(["subject_id", "cohort", "sex"]):
        mapping = {d["subject_id"]: d["cohort"] + d["sex"] for d in overall_dataset}
    ids, stratify_labels = zip(*mapping.items())

    sfk = StratifiedKFold(
        n_splits=hparams["folds"], shuffle=True, random_state=39201777
    )

    # Convert indexes back to subject ids
    folds = [
        ([ids[i] for i in train_set], [ids[i] for i in test_set])
        for train_set, test_set in sfk.split(ids, stratify_labels)
    ]

    return overall_dataset, folds, sampler


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    from qpn_prepare import prepare_qpn

    prepare_qpn(hparams["data_folder"], hparams["manifest_file"])
    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    dataset, folds, sampler = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Go through created folds
    overall_metrics = hparams["error_stats"]()
    overall_metrics.info_dicts = []
    male_metrics = hparams["error_stats"]()
    female_metrics = hparams["error_stats"]()
    for i, (train_ids, test_ids) in enumerate(folds):
        # Re-initialize model by loading hparams again
        hparams_file, run_opts, overrides = sb.parse_arguments(
            sys.argv[1:] + ["--fold", str(i + 1)]
        )
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Brain class initialization
        parkinson_brain = ParkinsonBrain(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=hparams["checkpointer"],
        )
        parkinson_brain.overall_metrics = overall_metrics
        parkinson_brain.male_metrics = male_metrics
        parkinson_brain.female_metrics = female_metrics

        # Select subsets
        train_data = dataset.filtered_sorted(
            key_test={
                "subject_id": lambda x: x in train_ids,
                #"sex": lambda x: x == "F",
            }
        )
        test_data = dataset.filtered_sorted(
            key_test={"subject_id": lambda x: x in test_ids}
        )

        # Training
        parkinson_brain.fit(
            parkinson_brain.hparams.epoch_counter,
            train_set=train_data,
            valid_set=test_data,
            train_loader_kwargs=hparams["train_dataloader_options"],
            valid_loader_kwargs=hparams["valid_dataloader_options"],
        )

        parkinson_brain.evaluate(test_data)

    # Store validation performance
    with open(hparams["overall_metrics_json"], "w") as w:
        w.write("Overall metrics:\n")
        json.dump(overall_metrics.summarize(), w)
        w.write("\nMale metrics:\n")
        json.dump(male_metrics.summarize(), w)
        w.write("\nFemale metrics:\n")
        json.dump(female_metrics.summarize(), w)

    with open(hparams["predictions_json"], "w") as w:
        predictions = {
            overall_metrics.ids[i]: {
                "score": round(float(overall_metrics.scores[i]), 3),
                "label": overall_metrics.labels[i], 
                **overall_metrics.info_dicts[i],
            }
            for i in range(len(overall_metrics.ids))
        }

        json.dump(predictions, w, indent=4)

