# !/usr/bin/python3
"""Recipe for optimizing a detector on DementiaBank's Pitt Corpus, an Alzheimers dataset.
We employ an encoder followed by a classifier.

To run this recipe, use the following command:
> python optimize_hparams.py {hyperparameter_file} {overrides}

Hyperparameters that are being tuned are defined in the objective function.

Author
    * Briac Cordelle 2026
"""

import sys
import json

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
import optuna

from torch.nn.functional import binary_cross_entropy

logger = sb.utils.logger.get_logger("train.py")

class AlzheimerBrain(sb.core.Brain):
    """Class for speaker embedding training"""
    def compute_forward(self, batch, stage):
        """
        Computation pipeline based on a encoder + speaker classifier for Alzheimer's detection.
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
        else:
            probs = torch.sigmoid(outputs.view(-1))
            self.error_metrics.append(batch.id, probs, labels.view(-1))

            # Use unweighted, unsmoothed score for comparable results across hparams
            loss = binary_cross_entropy(probs, labels.view(-1).float())

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            # Combine chunks using two strategies
            combined_avg = self.combine_chunks(how="avg")

            # Generate overall metrics
            metrics_comb_avg = self.metrics_by_category(
                combined_avg, target_category=None
            )

            # Log overall metrics
            chunk_stats = self.summarize_metrics(
                self.error_metrics, self.hparams.threshold
            )
            stage_stats.update({f"chunk_{k}": v for k, v in chunk_stats.items()})
            stage_stats.update(
                {f"comb_avg_{k}": v for k, v in metrics_comb_avg["overall"].items()}
            )

            # Dump metrics to file only on test
            if stage == sb.Stage.TEST:
                with open(self.metrics_json, "w") as f:
                    json.dump(combined_avg, f)
                    f.write("\nChunk stats: ")
                    json.dump(chunk_stats, f)
                    f.write("\nCombined stats: ")
                    json.dump(metrics_comb_avg, f)
                logger.info(f"Results stored {self.metrics_json}")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            lr = self.lr_scheduler.get_last_lr()

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.last_valid_stats = stage_stats

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

        combined_scores = {}
        for i, score, label in zip(ids, scores, labels):
            utt_id, chunk = i.rsplit("_", 1)

            if utt_id not in combined_scores:
                combined_scores[utt_id] = {
                    "scores": [round(score.item(), 3)],
                    "label": label.item(),
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
        """Simplify metrics to round(100 * (P, R, F1)), x-ent, and threshold"""
        all_metrics = metrics.summarize(threshold=threshold)
        target_metrics = ["precision", "recall", "F-score"]
        summary = {k: round(100 * all_metrics[k], 2) for k in target_metrics}
        summary["threshold"] = round(all_metrics["threshold"], 3)
        cross_ent = binary_cross_entropy(metrics.scores, metrics.labels.float())
        summary["bce"] = round(cross_ent.item(), 3)
        return summary


def train_and_evaluate(hparams, run_opts, hparams_file, overrides):
    """Train the model and evaluate on validation set, return F-score"""
    # Dataset prep
    from prepare_pitt import prepare_pitt

    sb.utils.distributed.run_on_main(
        prepare_pitt,
        kwargs={
            "data_folder": hparams["data_folder"],
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
    alzheimer_brain = AlzheimerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
    )

    alzheimer_brain.last_valid_stats = None

    # Training
    alzheimer_brain.fit(
        alzheimer_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )

    return alzheimer_brain.last_valid_stats["chunk_F-score"]

def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'hc': 0, 'ad': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(2)
    label_encoder.enforce_label("AD", 1)
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
    @sb.utils.data_pipeline.takes("ptype")
    @sb.utils.data_pipeline.provides("patient_type_encoded")
    def label_pipeline(ptype):
        """Defines the pipeline to process the diagnosis labels.
        Note that we have to assign a different integer to each class
        through the label encoder.
        """
        patient_type_encoded = label_encoder.encode_label_torch(ptype)
        return patient_type_encoded

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
            output_keys=["id", "sig", "patient_type_encoded"],
        )

    return datasets

if __name__ == "__main__":

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    print("Starting hyperparameter optimization with Optuna...")

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    def objective(trial):
        hparams['epochs'] = trial.suggest_int('epochs', 15, 50, step=1)
        hparams['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        hparams['base_lr'] = trial.suggest_float('base_lr', 1e-7, 1e-4, log=True)
        hparams['chunk_size'] = trial.suggest_int('chunk_size', 15, 60, step=1)
        hparams['embedding_size'] = trial.suggest_int('embedding_size', 512, 1280, step=32)
        hparams['dropout'] = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        hparams['snr_low'] = trial.suggest_float('snr_low', 0.0, 15.0, step=1.0)
        hparams['snr_delta'] = trial.suggest_float('snr_delta', 5.0, 20.0, step=1.0)
        hparams['drop_freq_low'] = trial.suggest_float('drop_freq_low', 0.0, 0.3, step=0.01)
        hparams['drop_freq_high'] = trial.suggest_float('drop_freq_high', 0.7, 1.0, step=0.01)
        hparams['drop_freq_count_low'] = trial.suggest_categorical('drop_freq_count_low', [1, 2, 3])
        hparams['drop_freq_count_delta'] = trial.suggest_categorical('drop_freq_count_delta', [0, 1, 2, 3, 4, 5, 6])
        hparams['drop_freq_width'] = trial.suggest_float('drop_freq_width', 0.01, 0.15, step=0.01)
        hparams['min_augmentations'] = trial.suggest_categorical('min_augmentations', [0, 1, 2])
        hparams['augment_prob'] = trial.suggest_float('augment_prob', 0.5, 1.0, step=0.1)
        hparams['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

        sb.utils.distributed.ddp_init_group(run_opts)

        score = train_and_evaluate(hparams, run_opts, hparams_file, overrides)
        return score

    study = optuna.create_study(
        storage="sqlite:///optuna_study.db",
        study_name="pitt_tuning",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=hparams['num_trials'])

    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
