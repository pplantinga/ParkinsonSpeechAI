# !/usr/bin/python3
"""Recipe for detection of dementia and related diseases.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hparams/wavlm_ecapa.yaml (for wavlm + ecapa)

Author
    * Peter Plantinga 2024
"""

import csv
import sys
import tqdm

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main

class DetectBrain(sb.core.Brain):
    """Class for training detector"""

    def compute_forward(self, batch, stage):
        """
        Computation pipeline based on a encoder + speaker classifier for detection.
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

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using patient-type as label."""

        # Get predictions and labels
        labels, _ = batch.diagnosis
        predictions, lens = predictions

        # Concatenate labels in case of wav_augment
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            patient_type = self.hparams.wav_augment.replicate_labels(labels)

        # Compute loss with weights
        predictions = self.hparams.log_softmax(predictions)
        loss = self.hparams.compute_cost(
            predictions, labels, lens, weight=self.weights
        )

        if stage == sb.Stage.TRAIN and hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        elif stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_loss},
                name=f"epoch_{epoch}_loss_{stage_loss:.2f}",
                min_keys=["loss"],
            )

    @torch.no_grad()
    def predict_test(self, test_set):
        self.modules.eval()
        test_set = test_set.filtered_sorted(sort_key="id")
        test_dl = sb.dataio.dataloader.make_dataloader(test_set)

        self.checkpointer.recover_if_possible(min_key="loss")
        with open(self.hparams.test_predictions_file, "w", newline="") as f:
            fields = ["uid", "diagnosis_control", "diagnosis_mci", "diagnosis_adrd"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for test_example in tqdm.tqdm(test_dl):
                predictions, lens = self.compute_forward(test_example, sb.Stage.TEST)
                predictions = torch.softmax(predictions.squeeze(), dim=-1)
                w.writerow({
                    "uid": test_example.id[0],
                    "diagnosis_control": predictions[0].cpu().numpy(),
                    "diagnosis_mci": predictions[1].cpu().numpy(),
                    "diagnosis_adrd": predictions[2].cpu().numpy(),
                })


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'hc': 0, 'pd': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Length of a chunk
    sentence_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("filepath")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(filepath):
        return sb.dataio.dataio.read_audio(filepath)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("diagnosis_control", "diagnosis_mci", "diagnosis_adrd")
    @sb.utils.data_pipeline.provides("diagnosis")
    def label_pipeline(diagnosis_control, diagnosis_mci, diagnosis_adrd):
        return torch.LongTensor([diagnosis_control, diagnosis_mci, diagnosis_adrd])

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset, json_path in hparams["manifests"].items():
        items = [audio_pipeline]
        keys = ["id", "sig"]

        # Add label for train/valid where we have them
        if dataset != "test":
            items.append(label_pipeline)
            keys.append("diagnosis")

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path, dynamic_items=items, output_keys=keys,
        )

    return datasets

if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset prep (parsing KCL MDVR and annotation into json files)
    from prepare_prepare import prepare_prepare

    run_on_main(
        prepare_prepare,
        kwargs={
            "data_folder": hparams["data_folder"],
            "manifests": hparams["manifests"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets = dataio_prep(hparams)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    detector = DetectBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Add weights for training balance
    detector.weights = torch.tensor(
        hparams["weights"], device=detector.device, requires_grad=False
    )

    # Training
    detector.fit(
        detector.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # TODO: Print estimates for TEST
    detector.predict_test(datasets["test"])
