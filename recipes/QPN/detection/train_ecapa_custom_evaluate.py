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

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main, main_process_only
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.core import AMPConfig
from speechbrain.processing.voice_analysis import vocal_characteristics, compute_gne

from torch.utils.data import DataLoader
from tqdm import tqdm

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

        if self.hparams.vocal_features == 5:
            voice_feats = vocal_characteristics(wavs)
            #gne = compute_gne(wavs)
            added_feats = torch.stack(voice_feats, dim=-1)
            feats = torch.cat((feats, added_feats), dim=-1)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        # Pass through log softmax
        outputs = self.hparams.log_softmax(outputs)

        # Outputs
        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using patient-type as label."""

        # Get predictions and labels
        labels, _ = batch.patient_type_encoded
        predictions, lens = predictions

        # Concatenate labels in case of wav_augment
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            patient_type = self.hparams.wav_augment.replicate_labels(labels)

        # Normalize weights
        max_weight = max(self.hparams.weight_pd, self.hparams.weight_hc)
        weight_pd = self.hparams.weight_pd / max_weight
        weight_hc = self.hparams.weight_hc / max_weight

        # Compute loss with weights
        weights = torch.tensor([weight_pd, weight_hc]).to(self.device)
        loss = self.hparams.compute_cost(predictions, labels, lens, weight=weights)

        if stage == sb.Stage.TRAIN and hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, labels, lens)

        if stage == sb.Stage.TEST:
            self.write_stats(predictions, batch)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

        if stage == sb.Stage.TEST:
            self.metrics_file = open(self.hparams.metrics_file, delimiter="")
            header = [
                "id",
                "score_pos",
                "score_neg",
                "label",
                "patient_type",
                "patient_gender",
                "patient_age",
                "patient_l1",
                "test_type",
            ]
            self.metrics_csv = csv.DictWriter(self.metrics_file, header)
            self.metrics_csv.writeheader()


    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

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
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            self.metrics_file.close()

    @main_process_only
    def write_stats(predictions, batch):
        for i in range(len(batch)):
            row = {
                "id": batch.id[i],
                "score_pos": predictions[i][0],
                "score_neg": predictions[i][1],
                "label": labels[i],
                "patient_type": batch.patient_type[i],
                "patient_gender": batch.patient_gender[i],
                "patient_age": batch.patient_age[i],
                "patient_l1": batch.patient_l1[i],
                "test_type": batch.test_type[i],
            }
            self.metrics_csv.writerow(row)

def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'hc': 0, 'pd': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(2)

    # Length of a chunk
    sentence_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "duration", "start")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, duration, start):
        if duration < hparams["sentence_len"]:
            sig, fs = torchaudio.load(wav)
        else:
            start_sample = int(start * hparams["sample_rate"])
            sig, fs = torchaudio.load(
                wav, num_frames=sentence_len_sample, frame_offset=start_sample
            )

        return sig.squeeze(0)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("patient_type")
    @sb.utils.data_pipeline.provides("patient_type", "patient_type_encoded")
    def label_pipeline(patient_type):
        """Defines the pipeline to process the patient type labels.
        Note that we have to assign a different integer to each class
        through the label encoder.
        """
        yield patient_type
        patient_type_encoded = label_encoder.encode_label_torch(patient_type)
        yield patient_type_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    train_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "normal_test_fr": hparams["test_fr_annotation"],
        "normal_test_en": hparams["test_en_annotation"],
    }

    hparams["dataloader_options"]["shuffle"] = True
    for dataset in train_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=train_info[dataset],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=[
                "id",
                "sig",
                "patient_type_encoded",
                "patient_type",
                "patient_gender",
                "patient_age",
                "patient_l1",
                "test_type",
            ],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="patient_type",
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

    run_on_main(
        prepare_neuro,
        kwargs={
            "data_folder": hparams["data_folder"],
            "train_annotation": hparams["train_annotation"],
            "test_annotation_fr": hparams["test_fr_annotation"],
            "test_annotation_en": hparams["test_en_annotation"],
            "valid_annotation": hparams["valid_annotation"],
            "keep_short_recordings": hparams["short_recordings"],
            "chunk_length": hparams["sentence_len"],
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
    parkinson_brain = ParkinsonBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    parkinson_brain.fit(
        parkinson_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Regular Testing FR
    #parkinson_brain.write_to_logs("Testing on French test set")
    regular_test_stats_fr = parkinson_brain.evaluate(
        test_set=datasets["normal_test_fr"],
        test_loader_kwargs=hparams["dataloader_options"],
    )

    # Regular Testing EN
    #parkinson_brain.write_to_logs("Testing on English test set")
    regular_test_stats_en = parkinson_brain.evaluate(
        test_set=datasets["normal_test_en"],
        test_loader_kwargs=hparams["dataloader_options"],
    )
