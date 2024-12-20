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
"""

import os
import random
import sys
import csv

import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.nn.functional as F

from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

import speechbrain as sb
from speechbrain.dataio.encoder import CategoricalEncoder
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.distributed import run_on_main, main_process_only, if_main_process


class ParkinsonBrain(sb.core.Brain):
    """Class for speaker embedding training"""

    def compute_forward(self, batch, stage, wavs=None, lens=None):
        """
        Computation pipeline based on a encoder + speaker classifier for parkinson's detection.
        Data augmentation and environmental corruption are applied to the
        input speech if present.
        """

        # Get wavs + lens
        if lens is None:
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

    def compute_objectives(self, outputs, batch, stage, labels=None):
        """Computes the loss using patient-type as label."""

        # Get predictions and labels
        if labels is None:
            labels, _ = batch.patient_type_encoded
        preds, lens = outputs

        # Concatenate labels in case of wav_augment
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            patient_type = self.hparams.wav_augment.replicate_labels(labels)

        # Normalize and tensorize weights
        max_weight = max(self.hparams.weight_pd, self.hparams.weight_hc)
        weight_pd = self.hparams.weight_pd / max_weight
        weight_hc = self.hparams.weight_hc / max_weight
        weights = torch.tensor([weight_pd, weight_hc]).unsqueeze(0).to(self.device)

        # Compute loss
        if self.hparams.loss == "aam":
            # Squeeze and ensure targets are one hot encoded (for AAM)
            preds = preds.squeeze(1)
            targets = labels.squeeze(1)
            targets = F.one_hot(targets.long(), preds.shape[1]).float()

            # Compute loss with weights
            preds = self.hparams.AAM_loss(preds, targets)

            # Pass through log softmax
            preds = F.log_softmax(preds, dim=1)

            # Pass through KLDiv Loss, apply weight and average
            KLDLoss = torch.nn.KLDivLoss(reduction="none")
            loss = KLDLoss(preds, targets) * weights
            loss = loss.sum() / targets.sum()

        elif self.hparams.loss == "focal":
            loss = self.hparams.focal_loss(preds, labels, weights)
        else:
            print("Unknown loss specified, please specify either focal or AAM loss")

        if stage == sb.Stage.TRAIN and hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            outputs, _ = outputs
            outputs = outputs.squeeze(1)
            outputs = (outputs[:, 0] - outputs[:, 1] + 2) / 4
            self.error_metrics.append(batch.id, outputs, labels.squeeze(1))

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
            stage_stats["ErrorRates"] = self.error_metrics.summarize(threshold=0.5)

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
                meta={self.hparams.error_metric: stage_stats["ErrorRates"][self.hparams.error_metric]},
                min_keys=[self.hparams.error_metric],
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


    def custom_evaluate(self, test_set, max_key=None, min_key=None, progressbar=None, test_loader_kwargs={}):
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (isinstance(test_set, DataLoader) or isinstance(test_set, LoopedLoader)):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(test_set, sb.Stage.TEST, **test_loader_kwargs)

        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)

        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                    test_set,
                    dynamic_ncols=True,
                    disable=not progressbar,
                    colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss = self.custom_loss_compute(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.on_stage_end(sb.Stage.TEST, avg_test_loss, None)
        self.step = 0
        return avg_test_loss


    def custom_loss_compute(self, batch, stage):
        losses = []
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Since each batch tensor will have a tensor of shape
        # [batch, chunks, audio info], we loop through the batch
        # dimension to get chunks (and pass this through our model)
        for i in range(wavs.size(0)):
            # Get the chunks and create a lens tensor (all the same len)
            chunks = wavs[i]
            chunk_lens = torch.full((chunks.size(0),), lens[i]).to(self.device)

            # Compute forward, extract score and index
            out = self.compute_forward(batch, stage, chunks, chunk_lens)
            predictions, lens = out

            # Get the labels and create a label tensor (similarly to lens tensor)
            patient_type_encoded, _ = batch.patient_type_encoded
            patient_type_encoded = torch.full((chunks.size(0),), patient_type_encoded[i, 0].item()).to(self.device)
            patient_type_encoded = patient_type_encoded.unsqueeze(1)

            # Get the info_dict
            info_dict = batch.info_dict

            # Get the keys and add other headers
            info = info_dict[0]
            info["cosine similarities"] = predictions.squeeze(1).tolist()
            info["label"] = torch.mean(patient_type_encoded.squeeze(0), dtype=torch.float32).item()

            # Write stats of this recording to predictions
            if if_main_process():
                self.write_prediction(info)

            # Compute loss for this recording's chunks and add to losses
            losses.append(self.compute_objectives(out, batch, stage, patient_type_encoded))

        # Compute and return loss
        loss = torch.stack(losses).mean()
        return loss.detach().cpu() # TODO make it possible to have batch_size > 1, (is this useful though)?


    def write_prediction(self, info):
        with open(self.hparams.predictions_file, "a") as f:
            writer = csv.writer(f)
            if self.step == 1:
                writer.writerow(list(info.keys()))
            else:
                writer.writerow(list(info.values()))

def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'hc': 0, 'pd': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(hparams["out_neurons"])

    # Length of a chunk
    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "duration", "info_dict")
    @sb.utils.data_pipeline.provides("sig", "info_dict")
    def audio_pipeline(wav, duration, info_dict):
        if duration < hparams["sentence_len"]:
            sig, fs = torchaudio.load(wav)
        else:
            duration_sample = int(duration * hparams["sample_rate"])
            start = random.randint(0, duration_sample - snt_len_sample)
            sig, fs = torchaudio.load(wav, num_frames=snt_len_sample, frame_offset=start)

        sig = sig.transpose(0, 1).squeeze(1)
        return sig, info_dict

    # Define test pipeline:
    @sb.utils.data_pipeline.takes("wav", "duration", "info_dict")
    @sb.utils.data_pipeline.provides("sig", "info_dict")
    def test_pipeline(wav, duration, info_dict):
        # Get duration of sample
        duration_sample = int(duration * hparams["sample_rate"])

        # Initialize an empty list to store the chunks
        chunks = []

        # Determine the number of chunks
        num_chunks = (duration_sample // (snt_len_sample // 2)) + 1

        # Case for short recordings
        if duration_sample <= snt_len_sample:
            sig, fs = torchaudio.load(wav)
            return sig, info_dict

        # Max chunks
        if num_chunks > hparams["max_test_chunks"]:
            num_chunks = hparams["max_test_chunks"]

        # Iterate over the chunks
        for i in range(num_chunks):
            start = i * (snt_len_sample // 2)
            stop = start + snt_len_sample

            # Ensure the last chunk doesn't go beyond the end of the WAV file
            if stop > duration_sample:
                # If the last chunk goes beyond end of wav by less than half snt_len_sample,
                # then ignore this chunk (it will have too much overlap with previous chunk)
                if stop - duration_sample < snt_len_sample // 2:
                    continue

                stop = duration_sample
                start = stop - snt_len_sample
                if start < 0:
                    start = 0

            num_frames = stop - start
            sig, fs = torchaudio.load(
                wav, num_frames=num_frames, frame_offset=start
            )

            # Transpose the signal to have shape [wav, chunks]
            sig = sig.transpose(0, 1).squeeze(1)
            chunks.append(sig)

        # Stack the chunks into a tensor of shape [chunks, wav] so output will be [batch, chunks, wav]
        output = torch.stack(chunks, dim=0)

        return output, info_dict

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("info_dict")
    @sb.utils.data_pipeline.provides("patient_type", "patient_type_encoded")
    def label_pipeline(info_dict):
        """Defines the pipeline to process the patient type labels.
        Note that we have to assign a different integer to each class
        through the label encoder.
        """
        yield info_dict["ptype"]
        patient_type_encoded = label_encoder.encode_label_torch(info_dict["ptype"])
        yield patient_type_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    train_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    hparams["dataloader_options"]["shuffle"] = True
    for dataset in train_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=train_info[dataset],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "patient_type_encoded", "info_dict"],
        )

    datasets["chunk_test"] = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        dynamic_items=[test_pipeline, label_pipeline],
        output_keys=["id", "sig", "patient_type_encoded", "info_dict"],
    )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    label_encoder.load_or_create(
        path=hparams["encoded_labels"],
        from_didatasets=[datasets["train"]],
        output_key="patient_type",
    )
    label_encoder.enforce_label("PD", 1)
    label_encoder.enforce_label("HC", 0)
    label_encoder.save(hparams["encoded_labels"])

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

    # Dataset prep
    from prepare_neuro import prepare_neuro

    run_on_main(
        prepare_neuro,
        kwargs={
            "data_folder": hparams["data_folder"],
            "train_annotation": hparams["train_annotation"],
            "test_annotation": hparams["test_annotation"],
            "valid_annotation": hparams["valid_annotation"],
            "remove_keys": hparams["remove_keys"],
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

    # Regular Testing
    regular_test_stats = parkinson_brain.evaluate(
        test_set=datasets["test"],
        min_key=hparams["error_metric"],
        test_loader_kwargs=hparams["dataloader_options"],
    )

    # Chunk Testing
    chunk_test_stats = parkinson_brain.custom_evaluate(
        test_set=datasets["chunk_test"],
        min_key=hparams["error_metric"],
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
