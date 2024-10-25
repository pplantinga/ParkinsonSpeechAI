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
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.core import AMPConfig

from torch.utils.data import DataLoader
from tqdm import tqdm


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

        # Pass through log softmax
        outputs = self.hparams.log_softmax(outputs)

        # Outputs
        return outputs, lens

    def compute_objectives(self, predictions, batch, stage, labels=None):
        """Computes the loss using patient-type as label."""

        # Get predictions and labels
        if labels is None:
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

    def write_to_logs(self, line):
        self.hparams.train_logger.log_stats(
            {line}
        )

    def custom_evaluate(self, test_set, max_key=None, min_key=None, progressbar=None, test_loader_kwargs={}):
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (isinstance(test_set, DataLoader) or isinstance(test_set, LoopedLoader)):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(test_set, sb.Stage.TEST, **test_loader_kwargs)

        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)

        # Create a dictionary for prediction stats
        categories = {"PD": 0, "HC": 0, "M": 0, "F": 0, "English": 0, "French": 0, "Other": 0, ">80": 0,
                      "71-80": 0, "61-70": 0, "51-60": 0, "<50": 0, "repeat": 0, "vowel_repeat": 0,
                      "recall": 0, "read_text": 0, "dpt": 0, "hbd": 0, "unk": 0}
        # Add keys to breakdown between PD and HC
        keys = list(categories.keys())
        for key in keys:
            pd_key = "PD_" + key
            hc_key = "HC_" + key
            categories[pd_key] = 0
            categories[hc_key] = 0

        # Create prediction stats dict with copies of categories dict to track total count + right count
        prediction_stats = {"right_count": categories.copy(), "total_count": categories.copy()}

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
                loss, prediction_stats = self.custom_evaluate_batch(batch, stage=sb.Stage.TEST,
                                                                    prediction_stats=prediction_stats)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.write_test_stats(prediction_stats)
            self.on_stage_end(sb.Stage.TEST, avg_test_loss, None)
        self.step = 0
        return avg_test_loss

    @torch.no_grad()
    def custom_evaluate_batch(self, batch, stage, prediction_stats):
        losses = []
        amp = AMPConfig.from_name(self.eval_precision)
        if self.use_amp:
            with torch.autocast(
                    dtype=amp.dtype, device_type=torch.device(self.device).type,
            ):
                losses, prediction_stats = self.custom_loss_compute(batch, stage, prediction_stats)
        else:
            losses, prediction_stats = self.custom_loss_compute(batch, stage, prediction_stats)

        loss = torch.stack(losses).mean()
        return loss.detach().cpu(), prediction_stats

    def custom_loss_compute(self, batch, stage, prediction_stats):
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

            # Compute forward for this recording's chunks
            out = self.compute_forward(batch, stage, chunks, chunk_lens)
            predictions, chunk_lens = out

            # Now get the labels and create a label tensor (similarly to lens tensor)
            patient_type_encoded, _ = batch.patient_type_encoded
            patient_type_encoded = torch.full((chunks.size(0),), patient_type_encoded[i, 0].item()).to(self.device)
            patient_type_encoded = patient_type_encoded.unsqueeze(1)

            # Get the info_dict
            info_dict = batch.info_dict

            # Update prediction statistics
            prediction_stats = self.update_inference_stats(prediction_stats, predictions,
                                                           patient_type_encoded[0], info_dict[i])

            # Compute loss for this recording's chunks and add to losses
            losses.append(self.compute_objectives(out, batch, stage, patient_type_encoded))

        return losses, prediction_stats  # TODO make it possible to have batch_size > 1

    def update_inference_stats(self, prediction_stats, predictions, label, info_dict):
        max_values, max_indices = torch.max(predictions, dim=-1)
        # Iterate through labels, determine whether the model was right
        # or wrong and add information to right/wrong predictions
        for i in range(max_indices.size(0)):
            correct = True if label == max_indices[i] else False
            ptype = info_dict["ptype"]

            age_bin = next((k for k, v in [
                (">80", info_dict["age"] > 80),
                ("71-80", info_dict["age"] > 70),
                ("61-70", info_dict["age"] > 60),
                ("51-60", info_dict["age"] > 50),
                ("<50", True)
            ] if v), "<50")

            if correct:
                prediction_stats["right_count"][age_bin] += 1
                prediction_stats["right_count"][ptype + "_" + age_bin] += 1
            prediction_stats["total_count"][age_bin] += 1
            prediction_stats["total_count"][ptype + "_" + age_bin] += 1

            # For the rest we can use the value in info_dict as a key
            for key in info_dict.keys():
                # Skip ages and updrs (for now)
                if key == "age" or key == "updrs":
                    continue
                # Update correct dict and count dict
                if correct:
                    prediction_stats["right_count"][info_dict[key]] += 1
                    prediction_stats["right_count"][ptype + "_" + info_dict[key]] += 1

                prediction_stats["total_count"][info_dict[key]] += 1
                prediction_stats["total_count"][ptype + "_" + info_dict[key]] += 1

        return prediction_stats

    def write_test_stats(self, prediction_stats):
        # Add percentages for stats
        keys_list = list(prediction_stats["right_count"].keys())

        for key in keys_list:
            # Avoid potential division by 0
            if prediction_stats["total_count"][key] == 0:
                prediction_stats["total_count"][key] = 1

            # Get percentage value
            percentage_key = "{}%".format(key)
            prediction_stats["right_count"][percentage_key] = prediction_stats["right_count"][key] \
                                                              / prediction_stats["total_count"][key]

        # Create file names
        right_stats_filepath = os.path.join(self.hparams.output_folder, "predictions.csv")

        # Write stats to file
        with open(right_stats_filepath, "w") as f:
            writer = csv.writer(f)
            for key in prediction_stats.keys():
                writer.writerow(prediction_stats[key].keys())
                writer.writerow(prediction_stats[key].values())

def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'hc': 0, 'pd': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

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

    # Dataset prep
    from prepare_neuro import prepare_neuro

    run_on_main(
        prepare_neuro,
        kwargs={
            "data_folder": hparams["data_folder"],
            "train_annotation": hparams["train_annotation"],
            "test_annotation": hparams["test_annotation"],
            "valid_annotation": hparams["valid_annotation"],
            "remove_repeats": hparams["remove_repeats"],
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
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )

    # Chunk Testing
    chunk_test_stats = parkinson_brain.custom_evaluate(
        test_set=datasets["chunk_test"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
