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
        voice_feats, voice_lens = batch.feats
        feats = self.compute_features(wavs, lens, voice_feats)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        # Pass through log softmax
        outputs = self.hparams.log_softmax(outputs)

        # Outputs
        return outputs, lens

    def compute_features(self, wavs, lens, voice_feats):
        feats = self.modules.compute_features(wavs)

        if self.hparams.add_vocal_features:
            min_len = min(feats.size(1), voice_feats.size(1))
            feats = feats[:, :min_len]
            voice_feats = voice_feats[:, :min_len]
            feats = torch.cat((feats, voice_feats), dim=-1)
            feats = self.modules.mean_var_norm(feats, lens)

        return feats

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
        self.hparams.train_logger.log_stats({"dummykey": line})

    def custom_evaluate(self, test_set, max_key=None, min_key=None, progressbar=None, test_loader_kwargs={}, language=None):
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
        prediction_stats = {"right": categories.copy(), "wrong": categories.copy(), "count": categories.copy()}

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
                loss, prediction_stats = self.custom_evaluate_batch(batch, stage=sb.Stage.TEST, prediction_stats=prediction_stats)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.write_test_stats(prediction_stats, language)
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

        return losses, prediction_stats #TODO make it possible to have batch_size > 1

    def update_inference_stats(self, prediction_stats, predictions, label, info_dict):
        max_values, max_indices = torch.max(predictions, dim=-1)
        # Iterate through labels, determine whether the model was right
        # or wrong and add information to right/wrong predictions
        for i in range(max_indices.size(0)):
            if label == max_indices[i]:
                correct = "right"
            else:
                correct = "wrong"

            if info_dict["patient_age"] > 80:
                prediction_stats[correct][">80"] += 1
                prediction_stats["count"][">80"] += 1
            elif info_dict["patient_age"] > 70:
                prediction_stats[correct]["71-80"] += 1
                prediction_stats["count"]["71-80"] += 1
            elif info_dict["patient_age"] > 60:
                prediction_stats[correct]["61-70"] += 1
                prediction_stats["count"]["61-70"] += 1
            elif info_dict["patient_age"] > 50:
                prediction_stats[correct]["51-60"] += 1
                prediction_stats["count"]["51-60"] += 1
            else:
                prediction_stats[correct]["<50"] += 1
                prediction_stats["count"]["<50"] += 1

            # For the rest we can use the value in info_dict as a key
            for key in info_dict.keys():
                # Skip ages
                if key == "patient_age":
                    continue

                # Update correct dict and count dict
                prediction_stats[correct][info_dict[key]] += 1
                prediction_stats["count"][info_dict[key]] += 1

        return prediction_stats

    def write_test_stats(self, prediction_stats, language):
        # Add percentages for stats
        keys_list = list(prediction_stats["right"].keys())

        for key in keys_list:
            # Avoid potential division by 0
            if prediction_stats["count"][key] == 0:
                 prediction_stats["count"][key] = 1

            percentage_key = "{}%".format(key)
            prediction_stats["wrong"][percentage_key] = prediction_stats["wrong"][key] / prediction_stats["count"][key]

        for key in keys_list:
            percentage_key = "{}%".format(key)
            prediction_stats["right"][percentage_key] = prediction_stats["right"][key] / prediction_stats["count"][key]

        # Create file names
        wrong_stats_filepath = os.path.join(self.hparams.output_folder, f"{language}_wrong_predictions.csv")
        right_stats_filepath = os.path.join(self.hparams.output_folder, f"{language}_right_predictions.csv")

        # Write stats to file
        with open(wrong_stats_filepath, "w") as f:
            writer = csv.writer(f)
            for key, value in prediction_stats["wrong"].items():
                writer.writerow([key, value])

        with open(right_stats_filepath, "w") as f:
            writer = csv.writer(f)
            for key, value in prediction_stats["right"].items():
                writer.writerow([key, value])

def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'hc': 0, 'pd': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Length of a chunk
    frame_rate = 100
    sentence_len_frames = int(frame_rate * hparams["sentence_len"])
    sentence_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "duration", "feat_file", "start")
    @sb.utils.data_pipeline.provides("sig", "feats")
    def audio_pipeline(wav, duration, feat_file, start):
        feats = torch.load(feat_file, weights_only=True)
        if duration < hparams["sentence_len"]:
            sig, fs = torchaudio.load(wav)
        else:
            start_frame = int(start * frame_rate)
            start_sample = int(start * hparams["sample_rate"])
            sig, fs = torchaudio.load(
                wav, num_frames=sentence_len_sample, frame_offset=start_sample
            )
            feats = feats[start_frame:start_frame + sentence_len_frames]

        return sig.squeeze(0), feats

    @sb.utils.data_pipeline.takes("patient_type", "patient_gender", "patient_age", "patient_l1", "test_type")
    @sb.utils.data_pipeline.provides("info_dict")
    def info_dict_pipeline(patient_type, patient_gender, patient_age, patient_l1, test_type):
        info_dict = {
            "patient_type": patient_type,
            "patient_gender": patient_gender,
            "patient_age": patient_age,
            "patient_l1": patient_l1,
            "test_type": test_type
        }
        return info_dict

    # Define test pipeline:
    @sb.utils.data_pipeline.takes("wav", "duration", "feat_file")
    @sb.utils.data_pipeline.provides("sig", "feats")
    def test_pipeline(wav, duration, feat_file):
        feats = torch.load(feat_file, weights_only=True)

        # Get duration of sample
        duration_sample = int(duration * hparams["sample_rate"])
        duration_frames = int(duration * frame_rate)

        # Load
        sig, fs = torchaudio.load(wav)
        sig = sig.squeeze(0)

        # Case for short recordings
        if duration_sample <= sentence_len_sample:
            return sig, feats 

        # Unfold chunks
        sig = sig.unfold(0, sentence_len_sample, sentence_len_sample // 2)
        feats = feats.unfold(0, sentence_len_frames, sentence_len_frames // 2)

        # Cap chunks at 24
        if sig.size(0) > 24:
            sig = sig[:24]
            feats = feats[:24]

        return sig, feats

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
            dynamic_items=[audio_pipeline, info_dict_pipeline, label_pipeline],
            output_keys=["id", "sig", "feats", "patient_type_encoded", "info_dict"],
        ).filtered_sorted(
            key_test={"info_dict": lambda x: x["test_type"] == "vowel_repeat"}
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
