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

import collections
import csv
import sys
import tqdm

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main


def llrd(encoder, name_of_layers_module, optimizer, lr, alpha=0.9):
    params = []
    for i, layer in enumerate(encoder.get_submodule(name_of_layers_module)):
        params.append({"params": layer.parameters(), "lr": lr * alpha**i})

    return optimizer(params)


def mixup(sample, lambdas):
    bs = sample.size(0) // 2
    return sample[:bs] * lambdas + sample[bs:] * (1 - lambdas)


def sample_lamda(size, alpha):
    return torch.beta(alpha, alpha)


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
        if stage != sb.Stage.TEST:
            labels, _ = batch.diagnosis
        else:
            labels = None

        # Augmentations, if specified
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Compute features
        if self.hparams.prep_ssl:
            feats, _ = batch.ssl_feats
        else:
            feats = self.modules.compute_features(wavs)

        with torch.no_grad():
            if hasattr(self.hparams, "mixup_alpha") and stage == sb.Stage.TRAIN:
                s = (feats.size(0) // 2, 1, 1)
                lambdas = self.hparams.beta_sampler.rsample(s).to(self.device)
                feats = mixup(feats, lambdas)
                labels = mixup(labels, lambdas.squeeze(-1))

        # Embeddings + speaker classifier
        #embeddings = self.modules.embedding_model(feats)
        #outputs = self.modules.classifier(embeddings).squeeze()
        outputs = feats.squeeze()
        print(outputs.shape)

        # Outputs
        return outputs, labels, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using patient-type as label."""

        predictions, labels, lens = predictions

        # Concatenate labels in case of wav_augment
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            patient_type = self.hparams.wav_augment.replicate_labels(labels)

        # Compute loss with weights, but only for train
        if stage == sb.Stage.TRAIN:
            loss = self.hparams.compute_cost(
                predictions, labels, label_smoothing=self.hparams.label_smoothing
            )

        # Record stats for the validation samples, without weighting/smoothing etc.
        elif stage == sb.Stage.VALID:
            loss = self.hparams.compute_cost(predictions, labels)
            pred_class = torch.argmax(predictions, dim=-1).squeeze()
            pred_class = self.hparams.label_encoder.decode_torch(pred_class)
            actual_class = torch.argmax(labels, dim=-1).squeeze()
            actual_class = self.hparams.label_encoder.decode_torch(actual_class)
            self.class_stats.append(batch.id, pred_class, actual_class)

        # Save prediction to be combined with same id predictions at the end
        if stage != sb.Stage.TRAIN:
            if len(batch.id) == 1:
                self.predictions[batch.id[0]] = predictions.cpu().detach()
            else:
                for uid, pred in zip(batch.id, predictions):
                    self.predictions[uid] = pred.cpu().detach()

            # Cannot compute loss for test because we don't have labels
            if stage == sb.Stage.TEST:
                loss = torch.zeros(1)

        return loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        if not should_step:
            return

        if hasattr(self.modules, "compute_features"):
            if self.modules.compute_features.freeze == False:
                self.wavlm_sched.step()
        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

    def init_optimizers(self):
        """Initialize optimizer for features and all else."""
        all_params = []
        wavlm_opt = None
        for name, module in self.modules.items():
            if name == "compute_features" and not self.hparams.freeze_wavlm:
                wavlm_opt = llrd(
                    module,
                    self.hparams.name_of_layers_module,
                    self.hparams.wavlm_opt_class,
                    self.hparams.lr_wavlm,
                    alpha=self.hparams.llrd_alpha,
                )
                self.wavlm_sched = self.hparams.scheduler(wavlm_opt)

            all_params.extend(module.parameters())

        self.optimizer = self.opt_class(all_params)
        self.optimizers_dict = {"opt_class": self.optimizer}
        self.checkpointer.add_recoverable("optimizer", self.optimizer)

        if wavlm_opt:
            self.optimizers_dict["wavlm"] = wavlm_opt
            self.checkpointer.add_recoverable("wavlm_opt", wavlm_opt)

    def on_stage_start(self, stage, epoch=None):
        """Initialize containers for metrics"""
        if stage != sb.Stage.TRAIN:
            self.predictions = {}
            if stage == sb.Stage.VALID:
                self.class_stats = self.hparams.class_stats()
        elif hasattr(self.hparams, "wavlm_unfreeze_epoch"):
            if self.hparams.wavlm_unfreeze_epoch == epoch:
                self.modules.compute_features.freeze = False

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID:
            stage_stats["accuracy"] = self.class_stats.summarize("accuracy")

            # Perform end-of-iteration things, like annealing, logging, etc.
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats,
                name=f"epoch_{epoch}_loss_{stage_loss:.3f}",
                min_keys=["loss"],
                max_keys=["accuracy"],
            )

            # When validation is finished, print more detailed stats
            self.class_stats.write_stats(sys.stdout)

        # For test, we write our predictions in case we want to upload for competition
        elif stage == sb.Stage.TEST:
            self.write_predictions()

            # Also print ssl weights
            if hasattr(self.hparams, "ssl_weights_file"):
                with open(self.hparams.ssl_weights_file, "w", encoding="utf-8") as w:
                    weights = self.modules.compute_features.weights.cpu().numpy()
                    for weight in weights:
                        w.write(str(weight))
                        w.write("\n")

    def write_predictions(self):
        """Write predictions to file for further analysis"""
        with open(self.predictions_file, "w", newline="") as f:
            fields = ["uid", "diagnosis_control", "diagnosis_mci", "diagnosis_adrd"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for uid, prediction in self.predictions.items():
                predictions = torch.softmax(prediction.squeeze(), dim=-1)
                w.writerow(
                    {
                        "uid": uid,
                        "diagnosis_control": predictions[0].numpy(),
                        "diagnosis_mci": predictions[1].numpy(),
                        "diagnosis_adrd": predictions[2].numpy(),
                    }
                )


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'control': 0, 'mci': 1, ..)
    hparams["label_encoder"] = sb.dataio.encoder.CategoricalEncoder()
    label_names = ("diagnosis_control", "diagnosis_mci", "diagnosis_adrd")
    hparams["label_encoder"].expect_len(len(label_names))
    hparams["label_encoder"].update_from_iterable(label_names)

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("filepath")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(filepath):
        return sb.dataio.dataio.read_audio(filepath)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes(*label_names)
    @sb.utils.data_pipeline.provides("diagnosis")
    def label_pipeline(*labels):
        return torch.FloatTensor(labels)

    # Pretrained ssl feature pipeline
    @sb.utils.data_pipeline.takes("ssl_path")
    @sb.utils.data_pipeline.provides("ssl_feats")
    def ssl_pipeline(ssl_path):
        return torch.load(ssl_path, map_location="cpu", weights_only=True).float()

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

        # If precomputed features are available, add them
        if hparams["prep_ssl"]:
            items.append(ssl_pipeline)
            keys.append("ssl_feats")

        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=json_path, dynamic_items=items, output_keys=keys
        )

    # Sort test set so the predictions file is in the expected order
    datasets["test"] = datasets["test"].filtered_sorted(sort_key="id")

    return datasets


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    from prepare_prepare import prepare_prepare

    run_on_main(
        prepare_prepare,
        kwargs={
            "data_folder": hparams["data_folder"],
            "manifests": hparams["manifests"],
            "prep_ssl": hparams["prep_ssl"],
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

    detector.predictions_file = detector.hparams.valid_predictions_file
    detector.evaluate(datasets["valid"], min_key="loss")
    detector.predictions_file = detector.hparams.test_predictions_file
    detector.evaluate(datasets["test"], min_key="loss")
