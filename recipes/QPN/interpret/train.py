# !/usr/bin/python3
"""Recipe for interpretability on the Quebec Parkinson's Network Speech Dataset

To run this recipe, use the following command:
> python train.py train.yaml

Author
    * Peter Plantinga 2024
"""

import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb

from torch.utils.data import DataLoader
from tqdm import tqdm

logger = sb.utils.logger.get_logger("train.py")


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

        # Augmentations, if specified (don't change batch size)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Compute features and sparsity loss (tied to 
        feats = self.modules.compute_features(wavs, lens)
        target_embeddings = self.modules.embedding_model(feats)
        predict_embeddings = self.modules.embedding_sae(feats)

        return predict_embeddings, target_embeddings

    def compute_objectives(self, outputs, batch, stage):
        """Computes the loss using patient-type as label."""

        # Get predictions and labels
        predict_embeddings, target_embeddings = outputs

        # Reconstruction loss improves fidelity, ensuring we get
        # an accurate representation of the model decision
        reconstruction_loss = self.sae_layer.get_fidelity_loss()
        self.fidelity_losses.append(reconstruction_loss.detach())

        # Sparsity loss is warmed up and makes features activate
        # less often which improves their interpretability
        sparsity_loss = self.sae_layer.sparse_loss.mean()
        self.sparsity_losses.append(sparsity_loss.detach().clone())
        sparsity_loss *= self.hparams.sparsity_weight * self.sparsity_warmup

        # Diversity loss encourages different features to activate by
        # maximizing the entropy of the mask probabilities.
        diversity_loss = self.sae_layer.diversity_loss.mean()
        self.diversity_losses.append(diversity_loss.detach().clone())
        diversity_loss *= self.hparams.diversity_weight * self.diversity_warmup

        return reconstruction_loss + sparsity_loss + diversity_loss

    def on_stage_start(self, stage, epoch=None):
        """Save losses for tracking."""
        self.fidelity_losses = []
        self.sparsity_losses = []
        self.diversity_losses = []
        if epoch == 1:
            self.sparsity_warmup = 0.0
            self.diversity_warmup = 0.0
        elif not hasattr(self, "sparsity_warmup"):
            self.sparsity_warmup = 1.0
            self.diversity_warmup = 1.0

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        stage_stats = {
            "loss": stage_loss,
            "fidelity": torch.stack(self.fidelity_losses).mean().detach(),
            "sparsity": torch.stack(self.sparsity_losses).mean().detach(),
            "temperature": self.sae_layer.mask_temperature.temperature,
            "diversity": torch.stack(self.diversity_losses).mean().detach()
        }
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID:
            lr = self.lr_scheduler.get_last_lr()

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats, min_keys=["fidelity"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """
        # Just create a pointer to the module for convenience
        self.sae_layer = self.modules.embedding_sae.get_submodule(
            self.hparams.sae_layer
        )

        # Now actually initialize the opts
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
            self.sae_layer.update_temperature()
            if self.sparsity_warmup < 1.0:
                self.sparsity_warmup += 1 / self.hparams.sparsity_warmup
            if self.diversity_warmup < 1.0:
                self.diversity_warmup += 1 / self.hparams.diversity_warmup


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    @sb.utils.data_pipeline.takes("wav", "duration", "start")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, duration, start):
        sig, fs = torchaudio.load(
            wav,
            num_frames=int(duration * hparams["sample_rate"]),
            frame_offset=int(start * hparams["sample_rate"]),
        )

        return sig.squeeze(0)

    datasets = {}
    train_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    for dataset in train_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=train_info[dataset],
            dynamic_items=[audio_pipeline],
            output_keys=["id", "sig"],
        )

    # Remove keys from training data for e.g. training only on men
    for key, values in hparams["train_keep_keys"].items():
        datasets["train"] = datasets["train"].filtered_sorted(
            key_test={"info_dict": lambda x: x[key] in values},
        )
    for key, values in hparams["test_keep_keys"].items():
        for dataset in ["valid", "test"]:
            datasets[dataset] = datasets[dataset].filtered_sorted(
                key_test={"info_dict": lambda x: x[key] in values},
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
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()
    hparams["embedding_sae"].insert_adapters()

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
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )

    logger.info("Final test result:")
    parkinson_brain.metrics_json = hparams["test_metrics_json"]
    parkinson_brain.evaluate(
        test_set=datasets["test"],
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
