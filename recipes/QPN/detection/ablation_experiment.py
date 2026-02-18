"""
Explore how well the "speech boundaries" hypothesis explains the model behavior
by ablating away the speech boundaries and measuring performance ont he final task.
"""

import sys
import json
import logging
import pprint
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from torch.nn.functional import binary_cross_entropy


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

        # Compute loss
        if stage == sb.Stage.TRAIN:
            loss = self.hparams.bce_loss(outputs, labels, weight=weight)

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
        # Combine chunks using two strategies
        combined_avg = self.combine_chunks(how="avg")

        # Generate overall metrics, using stored threshold for test set
        avg_threshold = None #if stage == sb.Stage.VALID else self.avg_threshold
        metrics_comb_avg = self.metrics_by_category(
            combined_avg, target_category=None, threshold=avg_threshold
        )

        # Log overall metrics
        chunk_stats = self.summarize_metrics(
            self.error_metrics, self.hparams.threshold
        )
        stage_stats.update({f"chunk_{k}": v for k, v in chunk_stats.items()})
        stage_stats.update(
            {f"comb_avg_{k}": v for k, v in metrics_comb_avg["overall"].items()}
        )
        print(stage_stats)

        # Log metrics split by given categories
        for category in self.hparams.metric_categories:
            threshold = metrics_comb_avg["overall"]["threshold"]
            cat_metrics = self.metrics_by_category(
                combined_scores=combined_avg, target_category=category, threshold=threshold
            )
            #print(f"Comb avg breakdown by {category}")
            #print(pprint.pformat(cat_metrics, indent=2, compact=True, width=300))


    def on_evaluate_start(self, max_key=None, min_key=None):
        """Recover best checkpoint for evaluation, keeping track of threshold"""
        if self.checkpointer is not None:
            checkpoint = self.checkpointer.recover_if_possible(
                max_key=max_key, min_key=min_key
            )
            self.avg_threshold = checkpoint.meta["comb_avg_threshold"]

    def combine_chunks(self, how="avg"):
        """Aggregates predictions made on all individual chunks"""
        ids = self.error_metrics.ids
        scores = self.error_metrics.scores
        labels = self.error_metrics.labels
        info_dicts = self.error_metrics.info_dicts

        combined_scores = {}
        for i, score, label, info_dict in zip(ids, scores, labels, info_dicts):
            utt_id, chunk = i.rsplit("_", 1)

            # Add a computed category to the info_dict, first/non-first language
            if (
                info_dict["l1"] == "French" and info_dict["lang"] == "fr"
                or info_dict["l1"] == "English" and info_dict["lang"] == "en"
            ):
                info_dict["first_lang"] = "First Language"
            elif info_dict["task"] == "vowel_repeat":
                info_dict["first_lang"] = "No Language"
            else:
                info_dict["first_lang"] = "Non-first language"


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

        return breakdown

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


def ablate_boundary_segments(sig, width=-1, random=False):
    """Use energy-based measure to find boundaries and ablate them."""
    if width < 0:
        return sig

    energy = sig.square().unfold(dimension=0, size=640, step=320).sum(dim=1)
    median = torch.nn.functional.pad(energy, (4, 4)).unfold(dimension=0, size=9, step=1).median(dim=1).values
    segments = (median > 0.01).float()
    boundaries = segments.diff().abs()
    indexes = boundaries.nonzero(as_tuple=True)[0]

    # 640 for central frame, plus 320 on either side per width marker
    ablate_width = (width + 1) * 640
    #generator = torch.Generator().manual_seed(123)
    for index in indexes:
        # Ablate random section if needed
        if random:
            start = torch.randint(sig.size(0), (1,))#, generator=generator)
        else:
            start = max(0, (index - width) * 320)

        # ABLATE!
        sig[start:start+ablate_width] = 0

    return sig


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.expect_len(2)
    label_encoder.enforce_label("PD", 1)
    label_encoder.enforce_label("HC", 0)

    @sb.utils.data_pipeline.takes("wav", "duration", "start")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, duration, start):
        sig, fs = sb.dataio.audio_io.load(
            wav,
            num_frames=int(duration * hparams["sample_rate"]),
            frame_offset=int(start * hparams["sample_rate"]),
        )

        # ABLATE THE BOUNDARIES
        # random controls whether the function returns an identical
        # number / size of random segments for comparison
        random = False
        sig = ablate_boundary_segments(sig.squeeze(0), width=5, random=random)

        # PREPEND BLANK FRAMES AS AN ATTENTION SINK
        # https://arxiv.org/abs/2309.17453
        # Without this, ablating (boundaries or random) both *help* performance
        sig = torch.nn.functional.pad(sig, (1000, 0))

        return sig

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

    datasets = {}
    train_info = {
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    out_keys = ["id", "sig", "patient_type_encoded", "info_dict"]
    for dataset in train_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=train_info[dataset],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=out_keys,
        )

        for key, values in hparams["test_keep_keys"].items():
            datasets[dataset] = datasets[dataset].filtered_sorted(
                key_test={"info_dict": lambda x: x[key] in values},
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

    datasets = dataio_prep(hparams)

    # Load pretrained model
    classifier_sd = torch.load("pretrained/classifier.ckpt")
    hparams["classifier"].load_state_dict(classifier_sd)
    embedding_sd = torch.load("pretrained/embedding_model.ckpt")
    hparams["embedding_model"].load_state_dict(embedding_sd)

    # Brain class initialization
    parkinson_brain = ParkinsonBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        #checkpointer=hparams["checkpointer"],
    )

    parkinson_brain.evaluate(
        test_set=datasets["test"],
        #max_key=hparams["error_metric"],
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
