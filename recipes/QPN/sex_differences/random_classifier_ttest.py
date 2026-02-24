import torch
import torchaudio

from sklearn.model_selection import StratifiedKFold

import speechbrain as sb

seeds = [2731948, 501928746, 758329, 20384719, 39201777]
f1_scores = [0.680, 0.680, 0.600, 0.640, 0.667]


def dataio_prep(seed):
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
        json_path="manifest.json",
        dynamic_items=[audio_pipeline, label_pipeline, demographics_pipeline],
        #dynamic_items=[opensmile_pipeline, label_pipeline],
        output_keys=["id", "sig", "cohort_encoded", "info_dict", "subject_id", "duration"],
    )


    # Select stratified folds by sex and cohort
    # first, generate a mapping from participant ids to their cohort and sex
    with overall_dataset.output_keys_as(["subject_id", "cohort", "sex"]):
        mapping = {d["subject_id"]: d["cohort"] + d["sex"] for d in overall_dataset}
    ids, stratify_labels = zip(*mapping.items())

    sfk = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=seed
    )

    # Convert indexes back to subject ids
    folds = [
        ([ids[i] for i in train_set], [ids[i] for i in test_set])
        for train_set, test_set in sfk.split(ids, stratify_labels)
    ]

    return overall_dataset, folds


if __name__ == "__main__":
   
    # Iterate seeds to generate scores for random classifier
    for seed in seeds:
        print("Seed:", seed)
        random_classifier_f1_scores = sb.utils.metric_stats.BinaryMetricStats()
        dataset, folds = dataio_prep(seed)

        # Go through tests folds to generate random predictions
        for i, (train_ids, test_ids) in enumerate(folds):
            test_data = dataset.filtered_sorted(
                key_test={"subject_id": lambda x: x in test_ids}
            )

            scores = torch.rand(len(test_data))
            random_classifier_f1_scores.append(
                ids=[d["id"] for d in test_data],
                scores=scores,
                labels=torch.tensor([d["cohort_encoded"] for d in test_data]),
            )

        print(random_classifier_f1_scores.summarize())
