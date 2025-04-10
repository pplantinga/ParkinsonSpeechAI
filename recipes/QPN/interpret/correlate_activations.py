"""Correlate the activations with various features

Author: Peter Plantinga
"""
import hyperpyyaml, torch, speechbrain, json, torchaudio, argparse, scipy
import torch.nn.functional as F
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
DEVICE = "cuda"


def load_models(hparams_file):
    """Load hparams and models and data"""
    overrides = {
        "data_folder": "/home/competerscience/Documents/data/Neuro_split",
        "storage_folder": "results",
        "pretrained_source": "/home/competerscience/Documents/data/ssl-models/whisper-small",
    }
    with open(hparams_file) as f:
        hparams = hyperpyyaml.load_hyperpyyaml(f, overrides=overrides)
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()
    compute_features = hparams["compute_features"].to(DEVICE).eval()
    embedding_model = hparams["embedding_model"].to(DEVICE).eval()
    classifier = hparams["classifier"].to(DEVICE).eval()
    hparams["embedding_sae"].insert_adapters()
    embedding_sae = hparams["embedding_sae"].to(DEVICE).eval()
    sae_layer = embedding_sae.get_submodule(hparams["sae_layer"])

    hparams["checkpointer"].recover_if_possible()

    models = (compute_features, embedding_model, sae_layer, classifier)

    with open(hparams["test_annotation"]) as f:
        test_data = json.load(f)

    with open(hparams["valid_annotation"]) as f:
        test_data.update(json.load(f))

    filtered_data = {
        k: v for k, v in test_data.items() if v["info_dict"]["task"] == "dpt"
    }

    return models, filtered_data


def audio_pipeline(wav, duration, start):
    """For loading audios"""
    sig, fs = torchaudio.load(
        wav,
        num_frames=int(duration * SAMPLE_RATE),
        frame_offset=int(start * SAMPLE_RATE),
    )

    return sig.to(DEVICE)


def collect_activations(models, test_data):
    """Collect the activation information"""
    compute_features, embedding_model, sae_layer, classifier = models
    sae_layer.enable_storage()

    # Iterate test data and compute+store network activations
    preactivations = {}
    activations = {}
    attention_scores = {}
    predictions = {}
    for sample_id, sample in test_data.items():
        audio = audio_pipeline(sample["wav"], sample["duration"], sample["start"])

        # Forward pass
        features = compute_features(audio)
        embedding = embedding_model(features)
        prediction = classifier(embedding)

        # Store computed activations
        preactivations[sample_id] = sae_layer.pre_activations.detach().clone()
        activations[sample_id] = sae_layer.get_activations().detach().clone()
        attention_scores[sample_id] = sae_layer.attention_scores.detach().clone()
        predictions[sample_id] = prediction.squeeze().detach().clone()

    return preactivations, activations, attention_scores, predictions


def collect_features(attention_scores, test_data):
    """Collect the (attention-weighted) vocal feature scores"""

    # Match whisper step size, so we can use attention scores
    vocal_feats = speechbrain.lobes.features.VocalFeatures(step_size=0.02)

    features = {}
    feats_attn = {}
    for sample_id, sample in test_data.items():
        audio = audio_pipeline(sample["wav"], sample["duration"], sample["start"])

        # Pad with one window to match whisper output, so we can use attention
        vocal_feat = vocal_feats(audio)
        vocal_feat = F.pad(vocal_feat, (0, 0, 0, 1500 - vocal_feat.size(1)))

        # Overall vocal feat mean
        features[sample_id] = vocal_feat.mean(dim=1).detach()

        # attention-weighted vocal feat mean
        s = (vocal_feat * attention_scores[sample_id]).mean(dim=1).detach()
        feats_attn[sample_id] = s

    return features, feats_attn


def collect_attributes(test_data):
    attributes = {}
    for sample_id, item in test_data.items():
        attributes[sample_id] = torch.tensor([
            int(item["info_dict"]["age"]),
            1 if item["info_dict"]["sex"] == "M" else 0,
            1 if item["info_dict"]["ptype"] == "HC" else 0,
            1 if item["info_dict"]["l1"] == "French" else 0,
            1 if item["info_dict"]["lang"] == "fr" else 0,
        ], device=DEVICE).float()

    return attributes


def correlation(vector_a, vector_b):
    cov = ((vector_a - vector_a.mean()) * (vector_b - vector_b.mean())).mean()
    return cov / vector_a.std() / vector_b.std()


def compute_correlations(features, preactivations, activations, predictions, attributes):
    """Compute correlations between features and activations.

    NOTE: correlations with predictions is a baseline -- any correlations
    should be higher than the correlation with the final prediction.
    """
    feature_matrix = torch.stack(tuple(features.values()), dim=-1).squeeze(0)
    preactivation_matrix = torch.stack(tuple(preactivations.values()), dim=-1).squeeze(0)
    activation_matrix = torch.stack(tuple(activations.values()), dim=-1).squeeze(0)
    prediction_matrix = torch.stack(tuple(predictions.values())).view(1, -1)
    attribute_matrix = torch.stack(tuple(attributes.values()), dim=-1)

    print(feature_matrix.shape)
    print(activation_matrix.shape)
    print(prediction_matrix.shape)
    print(attribute_matrix.shape)


    # Correlate predictions with activations as a baseline for each
    scores = []
    for i in range(activation_matrix.size(0)):
        score = []
        activations = activation_matrix[i].count_nonzero()
        if activations < 10:
            continue

        print(f"Activations for feature {i}: {activations}")
        score.append(activations)

        indexes = activation_matrix[i].nonzero().squeeze()
        activations = activation_matrix[i, indexes]
        predictions = prediction_matrix[0, indexes]

        corr = correlation(activations, predictions).abs()
        print(f"Correlation of active feats with prediction: {corr}")
        score.append(corr)
        corr, pvalue = scipy.stats.spearmanr(activations.cpu(), predictions.cpu())
        print(f"Rank Correlation of active with prediction: {corr}")

        for j in range(5):
            attributes = attribute_matrix[j, indexes]
            attr_corr = correlation(activations, attributes)
            print(f"Correlation of activations with attributes: {attr_corr}")

        #ovl_corr = correlation(preactivation_matrix[i], prediction_matrix[0]).abs()
        #print(f"Preactivation correlation with prediction: {ovl_corr}")

        # Try different features
        max_feat = 0
        max_feat_index = -1
        max_ovl_feat = 0
        max_ovl_feat_index = -1
        for j in range(feature_matrix.size(0)):
            feature = feature_matrix[j, indexes]
            feat_corr = correlation(activations, feature).abs()
            feat_ovl_corr = correlation(preactivation_matrix[i], feature_matrix[j]).abs()

            if i == 15 and j == 10:
                plot_correlations(activations, predictions, feature)

            if feat_corr > max_feat:
                max_feat = feat_corr
                max_feat_index = j

            if feat_ovl_corr > max_ovl_feat:
                max_ovl_feat = feat_ovl_corr
                max_ovl_feat_index = j

        print(f"Max corr feat {max_feat_index} with score {max_feat}")
        #print(f"Max ovl feat {max_ovl_feat_index} with score {max_ovl_feat}")
        score.append(max_feat_index)
        score.append(max_feat)

        scores.append(score)

    return scores


def plot_correlations(activations, predictions, feature):
    """Plot the correlation of the best dictionary feature with the predictions."""

    plt.scatter(activations.cpu(), predictions.cpu())
    plt.savefig("correlation-prediction.png")
    plt.clf()
    plt.scatter(activations.cpu(), feature.cpu())
    plt.savefig("correlation-activation.png")


# ####################################
# MAIN
# ####################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_file")
    args = parser.parse_args()

    models, test_data = load_models(args.hparams_file)
    preactivations, activations, attn_scores, predictions = collect_activations(models, test_data)
    features, feats_attn = collect_features(attn_scores, test_data)
    attributes = collect_attributes(test_data)

    correlations = compute_correlations(feats_attn, preactivations, activations, predictions, attributes)

    #plot_correlations(correlations)
