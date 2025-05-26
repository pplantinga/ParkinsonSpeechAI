"""Correlate the activations with various features
"""
import hyperpyyaml, torch, speechbrain, json, torchaudio, argparse, scipy, pandas, sklearn, numpy
import torch.nn.functional as F
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
DEVICE = "cuda"


def load_models(hparams_file):
    """Load hparams and models and data"""
    overrides = {
        "data_folder": "/path/to/qpn",
        "storage_folder": "results",
        "pretrained_source": "openai/whisper-small",
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

    with open(hparams["train_annotation"]) as f:
        train_data = json.load(f)

    filtered_data = {
        k: v for k, v in test_data.items() if v["info_dict"]["task"] == "dpt"
    }

    filtered_train = {
        k: v for k, v in train_data.items() if v["info_dict"]["task"] == "dpt"
    }

    return models, filtered_data, filtered_train


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
    activations = {}
    attention_scores = {}
    predictions = {}
    labels = {}
    for sample_id, sample in test_data.items():
        print(sample_id)
        audio = audio_pipeline(sample["wav"], sample["duration"], sample["start"])

        # Forward pass
        features = compute_features(audio)
        embedding = embedding_model(features)
        prediction = classifier(embedding)

        # Store computed activations
        activations[sample_id] = sae_layer.get_activations().detach().clone()
        attention_scores[sample_id] = sae_layer.attention_scores.detach().clone()
        predictions[sample_id] = prediction.squeeze().detach().clone()
        labels[sample_id] = 1 if sample["info_dict"]["ptype"] == "PD" else 0

    compute_f1_score(predictions, labels)

    return activations, attention_scores, predictions


def compute_f1_score(predictions, labels):
    """Compute the overall F1 score of the system per person"""
    per_person_predi = {}
    per_person_label = {}
    for utterance_id in predictions:
        person_id = utterance_id.split("_")[1]
        if person_id not in per_person_predi:
            per_person_predi[person_id] = []
            per_person_label[person_id] = float(labels[utterance_id])

        per_person_predi[person_id].append(predictions[utterance_id].cpu().sigmoid())

    for person_id in per_person_predi:
        per_person_predi[person_id] = float(numpy.mean(per_person_predi[person_id]) > 0.583)

    label_array = numpy.array(list(per_person_label.values()))
    predi_array = numpy.array(list(per_person_predi.values()))

    print(sklearn.metrics.f1_score(label_array, predi_array))

@torch.no_grad()
def collect_features(attention_scores, test_data):
    """Collect the (attention-weighted) vocal feature scores"""

    # Match whisper step size, so we can use attention scores
    vocal_feats = speechbrain.lobes.features.VocalFeatures(step_size=0.02)

    features = {}
    feats_attn = {}
    pause = {}
    energy = {}
    corrs = []
    for sample_id, sample in test_data.items():
        audio = audio_pipeline(sample["wav"], sample["duration"], sample["start"])
        sq_audio = audio.squeeze().square()
        e = sq_audio.unfold(dimension=0, size=640, step=320).sum(dim=-1)
        energy[sample_id] = e

        pause[sample_id] = compute_pause_features(audio)

        # Pad with one window to match whisper output, so we can use attention
        vocal_feat = vocal_feats(audio)
        vocal_feat = F.pad(vocal_feat, (0, 0, 0, 1500 - vocal_feat.size(1)))

        # Overall vocal feat mean
        features[sample_id] = vocal_feat.mean(dim=1).detach()

        # attention-weighted vocal feat mean
        if vocal_feat.size(-1) == attention_scores[sample_id].size(-1):
            s = (vocal_feat * attention_scores[sample_id]).mean(dim=1)
            feats_attn[sample_id] = s
        else:
            s = vocal_feat.unsqueeze(-1) * attention_scores[sample_id].unsqueeze(-2)
            feats_attn[sample_id] = s.flatten(-2, -1).mean(dim=1)

        # Cross correlation
        a = attention_scores[sample_id].squeeze()[:e.size(0)]
        #print(correlation(sma_norm(e), sma_norm(a)))
        #corrs.append(correlation(sma_norm(e), sma_norm(a)))
        corrs.append(energy_correlation(e, a))

    print("Median correlation:", torch.tensor(corrs).median())
    bins = torch.arange(11) / 10 - 0.7
    plt.rcParams["figure.figsize"] = (5,4)
    plt.hist(torch.tensor(corrs).cpu().numpy(), bins=bins.numpy())
    plt.axvline(x=0.0, color="white", linestyle="--")
    plt.title("Correlations between Energy and Attention")
    plt.xlabel("Correlation between Energy and Attention")
    plt.ylabel("Count of 30-second samples")
    plt.savefig("energy-attention-corrs.png", dpi=300, bbox_inches="tight")
    plt.clf()

    return features, feats_attn, pause, energy

def sma_norm(signal, kernel=11, clamp_min=1e-3, log=False):
    signal = signal.view(1, -1) 
    #signal = F.avg_pool1d(
    #    signal, kernel, stride=1, padding=kernel // 2, count_include_pad=False
    #)  
    weight = torch.hann_window(kernel, device=DEVICE).view(1, 1, -1) 
    signal = F.conv1d(signal, weight, padding=kernel // 2)
    signal = signal.squeeze(0) / signal.amax()
    if log:
        return signal.clamp(min=clamp_min).log()
    return signal

def energy_correlation(energy, attention):
    energy_sma = sma_norm(energy, kernel=21)
    attention_sma = sma_norm(attention, kernel=21)

    energy_binarized = (energy_sma > energy_sma.amax() * 0.05).float()
    attention_binarized = (attention_sma > attention_sma.amax() * 0.05).float()

    corr = correlation(energy_binarized, attention_binarized)
    #print(corr)
    return corr

def compute_pause_features(audio, n=10):
    energy = audio.squeeze().unfold(dimension=0, size=1600, step=800).sum(dim=0)

    # Silence = Energy is lower than 1% of peak
    pause_frames = energy < energy.amax() * 0.05

    # Ratio of silent frames to length
    pause_frame_count = pause_frames.count_nonzero()
    pause_frame_ratio = pause_frame_count / len(pause_frames)

    # Count 1-second unbroken silent intervals (Allow for 1 spurious frame)
    pause_count = 0
    cooldown = 0
    longest_pause = 0
    current_pause = 0
    for i in range(len(pause_frames) - n):
        if cooldown > 0:
            cooldown -= 1

        if not cooldown and pause_frames[i:i+n].count_nonzero() >= n - 1:
            pause_count += 1
            cooldown = n

        if pause_frames[i]:
            current_pause += 1
        else:
            current_pause = 0

        if current_pause > longest_pause:
            longest_pause = current_pause

    p = [pause_frame_count, pause_frame_ratio, pause_count, longest_pause]

    return torch.tensor(p, device=DEVICE)


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


def compute_correlations(features, activations, predictions, attributes, pause):
    """Compute correlations between features and activations.

    NOTE: correlations with predictions is a baseline -- any correlations
    should be higher than the correlation with the final prediction.
    """
    feature_matrix = torch.stack(tuple(features.values()), dim=-1).squeeze(0)
    activation_matrix = torch.stack(tuple(activations.values()), dim=-1).squeeze(0)
    attribute_matrix = torch.stack(tuple(attributes.values()), dim=-1)
    prediction_matrix = torch.stack(tuple(predictions.values()))
    pause_matrix = torch.stack(tuple(pause.values()), dim=-1)

    print(feature_matrix.shape)
    print(activation_matrix.shape)
    print(prediction_matrix.shape)
    print(attribute_matrix.shape)
    print(pause_matrix.shape)


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
        predictions = prediction_matrix[indexes]

        corr = correlation(activations, predictions)
        print(f"Correlation of active feats with prediction: {corr}")
        score.append(corr)
        corr, pvalue = scipy.stats.spearmanr(activations.cpu(), predictions.cpu())
        print(f"Rank Correlation of active with prediction: {corr}")
        print(f"Rank Correlation pvalue of active w/predict: {pvalue}")

        for j in range(5):
            attributes = attribute_matrix[j, indexes]
            corr, pvalue = scipy.stats.spearmanr(activations.cpu(), attributes.cpu())
            print(f"Correlation of activations with attribute {j}: {corr}")

        max_corr = 0
        max_feat_index = -1
        for j in range(pause_matrix.size(0)):
            pause_feat = pause_matrix[j, indexes]
            corr, _ = scipy.stats.spearmanr(activations.cpu(), pause_feat.cpu())
            #pause_corr = correlation(activations, pause_feat)

            if abs(corr) > abs(max_corr):
                max_corr = corr
                max_feat_index = j
        print("Max pause index", max_feat_index)
        print("Max pause correlation", max_corr)


        # Try different features
        max_feat = 0
        max_feat_index = -1
        max_pvalue = 0
        for j in range(feature_matrix.size(0)):
            feature = feature_matrix[j, indexes]
            #feat_corr = correlation(activations, feature).abs()
            feat_corr, pvalue = scipy.stats.spearmanr(activations.cpu(), feature.cpu())

            if i == 61 and j == 12:
                plot_correlations(activations, predictions, feature, feat_corr)

            if abs(feat_corr) > abs(max_feat):
                max_feat = feat_corr
                max_feat_index = j
                max_pvalue = pvalue

        print(f"Max corr feat {max_feat_index} with score {max_feat} and pvalue {max_pvalue}")
        score.append(max_feat_index)
        score.append(max_feat)

        scores.append(score)

    return scores


def plot_correlations(activations, predictions, feature, corr):
    """Plot the correlation of the best dictionary feature with the predictions."""

    feature *= 10 ** 5
    predictions = predictions.sigmoid()

    #cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(activations.cpu(), feature.cpu(), c=predictions.cpu())#, vmin=0, vmax=20, s=35, cmap=cm)

    plt.annotate("ρ = {:.3f}".format(corr), (0.15, 0.8), xycoords="figure fraction")
    plt.title("Dictionary Entry #61")
    plt.xlabel("Activation Strength")
    plt.ylabel("Spectral Flux (x10^5)")
    cb = plt.colorbar(sc, pad=0.08, label="Predicted Probability of PD")
    cb.set_label("Predicted Probability of PD", labelpad=-52)
    plt.savefig("correlation_spec_flux.pdf", dpi=300, bbox_inches="tight")
    plt.clf()

    #plt.scatter(activations.cpu(), predictions.cpu())
    #plt.savefig("correlation-prediction.png")
    #plt.clf()
    #plt.scatter(activations.cpu(), feature.cpu())
    #plt.savefig("correlation-activation.png")


def write_to_csv(
    activations,
    activations_train,
    feats,
    feats_train,
    predictions,
    predictions_train
):
    # Aggregate scores per person
    people_scores = {}
    for dataset, activation_dict, feat_dict, prediction_dict in [
            ("test", activations, feats, predictions),
            ("train", activations_train, feats_train, predictions_train),
    ]:
        for sample_id, matrix in activation_dict.items():
            subject_id = sample_id.split("_")[1]

            if subject_id not in people_scores:
                people_scores[subject_id] = {
                    "spectral_flux": 0,
                    "activation_score": 0,
                    "prediction_score": 0,
                    "chunk_count": 0,
                    "dataset": dataset,
                }

            spectral_flux = feat_dict[sample_id][0, 12].cpu().numpy() * 10000
            activation_score = activation_dict[sample_id][0, 61].cpu().numpy()
            prediction_score = prediction_dict[sample_id].sigmoid().cpu().numpy()

            people_scores[subject_id]["spectral_flux"] += spectral_flux
            people_scores[subject_id]["activation_score"] += activation_score
            people_scores[subject_id]["prediction_score"] += prediction_score
            people_scores[subject_id]["chunk_count"] += 1

    # Divide by count
    for subject_id in people_scores:
        n = people_scores[subject_id]["chunk_count"]
        people_scores[subject_id]["spectral_flux"] /= n
        people_scores[subject_id]["activation_score"] /= n
        people_scores[subject_id]["prediction_score"] /= n

    df = pandas.DataFrame.from_dict(people_scores, orient="index")
    df.to_csv("spectral_flux_scores.csv", index_label="subject_id")

    

# ####################################
# MAIN
# ####################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_file")
    args = parser.parse_args()

    models, test_data, train_data = load_models(args.hparams_file)
    activations, attn_scores, predictions = collect_activations(models, test_data)
    #activations_train, attn_scores_train, predictions_train = collect_activations(models, train_data)
    _, feats, pause, energy = collect_features(attn_scores, test_data)
    #_, feats_train, _ = collect_features(attn_scores_train, train_data)

    attributes = collect_attributes(test_data)

    correlations = compute_correlations(feats, activations, predictions, attributes, pause)

    #plot_correlations(correlations)

    #activations_train = {}
    #feats_train = {}
    #predictions_train = {}

    #write_to_csv(activations, activations_train, feats, feats_train, predictions, predictions_train)
