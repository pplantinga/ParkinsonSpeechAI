"""Correlate the activations with various features
"""
import hyperpyyaml, torch, speechbrain, json, torchaudio, argparse
import scipy, collections, unicodedata, nltk
import torch.nn.functional as F
import matplotlib.pyplot as plt

from speechbrain.decoders.seq2seq import S2SWhisperGreedySearcher

SAMPLE_RATE = 16000
DEVICE = "cuda"


def load_models(hparams_file):
    """Load hparams and models and data"""
    overrides = {
        "data_folder": "path/to/qpn",
        "storage_folder": "results",
        "pretrained_source": "openai/whisper-small",
        "compute_features": {"encoder_only": False, "output_attentions": True},
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


@torch.no_grad()
def collect_activations(models, test_data):
    """Collect the activation information"""
    compute_features, embedding_model, sae_layer, classifier = models
    sae_layer.enable_storage()

    searcher = S2SWhisperGreedySearcher(
        model=compute_features, min_decode_ratio=0.0, max_decode_ratio=1.0
    )
    tokizer = compute_features.tokenizer

    # Iterate test data and compute+store network activations
    words = {}
    activations = {}
    attention_scores = {}
    predictions = {}
    tokens = []
    count = 0
    for sample_id, sample in test_data.items():
        audio = audio_pipeline(sample["wav"], sample["duration"], sample["start"])

        energy = audio.squeeze().square().unfold(dimension=0, size=640, step=320)
        print(sample_id)

        # Forward pass
        mel = compute_features._get_mel(audio)
        features = compute_features.forward_encoder(mel)
        embedding = embedding_model(features)
        prediction = classifier(embedding)

        # Pass relative length to searcher
        length = torch.tensor([sample["duration"] / 30.0], device=DEVICE)
        # Set the language token
        lang_token = compute_features.to_language_token(sample["info_dict"]["lang"])
        searcher.set_lang_tokens(lang_token)
        hyps, lens, scores, log_probs, attns = searcher(features, length)
        if len(hyps[0]) < 3:
            continue

        # Count sentences, words, average lengths
        tok_count = len(hyps[0])
        utterance = tokizer.decode(hyps[0])
        word_count = len(utterance.split())
        word_count_per_second = word_count / sample["duration"]
        sentences = nltk.sent_tokenize(utterance)
        sent_count = len(sentences)
        sent_count_per_second = sent_count / sample["duration"]
        avg_sent_len = word_count / sent_count
        avg_word_len = sum(len(w) for w in utterance.split()) / word_count

        word_vector = [tok_count, word_count, word_count_per_second, sent_count]
        word_vector += [sent_count_per_second, avg_sent_len, avg_word_len]

        # Assembly and reshape
        detector_attn = sae_layer.attention_scores.squeeze(-1).clone()
        #cross_attn = torch.stack([a.view(-1, 1500).sum(0) for a in attns])

        # Remove EOS tokens
        #cross_attn = cross_attn[:-1]
        #log_probs = log_probs.squeeze()[:-1]

        #print("Decoder attn shape", detector_attn.shape)
        #print("Cross attn shape", cross_attn.shape)
        #print("log_probs shape", log_probs.shape)

        if count == 0 or count == 20:
            plot_energy_vs_attn(f"Attention_{count}.png", detector_attn, energy)

        # Find the top k attention frames
        #values, indexes = detector_attn.squeeze().topk(k=10)

        # Find decoder token with highest score for these ten frames
        #print(tokizer.decode(hyps[0]))
        #for index in indexes:
        #    decoder_idx = cross_attn[1:, index].argmax() + 1
        #    tok = tokizer.decode(hyps[0][decoder_idx])
        #    tokens.append(tok)
            #print(tok)
        

        #token_attn = (cross_attn * detector_attn).sum(dim=1)#, keepdim=True)
        #decoder_token_idx = token_attn.argmax()

        #print("Token attn shape", token_attn.shape)

        # Store computed activations
        #words[sample_id] = (log_probs * token_attn).sum(dim=0)
        #words[sample_id] = log_probs[decoder_token_idx].clone()
        words[sample_id] = torch.tensor(word_vector, device=DEVICE)
        activations[sample_id] = sae_layer.get_activations().clone()
        predictions[sample_id] = prediction.squeeze().clone()

    #print(collections.Counter(tokens).most_common(10))

    return words, activations, predictions


def plot_energy_vs_attn(fig_name, detector_attn, energy):
    # Frame rate is every 0.02 sec
    xs = torch.arange(detector_attn.size(1)) * 0.02
    plt.rcParams["figure.figsize"] = (5,1)
    plt.plot(xs, sma_norm(detector_attn).cpu())
    #plt.plot(xs, detector_attn.squeeze(0).cpu() / detector_attn.amax().cpu())
    plt.plot(xs[:-1], sma_norm(energy.sum(dim=1)).cpu())
    #plt.legend(["Attention 1", "Attentin 2", "Attention 3", "Attention 4", "Signal energy"])
    plt.legend(["Attn", "Energy"])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Norm. score")
    #plt.savefig("Attention-vs-Energy-multiattn.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.clf()


def sma_norm(signal, kernel=21, clamp_min=1e-3, log=False):
    signal = signal.view(1, -1)
    #signal = F.avg_pool1d(
    #    signal, kernel, stride=1, padding=kernel // 2, count_include_pad=False
    #)
    weight = torch.hann_window(kernel, device=DEVICE).view(1, 1, -1)
    signal = F.conv1d(signal, weight, padding=kernel // 2)
    signal = signal / signal.amax()
    signal = signal.squeeze(0).clamp(min=clamp_min)
    if log:
        return signal.log()
    return signal


def is_punctuation(token):
    c = token.strip()[0]
    return unicodedata.category(c).startswith("P")

def correlation(vector_a, vector_b):
    cov = ((vector_a - vector_a.mean()) * (vector_b - vector_b.mean())).mean()
    return cov / vector_a.std() / vector_b.std()


def compute_correlations(words, activations, predictions, tokenizer):
    """Compute correlations between features and activations.
    """
    word_matrix = torch.stack(tuple(words.values()), dim=-1)
    activation_matrix = torch.stack(tuple(activations.values()), dim=-1).squeeze(0)
    prediction_matrix = torch.stack(tuple(predictions.values())).view(1, -1)

    print(word_matrix.shape)
    print(activation_matrix.shape)
    print(prediction_matrix.shape)


    # Correlate predictions with activations as a baseline for each
    for i in range(activation_matrix.size(0)):
        activations = activation_matrix[i].count_nonzero()
        if activations < 10:
            continue

        print(f"Activations for feature {i}: {activations}")

        indexes = activation_matrix[i].nonzero().squeeze()
        activations = activation_matrix[i, indexes]
        predictions = prediction_matrix[0, indexes]

        corr = correlation(activations, predictions).abs()
        print(f"Correlation of active feats with prediction: {corr}")
        corr, pvalue = scipy.stats.spearmanr(activations.cpu(), predictions.cpu())
        print(f"Rank Correlation of active with prediction: {corr}")

        max_word_corr = 0
        max_word_idx = -1
        max_word_vec = None
        for j in range(word_matrix.size(0)):
            word_vec = word_matrix[j, indexes]
            #corr = correlation(activations, word_vec).abs()
            corr, _ = scipy.stats.spearmanr(activations.cpu(), word_vec.cpu())
            if abs(corr) > abs(max_word_corr):
                max_word_corr = corr
                max_word_idx = j
                max_word_vec = word_vec

        print(max_word_idx)
        print(max_word_corr)
        #word = tokenizer.decode([max_word_idx])
        #print(f"Max correlated word '{word}' with corr: {max_word_corr}")


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
    words, activations, predictions = collect_activations(models, test_data)

    compute_correlations(words, activations, predictions, models[0].tokenizer)

    #plot_correlations(correlations)
