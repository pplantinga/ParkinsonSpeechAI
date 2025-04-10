"""Correlate the activations with various features

Author: Peter Plantinga
"""
import hyperpyyaml, torch, speechbrain, json, torchaudio, argparse, scipy, collections
import torch.nn.functional as F
import matplotlib.pyplot as plt

from speechbrain.decoders.seq2seq import S2SWhisperGreedySearcher

SAMPLE_RATE = 16000
DEVICE = "cuda"


def load_models(hparams_file):
    """Load hparams and models and data"""
    overrides = {
        "data_folder": "/home/competerscience/Documents/data/Neuro_split",
        "storage_folder": "results",
        "pretrained_source": "/home/competerscience/Documents/data/ssl-models/whisper-small",
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

    # Iterate test data and compute+store network activations
    words = {}
    activations = {}
    attention_scores = {}
    predictions = {}
    tokens = []
    for sample_id, sample in test_data.items():
        audio = audio_pipeline(sample["wav"], sample["duration"], sample["start"])
        print(sample_id)

        # Forward pass
        mel = compute_features._get_mel(audio)
        features = compute_features.forward_encoder(mel)
        embedding = embedding_model(features)
        prediction = classifier(embedding)

        # Pass relative length to searcher
        length = torch.tensor([sample["duration"] / 30.0])
        # Set the language token
        lang_token = compute_features.to_language_token(sample["info_dict"]["lang"])
        searcher.set_lang_tokens(lang_token)
        hyps, lens, scores, log_probs, attns = searcher(features, length)
        if len(hyps[0]) < 3:
            continue

        # Assembly and reshape
        decoder_attn = sae_layer.attention_scores.squeeze(-1).clone()
        cross_attn = torch.stack([a.view(-1, 1500).sum(0) for a in attns])

        # Remove EOS tokens
        cross_attn = cross_attn[:-1]
        log_probs = log_probs.squeeze()[:-1]

        #print("Decoder attn shape", decoder_attn.shape)
        #print("Cross attn shape", cross_attn.shape)
        #print("log_probs shape", log_probs.shape)


        # Find the top k attention frames
        values, indexes = decoder_attn.squeeze().topk(k=10)

        # Find decoder token with highest score for these ten frames
        print(compute_features.tokenizer.decode(hyps[0]))
        for index in indexes:
            decoder_idx = cross_attn[1:, index].argmax() + 1
            tok = compute_features.tokenizer.decode(hyps[0][decoder_idx])
            tokens.append(tok)
            print(tok)

        # Combine
        #decoder_attn = F.avg_pool1d(decoder_attn, 5, 1, 2, count_include_pad=False)
        #decoder_attn[decoder_attn < 0.1 * decoder_attn.amax()] = 0

        #token_attn = (cross_attn * decoder_attn).sum(dim=1)#, keepdim=True)
        #decoder_token_idx = token_attn.argmax()

        #print("Token attn shape", token_attn.shape)

        # Store computed activations
        #words[sample_id] = (log_probs * token_attn).sum(dim=0)
        #words[sample_id] = log_probs[decoder_token_idx].clone()
        activations[sample_id] = sae_layer.get_activations().clone()
        predictions[sample_id] = prediction.squeeze().clone()

    print(collections.Counter(tokens).most_common(10))

    return words, activations, predictions


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
            corr = correlation(activations, word_vec).abs()
            if abs(corr) > max_word_corr:
                max_word_corr = abs(corr)
                max_word_idx = j
                max_word_vec = word_vec

        print(max_word_idx)
        word = tokenizer.decode([max_word_idx])
        print(f"Max correlated word '{word}' with corr: {max_word_corr}")


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
