import argparse
import pathlib
import torch
import torchaudio
import tqdm

from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM


def load_wavlm(wavlm_folder, device):
    wavlm = WavLM(
        source=wavlm_folder,
        save_path=wavlm_folder,
        freeze_feature_extractor=True,
        freeze=True,
        output_norm=False,
        output_all_hiddens=True,
    )
    return wavlm.to(device)


def process_data(data_folder, wavlm, device):
    data_folder = pathlib.Path(data_folder)
    for audio_filename in tqdm.tqdm(list(data_folder.glob("*/*.flac"))):
        sample, fs = torchaudio.load(audio_filename)
        sample = sample.to(device)
        feats = wavlm(sample)

        # Save only layers 4 and 25 at 16-bit precision
        # Layer 4 gets the most weight when using weighted ssl
        # Shape is [25, 1, 1499, 1024]
        save_feats = torch.cat([feats[4, 0], feats[-1, 0]], dim=1).to(torch.bfloat16)
        # Shape should be [1499, 2048]
        torch.save(save_feats, audio_filename.with_suffix(".pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder")
    parser.add_argument("wavlm_folder")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    wavlm = load_wavlm(args.wavlm_folder, args.device)
    process_data(args.data_folder, wavlm, args.device)
