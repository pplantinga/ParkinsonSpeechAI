import pandas as pd
import pathlib
import argparse
import torchaudio

def prepare_qpn(data_folder, manifest_filename):
    """Output one manifest, will be split by train script."""

    # Stop if manifest exists already
    manifest_filename = pathlib.Path(manifest_filename)
    manifest_filename.parent.mkdir(parents=True, exist_ok=True)
    if manifest_filename.exists():
        return

    # Load demographic data
    data_folder = pathlib.Path(data_folder)
    demographic_data = pd.read_csv(pathlib.Path(data_folder) / "demographics.csv")

    # Create dataframe of utterances (wavs)
    wavs = list(data_folder.glob("*.wav"))
    infos = [torchaudio.info(wav) for wav in wavs]
    durations = [i.num_frames / i.sample_rate for i in infos]
    subject_ids = [wav.name.split("_")[0] for wav in wavs]
    utterance_ids = [wav.stem for wav in wavs]
    wavs = [str(wav) for wav in wavs]
    utterance_data = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "wav": wavs,
            "utterance_id": utterance_ids,
            "duration": durations,
        }
    )

    # Merge demographics and utterances, then write to manifest file
    data = pd.merge(utterance_data, demographic_data, on="subject_id", how="left")
    data = data.set_index("utterance_id")
    data.to_json(manifest_filename, orient="index", indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder")
    parser.add_argument("--output_filename", default="manifest.json")
    args = parser.parse_args()
    prepare_qpn(args.data_folder, args.output_filename)
