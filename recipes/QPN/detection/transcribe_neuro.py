import torch
import whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import argparse
import json


def load_model(model_name, size=None):
    if model_name == "crisper":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        crisper_id = "nyrahealth/CrisperWhisper"

        crisper = AutoModelForSpeechSeq2Seq.from_pretrained(
            crisper_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        crisper.to(device)

        processor = AutoProcessor.from_pretrained(crisper_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=crisper,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            batch_size=16,
            return_timestamps="word",
            torch_dtype=torch_dtype,
            device=device,
        )
        return pipe

    else:
        return whisper.load_model(size, download_root=os.environ['WHISPER_CACHE'])


def transcribe_folder(audio_folder, split, save_path, model, model_name):
    audio_path = os.path.join(audio_folder, split)
    transcripts = {}

    for i in range(2):
        batch_path = os.path.join(audio_path, f"Batch{i + 1}")
        for f in batch_path:
            uttid = f.split(".")[0] + "_" + f"Batch{i + 1}"
            if model_name == "crisper":
                transcripts[uttid] = model(f)
            else:
                transcripts[uttid] = model.transcribe(f)

    with open(os.path.join(save_path, f"{split}.json"), 'w') as f:
        json.dump(transcripts, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="The model you would like to transcribe with. "
                             "Current choices are whisper or crisper.")
    parser.add_argument("--size", type=str, required=False, default=None,
                        help="Size of whisper model. Available choices are: "
                             "small, medium, base, tiny, large")
    parser.add_argument("--audio_folder", type=str, required=True,
                        help="Path to neuro files "
                             "(should have train, valid, test directories).")
    parser.add_argument("--save_path", type=str, default="/", required=False,
                        help="Where you want to save the transcripts.")
    args = parser.parse_args()

    model = load_model(args.model, args.size)

    transcribe_folder(args.audio_folder, "train", args.save_path, model, args.model)
    transcribe_folder(args.audio_folder, "valid", args.save_path, model, args.model)
    transcribe_folder(args.audio_folder, "test", args.save_path, model, args.model)
