import sys
import tqdm
import json
import torch
import pathlib
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


def transcribe_file(filename):
    audio, fs = torchaudio.load(filename)
    #audio = 2 * audio
    audio = torch.rand_like(audio) * 0.01 + audio
    text = pipe(audio.squeeze().numpy())
    return text["text"]

if __name__ == "__main__":

    assert len(sys.argv) == 2, "Expected one argument, folder to transcribe"

    folder = pathlib.Path(sys.argv[1])

    transcripts = {}
    for filename in tqdm.tqdm(list(folder.glob("*.flac"))):
        if "sfpd" in filename.name:
            print(transcribe_file(filename))

    #with open(sys.argv[1] + "_transcripts.json", "w") as w:
    #    json.dump(transcripts, w)
