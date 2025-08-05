import torchaudio
import torch
from transformers import AutoModel

norm_mean = -4.268
norm_std = 4.569

class EAT_AS2M(torch.nn.Module):
    """EAT model finetuned on AS2M dataset with mixtures of audioset"""
    def __init__(self, source=None, freeze=True):
        super().__init__()
        if source is None:
            source = "worstchan/EAT-large_epoch20_finetune_AS2M"
        self.model = AutoModel.from_pretrained(source, trust_remote_code=True)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def compute_mel(self, waveforms):
        """Normalize and convert to mel-spectrogram"""
        waveforms = waveforms - waveforms.mean(dim=1, keepdim=True)

        mels = [
            torchaudio.compliance.kaldi.fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
            ).unsqueeze(0)
            for waveform in waveforms
        ]

        mels = torch.stack(mels)
        mels = (mels - norm_mean) / (norm_std * 2)
        return mels

    def forward(self, waveform, length=None):
        """Extract features"""
        mel = self.compute_mel(waveform)
        return self.model.extract_features(mel)
