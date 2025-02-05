import torch
from nemo.collections.asr.models import EncDecCTCModelBPE, EncDecMultiTaskModel
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector


class NeMoEncoder(torch.nn.Module):
    """One of EncDecCTCModelBPE or EncDecMultiTaskModel, just keeping encoder part."""
    def __init__(self, source, freeze, nemo_type="CTC", source_dir=None):
        super().__init__()

        self.freeze = freeze

        if nemo_type == "CTC":
            ASR_Class = EncDecCTCModelBPE
        else:
            ASR_Class = EncDecMultiTaskModel

        # Instantiate from source
        connector = None
        if source_dir is not None:
            connector = SaveRestoreConnector()
            connector.model_extracted_dir = source_dir

        if source.endswith(".nemo"):
            self.model = ASR_Class.restore_from(
                source, map_location="cpu", save_restore_connector=connector
            )
        else:
            self.model = ASR_Class.from_pretrained(
                source, map_location="cpu", save_restore_connector=connector
            )

        # Delete all decoder params
        if hasattr(self.model, "transf_decoder"):
            del self.model.transf_decoder

        # Switch model to evaluation mode
        self.model.eval()
        if hasattr(self.model.preprocessor, 'featurizer'):
            if hasattr(self.model.preprocessor.featurizer, 'dither'):
                self.model.preprocessor.featurizer.dither = 0.0
            if hasattr(self.model.preprocessor.featurizer, 'pad_to'):
                self.model.preprocessor.featurizer.pad_to = 0

        # Freeze parameters
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, audio, length):
        """Extract spectral features and run through encoder"""
        length = (length * audio.size(1)).long()
        spec, spec_len = self.model.preprocessor(input_signal=audio, length=length)
        if self.freeze:
            with torch.no_grad():
                x, _ = self.model.encoder(audio_signal=spec, length=spec_len)
        else:
            x, _ = self.model.encoder(audio_signal=spec, length=spec_len)

        x = torch.transpose(x, 1, 2)

        return x
