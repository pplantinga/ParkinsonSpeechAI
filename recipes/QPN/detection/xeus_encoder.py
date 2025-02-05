import os
import pathlib

import torch
from espnet2.tasks.ssl import SSLTask

from speechbrain.utils.fetching import fetch

class XEUS_Encoder(torch.nn.Module):
    """Multi-lingual SSL model trained on over 4k languages, from William Chen et al. at CMU

    Arguments
    ---------
    source : str or path
        Expects a filepath to a folder containing "model/config.yaml" and "model/xeus_checkpoint.pth"
        or a huggingface repo identifier, e.g. espnet/XEUS, which saves the model in `save_path`
    save_path : str or path
        If the huggingface repo is passed, model will be downloaded to this folder.
    freeze : bool, default False
        Whether to disallow parameter updates (True means no updates).
    output_all_hiddens : bool, default False
        Whether to return all hidden layer outputs as well as final output.
    """
    def __init__(self, source, save_path=None, freeze=True, output_all_hiddens=False):
        super().__init__()

        self.freeze = freeze
        self.output_all_hiddens = output_all_hiddens
        if os.path.exists(source):
            save_path = pathlib.Path(source)
        else:
            # Download from HF
            save_path = pathlib.Path(save_path)
            (save_path / "model").mkdir(parents=True, exist_ok=True)
            fetch(filename="model/config.yaml", source=source, savedir=save_path)
            fetch(filename="model/xeus_checkpoint.pth", source=source, savedir=save_path)

        # Build model object
        self.model, self.train_args = SSLTask.build_model_from_file(
            save_path / "model" / "config.yaml",
            save_path / "model" / "xeus_checkpoint.pth",
        )

        # Freeze params
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, wav, length):
        """Forward pass, handling lengths and frozen."""
        # Convert lengths to absolute
        abs_lens = (length * wav.size(1)).long()
        if self.freeze:
            with torch.no_grad():
                x, _, _ = self.model.encode(wav, abs_lens, use_mask=False, use_final_output=False)
        else:
            x, _, _ = self.model.encode(wav, abs_lens, use_mask=False, use_final_output=False)

        if self.output_all_hiddens:
            return x
        else:
            return x[-1]
