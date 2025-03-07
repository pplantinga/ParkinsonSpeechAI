import torch
import transformers


class LLM_Encoder(torch.nn.Module):
    def __init__(self, pretrained_source, freeze=True):
        super().__init__()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_source, trust_remote_code=True)
        self.model = transformers.AutoModel.from_pretrained(pretrained_source, output_hidden_states=False, trust_remote_code=True)
        self.freeze = freeze

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, tokens, attn_mask):
        if self.freeze:
            with torch.no_grad():
                result = self.model(tokens, attn_mask)
        else:
            result = self.model(tokens, attn_mask)

        return result.last_hidden_state

