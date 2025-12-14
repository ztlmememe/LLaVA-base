from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class ProteinEncoderConfig:
    model_name: str
    max_length: int = 4096
    trust_remote_code: bool = True


class ProteinSequenceEncoder(torch.nn.Module):
    """Lightweight wrapper around Hugging Face protein encoders.

    The encoder outputs a pooled representation (mean over tokens) that can be
    consumed by a projector. Gradients can optionally flow to the encoder when
    joint finetuning is enabled.
    """

    def __init__(self, config: ProteinEncoderConfig):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code
        )
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Protein encoder config must expose hidden_size/d_model")
        self.hidden_size = hidden_size

    @torch.inference_mode()
    def embed(self, sequences: list[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """Return pooled embeddings without tracking gradients."""
        return self(sequences, device=device, require_grad=False)

    def forward(
        self, sequences: list[str], device: Optional[torch.device] = None, require_grad: bool = True
    ) -> torch.Tensor:
        tokenized = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        with torch.set_grad_enabled(require_grad):
            outputs = self.model(**tokenized)
            hidden = outputs.last_hidden_state
            pooled = hidden.mean(dim=1)
        return pooled


def build_protein_encoder(model_name: str, max_length: int = 4096) -> ProteinSequenceEncoder:
    cfg = ProteinEncoderConfig(model_name=model_name, max_length=max_length)
    return ProteinSequenceEncoder(cfg)


__all__ = ["ProteinEncoderConfig", "ProteinSequenceEncoder", "build_protein_encoder"]
