import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class ProteinSequenceTower(nn.Module):
    def __init__(self, model_name: str, dtype: torch.dtype = torch.float16, device: str = "cuda", **kwargs):
        super().__init__()
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = getattr(self.config, "hidden_size", None) or getattr(self.config, "d_model", None)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.to(device=self.device, dtype=self.dtype)

    @property
    def is_loaded(self):
        return hasattr(self, "model") and self.model is not None

    def load_model(self, device_map=None):
        if device_map:
            self.model.to(device_map)
        else:
            self.model.to(device=self.device, dtype=self.dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
