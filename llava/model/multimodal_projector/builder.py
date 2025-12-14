import torch
import torch.nn as nn
import re
from typing import Optional


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    prefix_length = getattr(config, 'mm_projector_prefix_length', 1)
    use_norm = getattr(config, 'mm_projector_use_norm', False)

    if projector_type == 'linear':
        if prefix_length == 1:
            return nn.Linear(config.mm_hidden_size, config.hidden_size)
        return PrefixProjector(config.mm_hidden_size, config.hidden_size, prefix_length, projector_type, use_norm)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        if prefix_length == 1:
            return nn.Sequential(*modules)
        return PrefixProjector(config.mm_hidden_size, config.hidden_size, prefix_length, 'mlp', use_norm, modules)

    if projector_type in {'residual_mlp', 'residual'}:
        return PrefixProjector(config.mm_hidden_size, config.hidden_size, prefix_length, projector_type, use_norm)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


class PrefixProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, prefix_length: int,
                 projector_type: str, use_norm: bool = False, preset_layers: Optional[nn.Module] = None):
        super().__init__()
        self.prefix_length = prefix_length
        self.projector_type = projector_type
        self.use_norm = use_norm
        hidden_layers = []
        if preset_layers is not None:
            hidden_layers.extend(preset_layers)
        elif projector_type in {'residual_mlp', 'residual'}:
            hidden_layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim)
            ])
        else:
            hidden_layers.append(nn.Linear(input_dim, output_dim * prefix_length))

        self.network = nn.Sequential(*hidden_layers) if len(hidden_layers) > 1 else hidden_layers[0]
        self.norm = nn.LayerNorm(output_dim) if use_norm else None

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if hidden_states.dim() == 3:
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                mask = attention_mask.float().unsqueeze(-1)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = hidden_states

        outputs = self.network(pooled)
        if self.projector_type in {'residual_mlp', 'residual'}:
            outputs = outputs + pooled
        if self.norm is not None:
            outputs = self.norm(outputs)
        outputs = outputs.view(outputs.size(0), self.prefix_length, -1)
        return outputs
