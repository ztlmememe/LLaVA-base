import torch
import torch.nn as nn


class ProteinProjector(nn.Module):
    """Map protein encoder representations to LLM prefix embeddings.

    The projector mirrors the multimodal MLP adapter used in LLaVA but
    operates on 1D protein embeddings. It produces a configurable number of
    prefix tokens that can be concatenated with LLM token embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        prefix_length: int = 8,
        projector_type: str = "linear",
    ) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.projector_type = projector_type

        if projector_type == "linear":
            self.proj = nn.Linear(input_dim, prefix_length * output_dim)
        elif projector_type.startswith("mlp"):
            # mlp<depth>x_gelu (e.g., mlp2x_gelu)
            depth = int(projector_type.split("mlp")[-1].split("x")[0] or 2)
            hidden = output_dim
            layers = [nn.Linear(input_dim, hidden), nn.GELU()]
            for _ in range(depth - 1):
                layers.extend([nn.Linear(hidden, hidden), nn.GELU()])
            layers.append(nn.Linear(hidden, prefix_length * output_dim))
            self.proj = nn.Sequential(*layers)
        elif projector_type == "residual":
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
            self.prefix_expander = nn.Linear(output_dim, prefix_length * output_dim)
        else:
            raise ValueError(f"Unsupported projector type: {projector_type}")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project pooled protein features to prefix embeddings.

        Args:
            hidden: Tensor of shape (batch, input_dim)

        Returns:
            Tensor of shape (batch, prefix_length, output_dim)
        """
        if self.projector_type == "residual":
            hidden = hidden + self.proj(hidden)
            projected = self.prefix_expander(hidden)
        else:
            projected = self.proj(hidden)
        batch, _ = projected.shape
        return projected.view(batch, self.prefix_length, -1)


__all__ = ["ProteinProjector"]
