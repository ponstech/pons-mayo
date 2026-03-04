"""
Multi-feature fusion for PONS-CLIP.

Fuses 3 image embeddings (b-mode, enhanced, improved) from a *frozen* vision
encoder into a single 512-d embedding. Only the fusion module is trainable.
"""

import torch
import torch.nn as nn

from .constants import PROJ_DIM, NUM_IMAGE_FEATURES


class FusionModule(nn.Module):
    """
    Fuses NUM_IMAGE_FEATURES (3) image embeddings into one PROJ_DIM vector.

    Options:
    - "weighted": learned scalar weights per modality, then normalize (no extra params beyond 3 scalars).
    - "mlp": concat 3*PROJ_DIM -> hidden -> PROJ_DIM.
    - "attention": learnable query, attention over the 3 embeddings, then project.
    """

    def __init__(
        self,
        dim: int = PROJ_DIM,
        num_features: int = NUM_IMAGE_FEATURES,
        fusion_type: str = "weighted",
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.fusion_type = fusion_type.lower()

        if self.fusion_type == "weighted":
            # Learnable weights per modality (positive, normalized to sum to 1 or use softmax)
            self.weights = nn.Parameter(torch.ones(num_features) / num_features)
        elif self.fusion_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(num_features * dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, dim),
            )
        elif self.fusion_type == "attention":
            self.query = nn.Parameter(torch.randn(1, 1, dim))
            self.scale = dim ** -0.5
            self.proj = nn.Linear(dim, dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, num_features, dim) — e.g. (B, 3, 512).

        Returns:
            fused: (batch_size, dim), L2-normalized.
        """
        if self.fusion_type == "weighted":
            w = torch.softmax(self.weights, dim=0)
            w = w.view(1, -1, 1).to(embeddings.device)
            fused = (embeddings * w).sum(dim=1)
        elif self.fusion_type == "mlp":
            B = embeddings.size(0)
            fused = self.mlp(embeddings.view(B, -1))
        elif self.fusion_type == "attention":
            # embeddings: (B, 3, dim); query: (1, 1, dim)
            q = self.query.to(embeddings.device)
            attn = torch.matmul(q, embeddings.transpose(1, 2)) * self.scale  # (1, 1, 3)
            attn = attn.softmax(dim=-1)
            fused = (attn @ embeddings).squeeze(1)
            fused = self.proj(fused)
        else:
            fused = embeddings.mean(dim=1)

        return fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)
