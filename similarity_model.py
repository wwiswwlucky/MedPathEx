# similarity_model.py
import torch
import torch.nn as nn

class SimilarityEncoder(nn.Module):
    def __init__(self, n_nodes: int, out_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_nodes, out_dim) * 0.01)

    def forward(self, A_norm: torch.Tensor) -> torch.Tensor:
        return A_norm @ self.W