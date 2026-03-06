import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelFusion(nn.Module):


    def __init__(self, dim):
        super().__init__()
        self.lam_logits = nn.Parameter(torch.zeros(3))
        self.proj = nn.Linear(dim, dim)

    def forward(self, H_meta, H_global, H_sim):
        lam = torch.softmax(self.lam_logits, dim=0)
        H = lam[0] * H_meta + lam[1] * H_global + lam[2] * H_sim
        Z = F.relu(self.proj(H))

        return Z, lam