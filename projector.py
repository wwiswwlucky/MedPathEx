import torch
import torch.nn as nn


class ProjectToDim(nn.Module):
    def __init__(self, in_dis: int, in_drug: int, in_gene: int, dim: int):
        super().__init__()
        self.proj_dis = nn.Linear(in_dis, dim, bias=True)
        self.proj_drug = nn.Linear(in_drug, dim, bias=True)
        self.proj_gene = nn.Linear(in_gene, dim, bias=True)

    def forward(self, X_dis, X_drug, X_gene):
        return self.proj_dis(X_dis), self.proj_drug(X_drug), self.proj_gene(X_gene)