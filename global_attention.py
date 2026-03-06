import torch
import torch.nn as nn
import torch.nn.functional as F

def build_edge_index_from_bipartite(A_np, row_offset, col_offset, device):
    import numpy as np
    r, c = np.nonzero(A_np)
    src = torch.tensor(r + row_offset, dtype=torch.long, device=device)
    dst = torch.tensor(c + col_offset, dtype=torch.long, device=device)
    return torch.stack([src, dst], dim=0)

def concat_hetero_edges(A_DR_np, A_DG_np, A_DiG_np, device):
    assert A_DR_np.shape[1] == A_DG_np.shape[0], "DR drug dim != DG drug dim"
    assert A_DiG_np.shape[0] == A_DR_np.shape[0], "DiG disease dim != DR disease dim"
    assert A_DiG_np.shape[1] == A_DG_np.shape[1], "DiG gene dim != DG gene dim"
    n_dis, n_drug = A_DR_np.shape
    n_gene = A_DG_np.shape[1]

    e1 = build_edge_index_from_bipartite(A_DR_np, 0, n_dis, device)
    e2 = build_edge_index_from_bipartite(A_DG_np, n_dis, n_dis + n_drug, device)
    e3 = build_edge_index_from_bipartite(A_DiG_np, 0, n_dis + n_drug, device)

    edge_index = torch.cat([e1, e2, e3], dim=1)
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    return edge_index, n_dis, n_drug, n_gene

def segment_softmax(src, index, num_nodes):
    max_per = torch.full((num_nodes,), -1e15, device=src.device, dtype=src.dtype)
    max_per.scatter_reduce_(0, index, src, reduce="amax", include_self=True)

    exp = torch.exp(src - max_per[index])
    sum_per = torch.zeros((num_nodes,), device=src.device, dtype=src.dtype)
    sum_per.scatter_add_(0, index, exp)
    return exp / (sum_per[index] + 1e-12)

class GlobalGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0, negative_slope=0.2):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, out_dim) * 0.01)
        self.a_dst = nn.Parameter(torch.randn(heads, out_dim) * 0.01)

    def forward(self, x, edge_index):
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        h = self.W(x).view(N, self.heads, self.out_dim)
        h_src = h[src]
        h_dst = h[dst]

        e = (h_src * self.a_src).sum(dim=-1) + (h_dst * self.a_dst).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        alphas = []
        for k in range(self.heads):
            alphas.append(segment_softmax(e[:, k], dst, N))
        alpha = torch.stack(alphas, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros((N, self.heads, self.out_dim), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * h_src)
        return out.mean(dim=1)