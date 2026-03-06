import torch
import torch.nn as nn
import torch.nn.functional as F
from global_attention import GlobalGATLayer

class GlobalHeteroEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim=128, out_dim=128, heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(GlobalGATLayer(in_dim, out_dim, heads=heads, dropout=dropout))
        else:
            self.layers.append(GlobalGATLayer(in_dim, hid_dim, heads=heads, dropout=dropout))
            for _ in range(n_layers - 2):
                self.layers.append(GlobalGATLayer(hid_dim, hid_dim, heads=heads, dropout=dropout))
            self.layers.append(GlobalGATLayer(hid_dim, out_dim, heads=heads, dropout=dropout))
        self.dropout = dropout

    def forward(self, X_all, edge_index):
        H = X_all
        for layer in self.layers:
            H = layer(H, edge_index)
            H = F.elu(H)
            H = F.dropout(H, p=self.dropout, training=self.training)
        return H