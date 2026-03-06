import torch
import torch.nn as nn

from feature_fusion import ChannelFusion


class MedPathEx(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.fuse_drug = ChannelFusion(dim)
        self.fuse_dis  = ChannelFusion(dim)
        self.fuse_gene = ChannelFusion(dim)

    def forward(
        self,
        dis_idx,
        drug_idx,
        H_drug_sim, H_dis_sim, H_gene_sim,
        H_drug_meta, H_dis_meta, H_gene_meta,
        H_drug_global, H_dis_global, H_gene_global
    ):


        Z_drug, lam_drug = self.fuse_drug(H_drug_meta, H_drug_global, H_drug_sim)
        Z_dis,  lam_dis  = self.fuse_dis (H_dis_meta,  H_dis_global,  H_dis_sim)
        Z_gene, lam_gene = self.fuse_gene(H_gene_meta, H_gene_global, H_gene_sim)

        z_r = Z_drug[drug_idx]
        z_d = Z_dis[dis_idx]

        logits = (z_r * z_d).sum(dim=1)

        extra = {
            "lam_drug": lam_drug.detach().cpu(),
            "lam_dis": lam_dis.detach().cpu(),
            "lam_gene": lam_gene.detach().cpu(),
        }

        return logits, extra