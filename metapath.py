import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class MetaPathSampler:


    def __init__(self, A_DR_np, A_DiG_np, A_DG_np=None, drug_sim_topk=None, seed=42):
        self.rng = np.random.default_rng(seed)
        self.A_DR = A_DR_np
        self.A_DiG = A_DiG_np
        self.A_DG = A_DG_np
        self.drug_sim_topk = drug_sim_topk
        self.dis2drug = [np.flatnonzero(A_DR_np[i]) for i in range(A_DR_np.shape[0])]
        self.drug2dis = [np.flatnonzero(A_DR_np[:, j]) for j in range(A_DR_np.shape[1])]
        self.dis2gene = [np.flatnonzero(A_DiG_np[i]) for i in range(A_DiG_np.shape[0])]
        self.gene2dis = [np.flatnonzero(A_DiG_np[:, g]) for g in range(A_DiG_np.shape[1])]

    def _choice(self, arr):
        if len(arr) == 0:
            return None
        return int(arr[self.rng.integers(0, len(arr))])


    def sample_RDR(self, drug_id, num_instances=20):
        inst = []
        for _ in range(num_instances):
            d = self._choice(self.drug2dis[drug_id])
            if d is None:
                continue
            r2 = self._choice(self.dis2drug[d])
            if r2 is None:
                continue
            inst.append((drug_id, d, r2))
        return inst


    def sample_RDGDR(self, drug_id, num_instances=20):
        inst = []
        for _ in range(num_instances):
            d1 = self._choice(self.drug2dis[drug_id])
            if d1 is None:
                continue
            g = self._choice(self.dis2gene[d1])
            if g is None:
                continue
            d2 = self._choice(self.gene2dis[g])
            if d2 is None:
                continue
            r2 = self._choice(self.dis2drug[d2])
            if r2 is None:
                continue
            inst.append((drug_id, d1, g, d2, r2))
        return inst

    def sample_DGD(self, dis_id, num_instances=20):
        inst = []
        for _ in range(num_instances):
            g = self._choice(self.dis2gene[dis_id])
            if g is None:
                continue
            d2 = self._choice(self.gene2dis[g])
            if d2 is None:
                continue
            inst.append((dis_id, g, d2))
        return inst


    def sample_DRD(self, dis_id, num_instances=20):
        inst = []
        for _ in range(num_instances):
            r = self._choice(self.dis2drug[dis_id])
            if r is None:
                continue
            d2 = self._choice(self.drug2dis[r])
            if d2 is None:
                continue
            inst.append((dis_id, r, d2))
        return inst
    def sample_DRRD(self, dis_id, num_instances=20):
        inst = []
        for _ in range(num_instances):
            r1 = self._choice(self.dis2drug[dis_id])
            if r1 is None:
                continue

            sim_list = None
            if self.drug_sim_topk is not None:
                sim_list = self.drug_sim_topk[r1] if isinstance(self.drug_sim_topk, list) else self.drug_sim_topk.get(r1, [])
            if not sim_list:

                sim_list = self.dis2drug[dis_id]

            r2 = self._choice(sim_list)
            if r2 is None:
                continue
            d2 = self._choice(self.drug2dis[r2])
            if d2 is None:
                continue
            inst.append((dis_id, r1, r2, d2))
        return inst


    def sample_GDRDG(self, gene_id, num_instances=20):
        inst = []
        for _ in range(num_instances):
            d1 = self._choice(self.gene2dis[gene_id])
            if d1 is None:
                continue
            r = self._choice(self.dis2drug[d1])
            if r is None:
                continue
            d2 = self._choice(self.drug2dis[r])
            if d2 is None:
                continue
            g2 = self._choice(self.dis2gene[d2])
            if g2 is None:
                continue
            inst.append((gene_id, d1, r, d2, g2))
        return inst


class InstanceAttn(nn.Module):

    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.W = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_heads)])
        self.a = nn.ParameterList([nn.Parameter(torch.randn(dim) * 0.01) for _ in range(n_heads)])
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h_i, inst_vecs):
        if inst_vecs.numel() == 0:
            return torch.zeros_like(h_i)
        outs = []
        for k in range(self.n_heads):

            hi = self.W[k](h_i)
            hv = self.W[k](inst_vecs)
            e = self.leaky((hv + hi) @ self.a[k])
            alpha = torch.softmax(e, dim=0)
            out = (alpha.unsqueeze(1) * hv).sum(dim=0)
            outs.append(out)
        return torch.mean(torch.stack(outs, dim=0), dim=0)

class MetaPathLevelAttn(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.q = nn.Parameter(torch.randn(dim) * 0.01)

    def forward(self, H_list):
        sem = []
        for H in H_list:
            s = torch.tanh(self.W(H)).mean(dim=0)  # (dim,)
            sem.append(s)
        sem = torch.stack(sem, dim=0)              # (P,dim)


        score = (torch.tanh(sem) * self.q).sum(dim=1)  # (P,)
        beta = torch.softmax(score, dim=0)             # (P,)

        H = 0
        for b, h in zip(beta, H_list):
            H = H + b * h
        return H, beta

class MetaPathEncoderInstance(nn.Module):

    def __init__(self, dim, n_heads=4):
        super().__init__()

        self.att_RDR   = InstanceAttn(dim, n_heads)
        self.att_RDGDR = InstanceAttn(dim, n_heads)
        self.att_DGD   = InstanceAttn(dim, n_heads)
        self.att_DRD   = InstanceAttn(dim, n_heads)
        self.att_DRRD  = InstanceAttn(dim, n_heads)
        self.att_GDRDG = InstanceAttn(dim, n_heads)
        self.mp_att_drug = MetaPathLevelAttn(dim)
        self.mp_att_dis  = MetaPathLevelAttn(dim)
        self.mp_att_gene = MetaPathLevelAttn(dim)

    @staticmethod
    def _max_pool_instance(node_feats):

        M = torch.stack(node_feats, dim=0)
        return torch.max(M, dim=0).values

    def forward(self, X_drug, X_dis, X_gene, sampler,
                K_rdr=20, K_rdgdr=20, K_dgd=20, K_drd=20, K_drrd=20, K_gdrdg=20):
        device = X_drug.device
        dim = X_drug.shape[1]


        H_rdr = torch.zeros((X_drug.shape[0], dim), device=device)
        H_rdgdr = torch.zeros((X_drug.shape[0], dim), device=device)

        for r in range(X_drug.shape[0]):
            h_i = X_drug[r]


            insts = sampler.sample_RDR(r, num_instances=K_rdr)
            inst_vecs = []
            for (r0, d, r2) in insts:
                v = self._max_pool_instance([X_drug[r0], X_dis[d], X_drug[r2]])
                inst_vecs.append(v)
            inst_vecs = torch.stack(inst_vecs, dim=0) if inst_vecs else torch.empty((0, dim), device=device)
            H_rdr[r] = self.att_RDR(h_i, inst_vecs)


            insts = sampler.sample_RDGDR(r, num_instances=K_rdgdr)
            inst_vecs = []
            for (r0, d1, g, d2, r2) in insts:
                v = self._max_pool_instance([X_drug[r0], X_dis[d1], X_gene[g], X_dis[d2], X_drug[r2]])
                inst_vecs.append(v)
            inst_vecs = torch.stack(inst_vecs, dim=0) if inst_vecs else torch.empty((0, dim), device=device)
            H_rdgdr[r] = self.att_RDGDR(h_i, inst_vecs)

        H_drug_meta, beta_drug = self.mp_att_drug([H_rdr, H_rdgdr])


        H_dgd  = torch.zeros((X_dis.shape[0], dim), device=device)
        H_drd  = torch.zeros((X_dis.shape[0], dim), device=device)
        H_drrd = torch.zeros((X_dis.shape[0], dim), device=device)

        for d in range(X_dis.shape[0]):
            h_i = X_dis[d]

            insts = sampler.sample_DGD(d, num_instances=K_dgd)
            inst_vecs = []
            for (d0, g, d2) in insts:
                v = self._max_pool_instance([X_dis[d0], X_gene[g], X_dis[d2]])
                inst_vecs.append(v)
            inst_vecs = torch.stack(inst_vecs, dim=0) if inst_vecs else torch.empty((0, dim), device=device)
            H_dgd[d] = self.att_DGD(h_i, inst_vecs)

            insts = sampler.sample_DRD(d, num_instances=K_drd)
            inst_vecs = []
            for (d0, r, d2) in insts:
                v = self._max_pool_instance([X_dis[d0], X_drug[r], X_dis[d2]])
                inst_vecs.append(v)
            inst_vecs = torch.stack(inst_vecs, dim=0) if inst_vecs else torch.empty((0, dim), device=device)
            H_drd[d] = self.att_DRD(h_i, inst_vecs)

            insts = sampler.sample_DRRD(d, num_instances=K_drrd)
            inst_vecs = []
            for (d0, r1, r2, d2) in insts:
                v = self._max_pool_instance([X_dis[d0], X_drug[r1], X_drug[r2], X_dis[d2]])
                inst_vecs.append(v)
            inst_vecs = torch.stack(inst_vecs, dim=0) if inst_vecs else torch.empty((0, dim), device=device)
            H_drrd[d] = self.att_DRRD(h_i, inst_vecs)

        H_dis_meta, beta_dis = self.mp_att_dis([H_dgd, H_drd, H_drrd])


        H_gdrdg = torch.zeros((X_gene.shape[0], dim), device=device)
        for g in range(X_gene.shape[0]):
            h_i = X_gene[g]
            insts = sampler.sample_GDRDG(g, num_instances=K_gdrdg)
            inst_vecs = []
            for (g0, d1, r, d2, g2) in insts:
                v = self._max_pool_instance([X_gene[g0], X_dis[d1], X_drug[r], X_dis[d2], X_gene[g2]])
                inst_vecs.append(v)
            inst_vecs = torch.stack(inst_vecs, dim=0) if inst_vecs else torch.empty((0, dim), device=device)
            H_gdrdg[g] = self.att_GDRDG(h_i, inst_vecs)

        H_gene_meta, beta_gene = self.mp_att_gene([H_gdrdg])

        extra = {
            "beta_drug": beta_drug.detach().cpu(),
            "beta_dis": beta_dis.detach().cpu(),
            "beta_gene": beta_gene.detach().cpu(),
        }
        return H_drug_meta, H_dis_meta, H_gene_meta, extra