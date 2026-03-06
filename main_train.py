# main_train.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score

from data import load_adjs, adj_to_edge_list, build_initial_features
from split import kfold_split_edges, remove_edges_from_adj, negative_sample

from similarity_encoder import load_similarity_matrix, build_similarity_adjs
from similarity_model import SimilarityEncoder

from metapath import MetaPathSampler, MetaPathEncoderInstance
from global_attention import concat_hetero_edges
from global_encoder import GlobalHeteroEncoder
from medpathex_model import MedPathEx
from projector import ProjectToDim
from sim_utils import build_drug_sim_topk


def train_one_fold(
    A_DG_np, A_DR_np, A_DiG_np,
    train_edges, test_edges,
    A_drug_sim, A_dis_sim, A_gene_sim,
    drug_sim_topk,
    epochs=3,
    dim=128,
    lr=1e-3,
    seed=42,
    K_inst=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_dis, n_drug = A_DR_np.shape
    n_gene = A_DG_np.shape[1]


    sim_drug = SimilarityEncoder(n_drug, dim).to(device)
    sim_dis  = SimilarityEncoder(n_dis,  dim).to(device)
    sim_gene = SimilarityEncoder(n_gene, dim).to(device)


    A_DR_train_np = remove_edges_from_adj(A_DR_np, test_edges)


    neg_train = negative_sample(A_DR_train_np, num_neg=len(train_edges), seed=seed)
    pos = train_edges
    neg = neg_train

    X_pairs = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], axis=0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_pairs))
    X_pairs = X_pairs[perm]
    y = y[perm]

    dis_idx  = torch.tensor(X_pairs[:, 0], dtype=torch.long, device=device)
    drug_idx = torch.tensor(X_pairs[:, 1], dtype=torch.long, device=device)
    labels   = torch.tensor(y, dtype=torch.float32, device=device)


    X_dis_np, X_drug_np, X_gene_np = build_initial_features(A_DG_np, A_DR_train_np, A_DiG_np)
    X_dis  = torch.tensor(X_dis_np, dtype=torch.float32, device=device)
    X_drug = torch.tensor(X_drug_np, dtype=torch.float32, device=device)
    X_gene = torch.tensor(X_gene_np, dtype=torch.float32, device=device)

    proj = ProjectToDim(
        in_dis=X_dis.shape[1],
        in_drug=X_drug.shape[1],
        in_gene=X_gene.shape[1],
        dim=dim
    ).to(device)


    X_dis_raw, X_drug_raw, X_gene_raw = X_dis, X_drug, X_gene


    sampler = MetaPathSampler(
        A_DR_np=A_DR_train_np,
        A_DiG_np=A_DiG_np,
        A_DG_np=A_DG_np,
        drug_sim_topk=drug_sim_topk,
        seed=seed
    )
    meta_enc = MetaPathEncoderInstance(dim=dim, n_heads=4).to(device)


    edge_index, n_dis2, n_drug2, n_gene2 = concat_hetero_edges(
        A_DR_train_np, A_DG_np, A_DiG_np, device=device
    )
    assert n_dis2 == n_dis and n_drug2 == n_drug and n_gene2 == n_gene

    global_enc = GlobalHeteroEncoder(
        in_dim=dim, hid_dim=dim, out_dim=dim, heads=4, n_layers=2, dropout=0.2
    ).to(device)


    model = MedPathEx(dim=dim).to(device)

    params = (
        list(model.parameters())
        + list(proj.parameters())
        + list(meta_enc.parameters())
        + list(global_enc.parameters())
        + list(sim_drug.parameters())
        + list(sim_dis.parameters())
        + list(sim_gene.parameters())
    )
    opt = Adam(params, lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    meta_enc.train()
    global_enc.train()
    proj.train()
    sim_drug.train(); sim_dis.train(); sim_gene.train()

    for ep in range(1, epochs + 1):
        opt.zero_grad()


        X_dis, X_drug, X_gene = proj(X_dis_raw, X_drug_raw, X_gene_raw)

        H_drug_meta, H_dis_meta, H_gene_meta, _ = meta_enc(
            X_drug, X_dis, X_gene, sampler,
            K_rdr=K_inst, K_rdgdr=K_inst, K_dgd=K_inst, K_drd=K_inst, K_drrd=K_inst, K_gdrdg=K_inst
        )

        X_all = torch.cat([X_dis, X_drug, X_gene], dim=0)
        H_all_global = global_enc(X_all, edge_index)

        H_dis_global  = H_all_global[:n_dis]
        H_drug_global = H_all_global[n_dis:n_dis+n_drug]
        H_gene_global = H_all_global[n_dis+n_drug:]

        H_drug_sim = sim_drug(A_drug_sim)
        H_dis_sim  = sim_dis(A_dis_sim)
        H_gene_sim = sim_gene(A_gene_sim)

        logits, extra = model(
            dis_idx, drug_idx,
            H_drug_sim, H_dis_sim, H_gene_sim,
            H_drug_meta, H_dis_meta, H_gene_meta,
            H_drug_global, H_dis_global, H_gene_global
        )

        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()

        lam_dis = extra["lam_dis"].detach().cpu().numpy()
        print(f"epoch {ep:02d} loss={loss.item():.4f}  lam_dis={lam_dis}")

    model.eval()
    meta_enc.eval()
    global_enc.eval()
    proj.eval()
    sim_drug.eval(); sim_dis.eval(); sim_gene.eval()

    with torch.no_grad():
        X_dis, X_drug, X_gene = proj(X_dis_raw, X_drug_raw, X_gene_raw)

        H_drug_meta, H_dis_meta, H_gene_meta, _ = meta_enc(
            X_drug, X_dis, X_gene, sampler,
            K_rdr=K_inst, K_rdgdr=K_inst, K_dgd=K_inst, K_drd=K_inst, K_drrd=K_inst, K_gdrdg=K_inst
        )

        X_all = torch.cat([X_dis, X_drug, X_gene], dim=0)
        H_all_global = global_enc(X_all, edge_index)
        H_dis_global  = H_all_global[:n_dis]
        H_drug_global = H_all_global[n_dis:n_dis+n_drug]
        H_gene_global = H_all_global[n_dis+n_drug:]

        H_drug_sim = sim_drug(A_drug_sim)
        H_dis_sim  = sim_dis(A_dis_sim)
        H_gene_sim = sim_gene(A_gene_sim)

        pos_test = test_edges
        neg_test = negative_sample(A_DR_np, num_neg=len(pos_test), seed=seed + 999)

        X_test = np.concatenate([pos_test, neg_test], axis=0)
        y_test = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))], axis=0)

        dis_t  = torch.tensor(X_test[:, 0], dtype=torch.long, device=device)
        drug_t = torch.tensor(X_test[:, 1], dtype=torch.long, device=device)

        logits_t, _ = model(
            dis_t, drug_t,
            H_drug_sim, H_dis_sim, H_gene_sim,
            H_drug_meta, H_dis_meta, H_gene_meta,
            H_drug_global, H_dis_global, H_gene_global
        )

        prob = torch.sigmoid(logits_t).detach().cpu().numpy().reshape(-1)
        auc = roc_auc_score(y_test, prob)
        aupr = average_precision_score(y_test, prob)

    return model, float(auc), float(aupr)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_DG_np, A_DR_np, A_DiG_np = load_adjs("adjacency_matrices")

    n_dis, n_drug = A_DR_np.shape
    n_gene = A_DG_np.shape[1]
    print("FULL n_dis, n_drug, n_gene =", n_dis, n_drug, n_gene)
    print("FULL DR edges =", int(A_DR_np.sum()))
    drug_sim_file = "drugmerged_matrix.xlsx"
    disease_sim_file = "diseasemerged_matrix.xlsx"
    gene_sim_file = "genemerged_matrix.xlsx"

    A_drug_sim, A_dis_sim, A_gene_sim = build_similarity_adjs(
        drug_file=drug_sim_file,
        disease_file=disease_sim_file,
        gene_file=gene_sim_file,
        n_drug=n_drug,
        n_dis=n_dis,
        n_gene=n_gene,
        device=str(device),
        strict_shape=True,
        clip_negative=False
    )
    print("A_DR", A_DR_np.shape, "edges", int(A_DR_np.sum()))
    print("A_DG", A_DG_np.shape, "edges", int(A_DG_np.sum()))
    print("A_DiG", A_DiG_np.shape, "edges", int(A_DiG_np.sum()))
    assert A_DR_np.shape[1] == A_DG_np.shape[0]
    assert A_DiG_np.shape[0] == A_DR_np.shape[0]
    assert A_DiG_np.shape[1] == A_DG_np.shape[1]
    drug_sim_topk = build_drug_sim_topk(drug_sim_file, n_drug=n_drug, topk=min(20, n_drug))

    dr_edges = adj_to_edge_list(A_DR_np)
    if len(dr_edges) < 10:
        raise ValueError(
            f"Too few DR edges after cropping: {len(dr_edges)}. "
            f"Increase n_dis_small/n_drug_small or change crop region."
        )

    #k = min(5, max(2, len(dr_edges) // 5))
    k=5
    folds = kfold_split_edges(dr_edges, k=k, seed=42)
    print("use k =", k, "num_edges =", len(dr_edges))

    aucs, auprs = [], []
    for i, (train_edges, test_edges) in enumerate(folds, start=1):
        print(f"\n========== Fold {i} ==========")
        _, auc, aupr = train_one_fold(
            A_DG_np, A_DR_np, A_DiG_np,
            train_edges, test_edges,
            A_drug_sim, A_dis_sim, A_gene_sim,
            drug_sim_topk=drug_sim_topk,
            epochs=2, dim=128, lr=1e-3, seed=42 + i,
            K_inst=3
        )
        aucs.append(auc)
        auprs.append(aupr)
        print(f"[Fold {i}] AUC={auc:.4f}  AUPR={aupr:.4f}")

    print("\n========== K-Fold Summary ==========")
    print(f"AUC : mean={np.mean(aucs):.4f}  std={np.std(aucs):.4f}")
    print(f"AUPR: mean={np.mean(auprs):.4f} std={np.std(auprs):.4f}")


if __name__ == "__main__":
    main()