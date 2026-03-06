import numpy as np
from similarity_encoder import load_similarity_matrix


def build_drug_sim_topk(drug_sim_file: str, n_drug: int, topk: int = 20):
    S = load_similarity_matrix(drug_sim_file)
    if S.shape[0] != n_drug:
        S = S[:n_drug, :n_drug]

    topk = int(min(topk, n_drug - 1))
    out = []
    for i in range(n_drug):
        row = S[i].copy()
        row[i] = -1e9
        idx = np.argpartition(-row, kth=topk)[:topk]
        idx = idx[np.argsort(-row[idx])]
        out.append([int(x) for x in idx])
    return out