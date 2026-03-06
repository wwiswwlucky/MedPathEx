# similarity_encoder.py
import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


def _to_torch_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def load_similarity_matrix(file_path: str) -> np.ndarray:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".npy":
        S = np.load(file_path).astype(np.float32)
    elif ext in [".xls", ".xlsx"]:
        S = pd.read_excel(file_path, header=None).values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .npy/.xls/.xlsx")

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"{file_path} is not a square matrix: {S.shape}")

    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    return S


def normalize_adj_dense(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = A.astype(np.float32)
    deg = A.sum(axis=1).astype(np.float32)

    deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
    mask = deg > 0
    deg_inv_sqrt[mask] = np.power(deg[mask] + eps, -0.5)

    D = np.diag(deg_inv_sqrt)
    return (D @ A @ D).astype(np.float32)


def build_similarity_adjs(
    drug_file: str,
    disease_file: str,
    gene_file: str,
    n_drug: int,
    n_dis: int,
    n_gene: int,
    device: Optional[Union[str, torch.device]] = None,
    strict_shape: bool = True,
    clip_negative: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device_t = _to_torch_device(device)

    S_drug = load_similarity_matrix(drug_file)
    S_dis = load_similarity_matrix(disease_file)
    S_gene = load_similarity_matrix(gene_file)

    if strict_shape:
        if S_drug.shape[0] != n_drug:
            raise ValueError(f"Drug sim shape {S_drug.shape} != expected ({n_drug},{n_drug})")
        if S_dis.shape[0] != n_dis:
            raise ValueError(f"Disease sim shape {S_dis.shape} != expected ({n_dis},{n_dis})")
        if S_gene.shape[0] != n_gene:
            raise ValueError(f"Gene sim shape {S_gene.shape} != expected ({n_gene},{n_gene})")
    else:
        # permissive mode: crop to expected
        S_drug = S_drug[:n_drug, :n_drug]
        S_dis = S_dis[:n_dis, :n_dis]
        S_gene = S_gene[:n_gene, :n_gene]

    if clip_negative:
        S_drug[S_drug < 0] = 0.0
        S_dis[S_dis < 0] = 0.0
        S_gene[S_gene < 0] = 0.0

    A_drug = torch.tensor(normalize_adj_dense(S_drug), dtype=torch.float32, device=device_t)
    A_dis = torch.tensor(normalize_adj_dense(S_dis), dtype=torch.float32, device=device_t)
    A_gene = torch.tensor(normalize_adj_dense(S_gene), dtype=torch.float32, device=device_t)

    return A_drug, A_dis, A_gene