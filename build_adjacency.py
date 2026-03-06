import pandas as pd
import numpy as np
from pathlib import Path


def read_edge_list(file_path):
    df = pd.read_excel(file_path, header=None)
    df = df.dropna(how="any")
    edges = df.values.astype(np.int64)

    edges -= 1


    if (edges < 0).any():
        raise ValueError(
            f"[read_edge_list] Found 0/negative ids after 1-based->0-based in {file_path}. "
            f"Check your Excel ids start from 1."
        )

    return edges


def infer_node_size(edges):
    src_max = int(edges[:, 0].max()) + 1
    dst_max = int(edges[:, 1].max()) + 1
    return src_max, dst_max


def build_bipartite_adj(edges, num_src, num_dst):
    A = np.zeros((num_src, num_dst), dtype=np.bool_)

    if int(edges[:, 0].max()) >= num_src or int(edges[:, 1].max()) >= num_dst:
        raise ValueError(
            f"[build_bipartite_adj] Edge index out of range: "
            f"max_src={edges[:,0].max()} max_dst={edges[:,1].max()} vs shape=({num_src},{num_dst})"
        )

    for s, d in edges:
        A[int(s), int(d)] = True

    return A


def build_all_adjacency(
        drug_gene_file,
        disease_drug_file,
        disease_gene_file,
        save_dir="adjacency_matrices"):
    Path(save_dir).mkdir(exist_ok=True)
    dg_edges = read_edge_list(drug_gene_file)
    drug_num_dg, gene_num_dg = infer_node_size(dg_edges)
    print(f"DG inferred: drug_num={drug_num_dg}, gene_num={gene_num_dg}")

    dr_edges = read_edge_list(disease_drug_file)
    disease_num_dr, drug_num_dr = infer_node_size(dr_edges)
    print(f"DR inferred: disease_num={disease_num_dr}, drug_num={drug_num_dr}")

    dig_edges = read_edge_list(disease_gene_file)
    disease_num_dig, gene_num_dig = infer_node_size(dig_edges)
    print(f"DiG inferred: disease_num={disease_num_dig}, gene_num={gene_num_dig}")


    drug_num = max(drug_num_dg, drug_num_dr)
    disease_num = max(disease_num_dr, disease_num_dig)
    gene_num = max(gene_num_dg, gene_num_dig)

    print("\n========== 4. Global Node Size ==========")
    print(f"Global drug_num={drug_num}")
    print(f"Global disease_num={disease_num}")
    print(f"Global gene_num={gene_num}")

    if drug_num_dg != drug_num_dr:
        print(
            f"[WARN] Drug count mismatch is allowed: "
            f"DG={drug_num_dg}, DR={drug_num_dr}, use global drug_num={drug_num}"
        )
    if disease_num_dr != disease_num_dig:
        print(
            f"[WARN] Disease count mismatch is allowed: "
            f"DR={disease_num_dr}, DiG={disease_num_dig}, use global disease_num={disease_num}"
        )
    if gene_num_dg != gene_num_dig:
        print(
            f"[WARN] Gene count mismatch is allowed: "
            f"DG={gene_num_dg}, DiG={gene_num_dig}, use global gene_num={gene_num}"
        )

    A_DG = build_bipartite_adj(dg_edges, drug_num, gene_num)


    A_DR = build_bipartite_adj(dr_edges, disease_num, drug_num)

    A_DiG = build_bipartite_adj(dig_edges, disease_num, gene_num)


    np.save(f"{save_dir}/A_DrugGene.npy", A_DG)
    np.save(f"{save_dir}/A_DiseaseDrug.npy", A_DR)
    np.save(f"{save_dir}/A_DiseaseGene.npy", A_DiG)

    print("\n========== Saved ==========")
    print(f"Drug-Gene matrix shape: {A_DG.shape}, edges={int(A_DG.sum())}")
    print(f"Disease-Drug matrix shape: {A_DR.shape}, edges={int(A_DR.sum())}")
    print(f"Disease-Gene matrix shape: {A_DiG.shape}, edges={int(A_DiG.sum())}")
    
    return A_DG, A_DR, A_DiG
if __name__ == "__main__":
    build_all_adjacency(
        "Association of drug gene numbers.xlsx",
        "Disease drug serial number association.xlsx",
        "Disease gene number association.xlsx",
        "adjacency_matrices"
    )
