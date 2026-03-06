import numpy as np

def load_adjs(adj_dir="adjacency_matrices"):
    A_DG  = np.load(f"{adj_dir}/A_DrugGene.npy")
    A_DR  = np.load(f"{adj_dir}/A_DiseaseDrug.npy")
    A_DiG = np.load(f"{adj_dir}/A_DiseaseGene.npy")
    return A_DG, A_DR, A_DiG

def build_initial_features(A_DG, A_DR, A_DiG):

    X_dis  = np.concatenate([A_DR, A_DiG], axis=1)
    X_drug = np.concatenate([A_DR.T, A_DG], axis=1)
    X_gene = np.concatenate([A_DG.T, A_DiG.T], axis=1)
    return X_dis, X_drug, X_gene

def adj_to_edge_list(A):

    rows, cols = np.nonzero(A)
    return np.stack([rows, cols], axis=1)