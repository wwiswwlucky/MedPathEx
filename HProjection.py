import numpy as np
import torch
import torch.nn as nn

def build_node_features(disease_drug_mat, disease_gene_mat, gene_drug_mat):
  
    n_d, n_r = disease_drug_mat.shape
    # disease_gene_mat.shape = (n_d, n_g)
    # gene_drug_mat.shape    = (n_g, n_r)
    _, n_g = disease_gene_mat.shape
    
    # (1) 疾病节点特征: concat(疾病-药物, 疾病-基因)
    disease_feat = np.concatenate([disease_drug_mat, disease_gene_mat], axis=1)
    # shape = (n_d, n_r + n_g)

    # (2) 药物节点特征: 需要先转置 disease_drug_mat => (n_r, n_d)， gene_drug_mat => (n_r, n_g)
    drug_disease_mat = disease_drug_mat.T    # shape=(n_r, n_d)
    drug_gene_mat    = gene_drug_mat.T       # shape=(n_r, n_g)
    drug_feat = np.concatenate([drug_disease_mat, drug_gene_mat], axis=1)
    # shape = (n_r, n_d + n_g)

    # (3) 基因节点特征: disease_gene_mat => (n_d, n_g)，转置得到 (n_g, n_d)
    #                  gene_drug_mat => (n_g, n_r)
    gene_disease_mat = disease_gene_mat.T    # shape=(n_g, n_d)
    # gene_drug_mat 已是 (n_g, n_r)
    gene_feat = np.concatenate([gene_disease_mat, gene_drug_mat], axis=1)
    # shape = (n_g, n_d + n_r)
    
    return disease_feat, drug_feat, gene_feat


class HeteroProjection(nn.Module):
    """
    针对疾病(disease), 药物(drug), 基因(gene) 三类节点，
    各自使用一个独立的可学习线性映射 W_v，将其特征映射到同样的 embed_dim 维空间。
    """
    def __init__(self, in_dim_d, in_dim_r, in_dim_g, embed_dim):
        super(HeteroProjection, self).__init__()
        # 这里分别定义三套 Linear
        # weight shape = [embed_dim, in_dim_*]
        self.disease_proj = nn.Linear(in_dim_d, embed_dim, bias=False)
        self.drug_proj    = nn.Linear(in_dim_r, embed_dim, bias=False)
        self.gene_proj    = nn.Linear(in_dim_g, embed_dim, bias=False)

        # 可根据需要选择初始化方式
        nn.init.xavier_uniform_(self.disease_proj.weight)
        nn.init.xavier_uniform_(self.drug_proj.weight)
        nn.init.xavier_uniform_(self.gene_proj.weight)

    def forward(self, disease_feat, drug_feat, gene_feat):
     
        h_disease = self.disease_proj(disease_feat)
        h_drug    = self.drug_proj(drug_feat)
        h_gene    = self.gene_proj(gene_feat)
        return h_disease, h_drug, h_gene


if __name__ == '__main__':
  
    n_d, n_r, n_g = 4, 5, 6
    disease_drug_mat = np.random.randint(0, 2, size=(n_d, n_r))  # shape = (4, 5)
    disease_gene_mat = np.random.randint(0, 2, size=(n_d, n_g))  # shape = (4, 6)
    gene_drug_mat    = np.random.randint(0, 2, size=(n_g, n_r))  # shape = (6, 5)

    print('Disease-Drug shape:', disease_drug_mat.shape)
    print('Disease-Gene shape:', disease_gene_mat.shape)
    print('Gene-Drug shape:   ', gene_drug_mat.shape)

    # 1) 构建初始特征
    disease_feat, drug_feat, gene_feat = build_node_features(disease_drug_mat, disease_gene_mat, gene_drug_mat)
    print('disease_feat shape =', disease_feat.shape)  # (n_d, n_r + n_g)
    print('drug_feat shape    =', drug_feat.shape)     # (n_r, n_d + n_g)
    print('gene_feat shape    =', gene_feat.shape)     # (n_g, n_d + n_r)

    # 转为 Torch Tensor
    disease_feat_t = torch.tensor(disease_feat, dtype=torch.float32)
    drug_feat_t    = torch.tensor(drug_feat,    dtype=torch.float32)
    gene_feat_t    = torch.tensor(gene_feat,    dtype=torch.float32)

    # 2) 线性投影
    #    这里 in_dim_* 分别对应 (n_r+n_g), (n_d+n_g), (n_d+n_r)
    proj_model = HeteroProjection(
        in_dim_d=disease_feat.shape[1], 
        in_dim_r=drug_feat.shape[1], 
        in_dim_g=gene_feat.shape[1],
        embed_dim=8  # 目标空间维度
    )

    h_disease, h_drug, h_gene = proj_model(disease_feat_t, drug_feat_t, gene_feat_t)
    print('Projected disease:', h_disease.shape)  # [n_d, embed_dim]
    print('Projected drug:   ', h_drug.shape)     # [n_r, embed_dim]
    print('Projected gene:   ', h_gene.shape)     # [n_g, embed_dim]
