import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
  
    def __init__(self, d_in, d_out):
        super().__init__()
        # 可学习参数 λ1, λ2, λ3, 初始化为 1/3 或其他
        self.lambda1 = nn.Parameter(torch.tensor(1.0))
        self.lambda2 = nn.Parameter(torch.tensor(1.0))
        self.lambda3 = nn.Parameter(torch.tensor(1.0))

        # 最终线性变换: W0 ∈ R^{d_out x d_in}
        self.W0 = nn.Linear(d_in, d_out, bias=True)

        # 这里可自行选择激活函数
        self.activation = nn.ReLU()

    def forward(self, feat_local, feat_global, feat_sim):
        """
        feat_local, feat_global, feat_sim: shape=(N, d_in)
          N 表示节点数, d_in 表示输入维度 (三种特征维度一致)

        返回:
          h_final: shape=(N, d_out)
        """
        # 加权求和 => shape=(N, d_in)
        h_combined = (self.lambda1 * feat_local +
                      self.lambda2 * feat_global +
                      self.lambda3 * feat_sim)

        # 线性变换 + 激活 => shape=(N, d_out)
        h_final = self.activation(self.W0(h_combined))
        return h_final

class HeteroFeatureFusion(nn.Module):
    """
    当我们有 'disease', 'drug', 'gene' 三类节点,
    且每类节点的 local/global/similarity 特征维度相同 (都为 d_in),
    我们可以对三者做同样的特征融合.
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.disease_fusion = FeatureFusion(d_in, d_out)
        self.drug_fusion    = FeatureFusion(d_in, d_out)
        self.gene_fusion    = FeatureFusion(d_in, d_out)

    def forward(
        self,
        disease_local, disease_global, disease_sim,
        drug_local,    drug_global,    drug_sim,
        gene_local,    gene_global,    gene_sim
    ):
        """
        参数:
          disease_local/global/sim: (num_disease, d_in)
          drug_local/global/sim:    (num_drug, d_in)
          gene_local/global/sim:    (num_gene, d_in)

        返回:
          h_disease: (num_disease, d_out)
          h_drug:    (num_drug, d_out)
          h_gene:    (num_gene, d_out)
        """
        h_disease = self.disease_fusion(disease_local, disease_global, disease_sim)
        h_drug    = self.drug_fusion(drug_local, drug_global, drug_sim)
        h_gene    = self.gene_fusion(gene_local, gene_global, gene_sim)

        return h_disease, h_drug, h_gene


if __name__ == "__main__":
    # 假设维度 d_in=16, d_out=8
    d_in, d_out = 16, 8

    # 模拟一下三种节点数
    num_disease, num_drug, num_gene = 4, 5, 6

    fusion_model = HeteroFeatureFusion(d_in, d_out)


    disease_local    = torch.rand(num_disease, d_in)
    disease_global   = torch.rand(num_disease, d_in)
    disease_sim      = torch.rand(num_disease, d_in)

    drug_local       = torch.rand(num_drug, d_in)
    drug_global      = torch.rand(num_drug, d_in)
    drug_sim         = torch.rand(num_drug, d_in)

    gene_local       = torch.rand(num_gene, d_in)
    gene_global      = torch.rand(num_gene, d_in)
    gene_sim         = torch.rand(num_gene, d_in)

    # 前向
    h_disease, h_drug, h_gene = fusion_model(
        disease_local, disease_global, disease_sim,
        drug_local,    drug_global,    drug_sim,
        gene_local,    gene_global,    gene_sim
    )
    print("h_disease shape:", h_disease.shape)  # (4, 8)
    print("h_drug shape:   ", h_drug.shape)     # (5, 8)
    print("h_gene shape:   ", h_gene.shape)     # (6, 8)

    # 查看可学习的 λ1, λ2, λ3
    print("Disease λ1, λ2, λ3:",
          fusion_model.disease_fusion.lambda1.item(),
          fusion_model.disease_fusion.lambda2.item(),
          fusion_model.disease_fusion.lambda3.item())
