import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def compute_association_score(h_disease, h_drug, disease_indices, drug_indices):
    """
    根据疾病和药物节点特征, 计算一批 (disease_i, drug_j) 的关联得分 y_hat.
    公式(16): y_hat_{ij} = sigma( h_i^T * h_j )
    
    参数:
      h_disease: shape=(num_disease, d_dim), 每个疾病节点的特征
      h_drug: shape=(num_drug, d_dim), 每个药物节点的特征
      disease_indices: shape=(batch_size,), 表示哪些疾病节点
      drug_indices: shape=(batch_size,), 表示对应药物节点
    返回:
      score: shape=(batch_size,), 介于 [0,1] 的关联概率
    """
    # 取出对应特征
    # (batch_size, d_dim)
    d_feat = h_disease[disease_indices]  
    r_feat = h_drug[drug_indices]
    # 内积 => shape=(batch_size,)
    logits = (d_feat * r_feat).sum(dim=1)
    # sigmoid => 概率
    probs = torch.sigmoid(logits)
    return probs


class AssocModel(nn.Module):
    """
    演示一个整体流程, 其中:
      h_disease, h_drug 由外部传入 (例如融合后的 final embedding),
      只负责 compute_association_score + loss.

    如果你的 fusion λ1, λ2, λ3 也需训练, 可将它们也放在本类中.
    """
    def __init__(self, h_disease, h_drug):
        super().__init__()
        # 注册固定 embedding 供示例, 真实情况你可用 Parameter 或 forward 生成
        self.h_disease = nn.Parameter(h_disease, requires_grad=False)
        self.h_drug    = nn.Parameter(h_drug, requires_grad=False)

    def forward(self, disease_indices, drug_indices):
        # 这里 disease_indices, drug_indices shape=(batch_size,)
        # => compute_association_score
        logits = (self.h_disease[disease_indices] * self.h_drug[drug_indices]).sum(dim=1)
        return logits


if __name__ == '__main__':
    # 假设我们有4个疾病, 5个药物 => final embedding dimension=8
    num_d, num_r, dim = 4, 5, 8
    # 随机生成示例 embedding
    disease_embed = torch.randn(num_d, dim)
    drug_embed    = torch.randn(num_r, dim)

    # 构造一个简单模型
    model = AssocModel(disease_embed, drug_embed)

    # 构造一些正/负例
    # e.g. 正例 => [(0,0), (1,2)], 负例 => [(2,1), (1,4)]
    pos_pairs = torch.tensor([[0,0], [1,2]], dtype=torch.long)  # (2,2)
    neg_pairs = torch.tensor([[2,1], [1,4]], dtype=torch.long)  # (2,2)

    # 组合 => 4条样本
    disease_indices = torch.cat([pos_pairs[:,0], neg_pairs[:,0]], dim=0)  # (4,)
    drug_indices    = torch.cat([pos_pairs[:,1], neg_pairs[:,1]], dim=0) # (4,)
    labels = torch.cat([torch.ones(pos_pairs.size(0)), torch.zeros(neg_pairs.size(0))], dim=0)
    # shape=(4,) => e.g. [1,1,0,0]

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    bce_loss  = nn.BCEWithLogitsLoss()  
    # BCEWithLogitsLoss 内部包含 sigmoid, 所以 forward(...) 直接输出 logits 即可

    # 训练循环 (仅展示一个简单epoch)
    model.train()
    optimizer.zero_grad()
    logits = model(disease_indices, drug_indices)  # (4,)
    loss = bce_loss(logits, labels)
    loss.backward()
    optimizer.step()
    print(f'Training Loss: {loss.item():.4f}')

    # 测试 => 计算关联概率
    model.eval()
    with torch.no_grad():
        # 计算 pos_pairs => disease_indices=[0,1], drug_indices=[0,2]
        test_logits = model(pos_pairs[:,0], pos_pairs[:,1])
        test_probs = torch.sigmoid(test_logits)
        print('Positive pairs probability:', test_probs)
    
    print('[Done] Association Score Calculation + Training Example!')
