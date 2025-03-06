import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAttention(nn.Module):
   
    def __init__(self, in_dim, attn_dim=None, negative_slope=0.2):
        """
        参数:
          in_dim: 节点特征维度 d
          attn_dim: 注意力内部维度, 若为 None 则默认等于 in_dim
          negative_slope: LeakyReLU 的斜率
        """
        super(GlobalAttention, self).__init__()
        if attn_dim is None:
            attn_dim = in_dim
        # W \in R^{attn_dim x (2*in_dim)}
        self.W = nn.Linear(2 * in_dim, attn_dim, bias=False)
        # a \in R^{attn_dim, 1}, 最终做点积
        self.a = nn.Parameter(torch.empty(attn_dim, 1))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.in_dim = in_dim
        self.attn_dim = attn_dim

    def forward(self, H):
      
        N, d = H.shape
        device = H.device

        # 1. 构造所有节点对 (i, j) 的 concat 特征: [h_i || h_j]
        #   结果 shape=(N, N, 2*d)
        #   注意大规模 N 时要小心内存
        H_i = H.unsqueeze(1).expand(N, N, d)   # (N, N, d)
        H_j = H.unsqueeze(0).expand(N, N, d)   # (N, N, d)
        H_cat = torch.cat([H_i, H_j], dim=2)   # (N, N, 2*d)

        # 2. 通过线性层 W: shape=(N, N, attn_dim)
        W_H = self.W(H_cat)

        # 3. LeakyReLU
        E = self.leaky_relu(W_H)  # (N, N, attn_dim)

        # 4. 与 a 做点积, 计算 e_{i,j}
        #    a shape=(attn_dim, 1) => broadcast => (N, N)
        #    E shape=(N, N, attn_dim)
        #    => e_{i,j} = (E_{i,j} dot a). shape=(N, N, 1) => squeeze -> (N, N)
        e_scores = torch.matmul(E, self.a).squeeze(-1)  # (N, N)

        # 5. softmax w.r.t. j
        alpha = F.softmax(e_scores, dim=1)  # (N, N)

        # 6. 加权求和, h_{v_i}' = sum_j alpha_{i,j} * h_j
        #   alpha shape=(N, N), H shape=(N, d)
        #   alpha_{i,j} * h_j -> 先把 alpha 扩展成 (N, N, 1)
        alpha_3d = alpha.unsqueeze(-1).expand(N, N, d)  # (N, N, d)
        # H_j = shape (1, N, d) + broadcast => (N, N, d)
        H_j2 = H.unsqueeze(0).expand(N, N, d)
        # element-wise product
        out = alpha_3d * H_j2  # (N, N, d)
        # sum over j => (N, d)
        H_prime = out.sum(dim=1)

        return H_prime
