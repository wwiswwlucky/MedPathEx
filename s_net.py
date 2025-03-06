import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adj):
    """
    对给定的相似度邻接矩阵 A 执行归一化:
       A_tilde = A + I  (加自环)
       D_tilde(i) = sum_j(A_tilde(i,j))
       A_norm = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)

    参数:
    ------
    adj: torch.Tensor, shape=[N, N], 实数相似矩阵(对称)

    返回:
    ------
    A_norm: torch.Tensor, shape=[N, N]
        归一化后的邻接矩阵
    """
    # 1) 加自环
    n = adj.shape[0]
    device = adj.device
    A_tilde = adj + torch.eye(n, device=device)

    # 2) 计算度 D(i) = sum(A_tilde(i,:))
    degrees = A_tilde.sum(dim=1)  # shape: [N]

    # 3) D^(-1/2)
    d_inv_sqrt = torch.pow(degrees, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # 避免除 0

    # 4) 组合得到 A_norm
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt  # shape: [N, N]

    return A_norm


class ProjectionLayer(nn.Module):
    """
    实现公式 h_S = W_S * x_S
    """
    def __init__(self, in_dim, out_dim):
        super(ProjectionLayer, self).__init__()
        # 使用线性层来模拟可学习矩阵 W_S
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        # x shape: [N, in_dim]
        # 输出 shape: [N, out_dim]
        return self.linear(x)


class GCNLayer(nn.Module):
    """
    H^{(l+1)} = ReLU( A_norm * H^{(l)} * W^{(l)} )
    仅展示单层, 不包含多头注意力等.
    """
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)  # 参数初始化

    def forward(self, A_norm, H_in):
        """
        参数:
        ------
        A_norm: [N, N], 归一化邻接矩阵
        H_in:   [N, in_dim], 上一层输出特征 (或初始特征)

        返回:
        ------
        H_out:  [N, out_dim], 当前层输出特征
        """
        # (N, in_dim) * (in_dim, out_dim) -> (N, out_dim)
        support = torch.matmul(H_in, self.weight)
        # (N, N) * (N, out_dim) -> (N, out_dim)
        H_out = torch.matmul(A_norm, support)
        # ReLU
        H_out = F.relu(H_out)
        return H_out


class GCNNet(nn.Module):
    """
    一个简单的两层 GCN:
       H^(1) = ReLU(A_norm * H^(0) * W^(0))
       H^(2) = ReLU(A_norm * H^(1) * W^(1))
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCNNet, self).__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, A_norm, X):
        # 第一层
        h = self.gcn1(A_norm, X)
        # 第二层
        h = self.gcn2(A_norm, h)
        return h

###############################
# 5. 综合示例: Projection + GCN
###############################
def extract_similarity_features(sim_matrix, input_features, proj_out_dim=64, gcn_hidden_dim=32, gcn_out_dim=16):
    """
    给定一个相似性矩阵 (N,N) 和节点初始特征 (N, M)，
    先做线性投影 h_S = W_S * x_S (维度: proj_out_dim)，
    然后做两层GCN得到最终的节点表示 (维度: gcn_out_dim)。
    
    返回: h_final, shape=[N, gcn_out_dim]
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) 把 sim_matrix, input_features 转成 Torch Tensor
    A = torch.tensor(sim_matrix, dtype=torch.float32, device=device)
    X = torch.tensor(input_features, dtype=torch.float32, device=device)

    # 2) 对 A 做归一化
    A_norm = normalize_adjacency(A)

    # 3) Projection
    projection = ProjectionLayer(in_dim=X.shape[1], out_dim=proj_out_dim).to(device)
    X_proj = projection(X)  # [N, proj_out_dim]

    # 4) GCN
    gcn_model = GCNNet(in_dim=proj_out_dim, hidden_dim=gcn_hidden_dim, out_dim=gcn_out_dim).to(device)
    h_final = gcn_model(A_norm, X_proj)  # [N, gcn_out_dim]

    return h_final


if __name__ == '__main__':
    # 假设我们有 7 个节点(N=7)的相似矩阵(与论文示例数值类似)
    sim_example = [
        [1, 0, 0, 0, 0, 0, 0.054744526],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0.054744526, 0, 0, 0, 0, 0, 1]
    ]
    sim_example = torch.tensor(sim_example, dtype=torch.float32)

    # 模拟初始特征：假设每个节点有 10 维原始特征
    N = sim_example.shape[0]
    M = 10
    init_features = torch.randn(N, M)  #  [N, M]

    # 归一化邻接
    A_norm = normalize_adjacency(sim_example)

    # Projection
    projection_layer = ProjectionLayer(in_dim=M, out_dim=5)
    X_proj = projection_layer(init_features)
    print(f'[Projection] X_proj shape = {X_proj.shape}')  # [7, 5]

    # 两层GCN
    gcn_net = GCNNet(in_dim=5, hidden_dim=4, out_dim=2)
    h_out = gcn_net(A_norm, X_proj)
    print(f'[GCN Output] h_out shape = {h_out.shape}')  # [7, 2]

    # 也可直接调用封装的 extract_similarity_features 函数
    # (如在生产环境，通常不会写测试在同文件里)
    final_rep = extract_similarity_features(
        sim_matrix=sim_example.numpy(),          # NxN
        input_features=init_features.numpy(),    # NxM
        proj_out_dim=5,
        gcn_hidden_dim=4,
        gcn_out_dim=2
    )
    print('[Final Representation] shape:', final_rep.shape)  # [7, 2]
