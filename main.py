import argparse
from models import SimilarityNetwork, HeteroGNN, FeatureFusion
from data.loader import BioDataLoader

def main(config):
    # 1. 数据准备
    loader = BioDataLoader(config.data_dir)
    sim_data = loader.load_similarity()
    hetero_data = loader.load_association()
    train_pairs, test_pairs = loader.generate_pairs()
    
    # 2. 模型初始化
    sim_net = SimilarityNetwork(
        feat_dims={'drug': 512, 'disease': 256, 'gene': 1024},
        hidden_dim=128
    )
    hetero_net = HeteroGNN(
        feat_dims={'drug': 64, 'disease': 64, 'gene': 64},
        hidden_dim=128,
        metapaths=['RDR', 'DGD']
    )
    fusion = FeatureFusion(hidden_dim=128)
    predictor = AssociationPredictor()
    
    # 3. 训练流程
    trainer = Trainer(
        models=[sim_net, hetero_net, fusion, predictor],
        lr=config.lr,
        device=config.device
    )
    
    # 4. 训练循环
    for epoch in range(config.epochs):
        loss = trainer.train_epoch(
            sim_data, 
            hetero_data,
            train_pairs
        )
        print(f"Epoch {epoch+1}/{config.epochs} Loss: {loss:.4f}")
        
        # 验证步骤
        if epoch % 5 == 0:
            metrics = trainer.evaluate(test_pairs)
            print(f"Validation Metrics: {metrics}")

class Trainer:
    """训练管理类"""
    def __init__(self, models, lr, device):
        self.models = nn.ModuleList(models)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
    
    def train_epoch(self, sim_data, hetero_data, pairs):
        self.train()
        # 前向传播流程
        sim_feat = self.models[0](sim_data)
        hetero_feat = self.models[1](hetero_data)
        fused = self.models[2](hetero_feat, sim_feat)
        preds = self.models[3](fused)
        
        # 损失计算
        loss = self.criterion(preds, pairs)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    main(args)
