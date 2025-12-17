import torch
import torch.nn as nn
from algorithms.utils.util import init
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase

class MultiModalBase(nn.Module):
    """多模态特征提取基础网络"""
    def __init__(self, args, obs_shape):
        super(MultiModalBase, self).__init__()
        
        self.target_encoder = CNNBase(args, obs_shape['target'].shape)
        self.utility_encoder = CNNBase(args, obs_shape['utility'].shape)
        self.pos_encoder = MLPBase(args, obs_shape['self_pos'].shape)
        
        # 特征融合层
        hidden_size = 128   # 统一设定隐藏维度（你只需要在 args 里写一次）

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.fusion = nn.Sequential(
            init_(nn.Linear(hidden_size * 3, hidden_size)),  # 三路拼接后降维
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size))       # 保持 hidden_size
        )
        
    def forward(self, obs_dict):
        # 分别提取特征
        target_feat = self.target_encoder(obs_dict['target'])
        utility_feat = self.utility_encoder(obs_dict['utility'])
        pos_feat = self.pos_encoder(obs_dict['self_pos'])
        
        # 特征融合
        combined = torch.cat([target_feat, utility_feat, pos_feat], dim=-1)
        return self.fusion(combined)