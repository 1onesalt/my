import torch
import torch.nn as nn
from .util import init

class FusionCNN(nn.Module):
    """
    智能融合网络：支持单路 CNN (Actor) 和 双路 CNN (Critic)
    """
    def __init__(self, obs_shape, hidden_size=64, use_orthogonal=True, activation_id=1, kernel_size=3, stride=1):
        super(FusionCNN, self).__init__()
        
        self.total_dim = obs_shape[0]
        
        # 硬编码维度
        self.c, self.h, self.w = 4, 16, 16
        self.single_grid_dim = self.c * self.h * self.w # 1024
        self.single_state_dim = 3
        self.single_obs_dim = self.single_grid_dim + self.single_state_dim # 1027
        
        # === 智能判断模式 ===
        if self.total_dim <= 1100:
            # 模式 A: Actor (局部观测, ~1027维)
            self.mode = "single"
            self.vector_dim = self.total_dim - self.single_grid_dim
            cnn_output_dim = 512 # 32*4*4
        else:
            # 模式 B: Critic (全局观测, ~2054维)
            # 假设全局观测是两个局部观测的拼接
            self.mode = "dual"
            self.num_agents = self.total_dim // self.single_obs_dim # 应该是 2
            
            # 除去所有 Grid 剩下的向量维度
            self.vector_dim = self.total_dim - (self.single_grid_dim * self.num_agents)
            
            # 双路 CNN，输出维度翻倍
            cnn_output_dim = 512 * self.num_agents 
            
            print(f"[FusionCNN] Initialized in DUAL mode. Input: {self.total_dim}, Agents: {self.num_agents}")

        # --- CNN 骨干网络 ---
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # 共享权重的 CNN 特征提取器
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(self.c, 16, kernel_size=3, stride=1, padding=1)),
            active_func,
            init_(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)), # 16->8
            active_func,
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)), # 8->4
            active_func,
            nn.Flatten(),
        )

        # --- 融合 MLP ---
        self.fusion_input_dim = cnn_output_dim + self.vector_dim
        
        self.fusion_mlp = nn.Sequential(
            init_(nn.Linear(self.fusion_input_dim, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size)
        )

    def forward(self, obs):
        if self.mode == "single":
            return self._forward_single(obs)
        else:
            return self._forward_dual(obs)

    def _forward_single(self, obs):
        # 提取 Grid (前1024)
        grid = obs[:, :self.single_grid_dim]
        # 提取 Vector (后3)
        vector = obs[:, self.single_grid_dim:]
        
        # CNN 处理
        x_grid = grid.view(-1, self.c, self.h, self.w)
        cnn_feat = self.cnn(x_grid)
        
        # 融合
        concat = torch.cat([cnn_feat, vector], dim=1)
        return self.fusion_mlp(concat)

    def _forward_dual(self, obs):
        # 全局观测结构: [Grid1, State1, Grid2, State2]
        # 我们需要将其拆解
        
        cnn_feats = []
        vectors = []
        
        # 逐个智能体切片
        for i in range(self.num_agents):
            start = i * self.single_obs_dim
            end = (i + 1) * self.single_obs_dim
            agent_obs = obs[:, start:end]
            
            # 拆分 Grid 和 Vector
            grid = agent_obs[:, :self.single_grid_dim]
            vec = agent_obs[:, self.single_grid_dim:]
            
            # CNN 提取特征
            x_grid = grid.view(-1, self.c, self.h, self.w)
            cnn_out = self.cnn(x_grid)
            
            cnn_feats.append(cnn_out)
            vectors.append(vec)
            
        # 拼接所有特征: [CNN1, CNN2, Vec1, Vec2]
        # 这样保留了所有空间信息
        all_cnn = torch.cat(cnn_feats, dim=1)
        all_vec = torch.cat(vectors, dim=1)
        
        concat = torch.cat([all_cnn, all_vec], dim=1)
        return self.fusion_mlp(concat)