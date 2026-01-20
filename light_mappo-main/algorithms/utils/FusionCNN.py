import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import init, get_clones

class FusionCNN(nn.Module):
    """
    对应论文 4.4.1 节：特征提取模块与特征融合模块
    输入: Flattened observation [Grid (U*V*C) | Self_State (3)]
    输出: Hidden Feature Vector
    """
    def __init__(self, obs_shape, hidden_size=64, use_orthogonal=True, activation_id=1, kernel_size=3, stride=1):
        super(FusionCNN, self).__init__()
        
        # --- 1. 参数解析与拆分维度 ---
        # 假设 obs_shape 是一个元组，例如 (1027,)
        self.total_dim = obs_shape[0]
        
        # 硬编码来自环境的维度 (需与环境保持一致)
        self.c, self.h, self.w = 4, 16, 16 # Channel=4, H=16, W=16
        self.grid_flat_dim = self.c * self.h * self.w # 1024
        self.state_dim = 3 
        
        # 验证维度匹配
        assert self.total_dim == self.grid_flat_dim + self.state_dim, \
            f"Input dim {self.total_dim} does not match Grid({self.grid_flat_dim}) + State({self.state_dim})"

        # --- 2. CNN 前端 (特征提取模块) ---
        # "CNN 前端由三层卷积层构成"
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.c, out_channels=16, kernel_size=3, stride=1, padding=1)),
            active_func,
            # Layer 2
            init_(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)), # Downsample 16->8
            active_func,
            # Layer 3
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)), # Downsample 8->4
            active_func,
            nn.Flatten(),
        )
        
        # 计算 CNN 输出维度: 32 channels * 4 * 4 = 512
        self.cnn_out_dim = 32 * 4 * 4

        # --- 3. 特征融合模块 (MLP) ---
        # "CNN提取的空间特征在展平后，与归一化处理后的自身状态向量Os进行拼接，并输入至全连接层"
        self.fusion_input_dim = self.cnn_out_dim + self.state_dim
        
        self.fusion_mlp = nn.Sequential(
            init_(nn.Linear(self.fusion_input_dim, hidden_size)),
            active_func,
            init_(nn.LayerNorm(hidden_size))
        )

    def forward(self, obs):
        # obs shape: (batch_size, 1027)
        
        # 1. 数据切片 (Slicing)
        # Grid 部分: 前 1024 维
        grid_flat = obs[:, :self.grid_flat_dim] 
        # State 部分: 后 3 维
        state_vec = obs[:, self.grid_flat_dim:] 
        
        # 2. 维度重塑 (Reshape for CNN)
        # (Batch, 1024) -> (Batch, 4, 16, 16)
        # 注意: PyTorch Conv2d 需要 (N, C, H, W)
        x_grid = grid_flat.view(-1, self.c, self.h, self.w)
        
        # 3. CNN 前向传播
        x_cnn_feat = self.cnn(x_grid) # Output: (Batch, 512)
        
        # 4. 特征拼接 (Concatenation)
        concat_feat = torch.cat([x_cnn_feat, state_vec], dim=1) # (Batch, 515)
        
        # 5. 融合 MLP
        output = self.fusion_mlp(concat_feat) # (Batch, hidden_size)
        
        return output