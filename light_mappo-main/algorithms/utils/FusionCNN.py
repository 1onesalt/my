import torch
import torch.nn as nn
from .util import init

class FusionCNN(nn.Module):
    """
    融合网络：
    - Actor: 单智能体局部观测 [grid_flat(1024), self_state(3)]
    - Critic: 多智能体集中式输入，支持 [obs_i, active_mask_i] token 结构
    """
    def __init__(
        self,
        obs_shape,
        hidden_size=64,
        use_orthogonal=True,
        activation_id=1,
        kernel_size=3,
        stride=1,
        mode="auto",
    ):
        super(FusionCNN, self).__init__()

        self.total_dim = obs_shape[0]

        # 固定输入结构：grid(16*16*4) + self_state(3)
        self.c, self.h, self.w = 4, 16, 16
        self.single_grid_dim = self.c * self.h * self.w  # 1024
        self.single_state_dim = 3
        self.single_obs_dim = self.single_grid_dim + self.single_state_dim  # 1027
        self.token_with_mask_dim = self.single_obs_dim + 1  # 1028
        self.mode = mode

        if self.mode == "auto":
            self.mode = self._infer_mode()

        if self.mode == "single":
            self.vector_dim = self.total_dim - self.single_grid_dim
            self.num_agents = 1
            self.token_dim = self.single_obs_dim
            self.has_active_mask_per_token = False
            cnn_output_dim = 512  # 32*4*4
        else:
            self.mode = "dual"
            self.num_agents, self.token_dim, self.has_active_mask_per_token = self._parse_centralized_layout()
            self.vector_dim = self.num_agents * self.single_state_dim + (
                self.num_agents if self.has_active_mask_per_token else 0
            )
            cnn_output_dim = 512 * self.num_agents

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

    def _infer_mode(self):
        if self.total_dim == self.single_obs_dim:
            return "single"
        if self.total_dim > self.single_obs_dim:
            return "dual"
        raise ValueError(
            f"[FusionCNN] Invalid obs dim={self.total_dim}, expected at least {self.single_obs_dim}"
        )

    def _parse_centralized_layout(self):
        # 新版 runner 结构：[obs_i, active_mask_i] * N
        if self.total_dim % self.token_with_mask_dim == 0:
            return self.total_dim // self.token_with_mask_dim, self.token_with_mask_dim, True

        # 兼容旧版结构：[obs_i] * N（无 active_mask）
        if self.total_dim % self.single_obs_dim == 0:
            return self.total_dim // self.single_obs_dim, self.single_obs_dim, False

        raise ValueError(
            "[FusionCNN] Unsupported centralized input dim "
            f"{self.total_dim}. Expected N*{self.token_with_mask_dim} or N*{self.single_obs_dim}."
        )

    def _grid_flat_to_chw(self, grid_flat):
        # 环境按 (H, W, C) flatten，这里恢复为 CNN 需要的 (C, H, W)
        return grid_flat.view(-1, self.h, self.w, self.c).permute(0, 3, 1, 2).contiguous()

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
        x_grid = self._grid_flat_to_chw(grid)
        cnn_feat = self.cnn(x_grid)

        # 融合
        concat = torch.cat([cnn_feat, vector], dim=1)
        return self.fusion_mlp(concat)

    def _forward_dual(self, obs):
        # 集中式输入按 token 切片。token:
        # - 新版: [grid(1024), state(3), active_mask(1)]
        # - 旧版: [grid(1024), state(3)]
        cnn_feats = []
        vectors = []

        for i in range(self.num_agents):
            token_start = i * self.token_dim
            token_end = (i + 1) * self.token_dim
            token = obs[:, token_start:token_end]
            agent_obs = token[:, :self.single_obs_dim]

            grid = agent_obs[:, :self.single_grid_dim]
            vec = agent_obs[:, self.single_grid_dim:]

            if self.has_active_mask_per_token:
                vec = torch.cat([vec, token[:, self.single_obs_dim:self.single_obs_dim + 1]], dim=1)

            x_grid = self._grid_flat_to_chw(grid)
            cnn_out = self.cnn(x_grid)

            cnn_feats.append(cnn_out)
            vectors.append(vec)

        all_cnn = torch.cat(cnn_feats, dim=1)
        all_vec = torch.cat(vectors, dim=1)

        concat = torch.cat([all_cnn, all_vec], dim=1)
        return self.fusion_mlp(concat)