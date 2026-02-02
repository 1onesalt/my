"""
# @Time    : 2021/7/1 6:53 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : r_actor_critic.py
"""

import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from algorithms.utils.popart import PopArt
from utils.util import get_shape_from_obs_space

# 在 r_actor_critic.py 头部导入
from algorithms.utils.cnn import CNNBase # 确保导入了 FusionCNN
from algorithms.utils.FusionCNN import FusionCNN
from algorithms.utils.mlp import MLPBase


# class AdvancedCombinedExtractor(nn.Module):
#     def __init__(self, observation_space, features_dim=256):
#         super(AdvancedCombinedExtractor, self).__init__()
#         self._features_dim = features_dim
#         self.extractors = nn.ModuleDict()
#         total_concat_size = 0
        
# # --- 硬编码维度 (无 Utility) ---
#         self.phd_shape = (3, 64, 64) 
#         self.pos_shape = (2,)
        
#         self.phd_dim = 3 * 64 * 64
#         self.pos_dim = 2
        
#         # 1. CNN 模块 (论文: "CNN前端由三层卷积层构成")
#         # 仅处理 PHD 热力图 (3通道)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 4 * 4, 128), nn.ReLU()
#         )
#         self.cnn_out_dim = 128

#         # 2. 特征融合 MLP (论文: "与自身状态向量拼接...输入至全连接层")
#         self.fusion_input_dim = self.cnn_out_dim + self.pos_dim
        
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(self.fusion_input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128), 
#             nn.ReLU()
#         )
#         self.final_out_dim = 128

#     def forward(self, observations):
# # 1. 还原切片
#         # 前面部分是 PHD Map
#         obs_phd = observations[:, :self.phd_dim].view(-1, *self.phd_shape)
#         # 后面部分是 Pos
#         obs_pos = observations[:, self.phd_dim:]
        
#         # 2. CNN 提取空间特征
#         cnn_feat = self.cnn(obs_phd)
        
#         # 3. 拼接 (Concat)
#         fusion_input = torch.cat([cnn_feat, obs_pos], dim=1)
        
#         # 4. MLP 融合
#         output = self.fusion_mlp(fusion_input)
        
#         return output
        
#     @property
#     def output_dim(self):
#         return self.final_out_dim


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        if args.use_fusion_network: # 需要在 config.py 中添加此参数
                    self.base = FusionCNN(
                        obs_shape, 
                        self.hidden_size, 
                        self._use_orthogonal
                    )
        elif len(obs_shape) == 3:
                    # 原有的 Image 处理逻辑
                    self.base = CNNBase(args, obs_shape)
        else:
            # 原有的 Vector 处理逻辑
            self.base = MLPBase(args, obs_shape)

        # self.base = AdvancedCombinedExtractor(obs_space)
        # input_dim = self.base.output_dim

        if hasattr(self.base, 'output_dim'):
                    input_dim = self.base.output_dim
        else:
            # FusionCNN 最后输出的是 hidden_size
            input_dim = self.hidden_size


        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if args.use_fusion_network:
            # 实例化 FusionCNN 作为 Critic 的基础网络（Base）
            # 注意：这里的 share_obs_shape 维度必须与 FusionCNN 预期的输入一致
            self.base = FusionCNN(cent_obs_shape, self.hidden_size, self._use_orthogonal)
                # --- 修改结束 ---
        elif len(cent_obs_shape) == 3:
            self.base = CNNBase(args, cent_obs_shape)
        else:
            self.base = MLPBase(args, cent_obs_shape)

        # self.base = AdvancedCombinedExtractor(cent_obs_space)
        # input_dim = self.base.output_dim

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states


