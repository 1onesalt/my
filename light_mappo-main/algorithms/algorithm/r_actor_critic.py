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

        # obs_shape = get_shape_from_obs_space(obs_space)
        # base = CNNBase if len(obs_shape) == 3 else MLPBase   #obs_shape = (C, H, W) → CNN、obs_shape = (N,) → MLP
        # self.base = base(args, obs_shape)                    #创建base网络
        from algorithms.utils.multimodal import MultiModalBase
        self.base = MultiModalBase(args, obs_space)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:  #是否启用RNN 层
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)  #根据网络提取的特征，生成具体动作

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
        # obs = check(obs).to(**self.tpdv)        #把 numpy → torch、reshape、检查 dtype 等
        # rnn_states = check(rnn_states).to(**self.tpdv)
        # masks = check(masks).to(**self.tpdv)
        # if available_actions is not None:
        #     available_actions = check(available_actions).to(**self.tpdv)

        # 确保输入是字典格式
        if not isinstance(obs, dict):
            raise ValueError("Observation must be a dictionary")
            
        # 转换每个观测分量到tensor
        for key in obs:
            if not torch.is_tensor(obs[key]):
                obs[key] = torch.FloatTensor(obs[key])
            obs[key] = obs[key].to(**self.tpdv)
        
        actor_features = self.base(obs)        #过 base 网络提取特征

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
        # obs = check(obs).to(**self.tpdv)
        # rnn_states = check(rnn_states).to(**self.tpdv)
        # action = check(action).to(**self.tpdv)
        # masks = check(masks).to(**self.tpdv)
        # if available_actions is not None:
        #     available_actions = check(available_actions).to(**self.tpdv)

        # if active_masks is not None:
        #     active_masks = check(active_masks).to(**self.tpdv)

        # actor_features = self.base(obs)

        if not isinstance(obs, dict):
            raise ValueError("Observation must be a dictionary")
        
        processed_obs = {}
        for key in obs:
            if not torch.is_tensor(obs[key]):
                processed_obs[key] = torch.FloatTensor(obs[key])
            else:
                processed_obs[key] = obs[key]
            processed_obs[key] = processed_obs[key].to(**self.tpdv)
        
        actor_features = self.base(processed_obs)

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

        # cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        # base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        # self.base = base(args, cent_obs_shape)

        # 同样使用多模态base
        from algorithms.utils.multimodal import MultiModalBase
        self.base = MultiModalBase(args, cent_obs_space)

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
        # cent_obs = check(cent_obs).to(**self.tpdv)
        # rnn_states = check(rnn_states).to(**self.tpdv)
        # masks = check(masks).to(**self.tpdv)

        # critic_features = self.base(cent_obs)

        # 修改观测处理部分
        if not isinstance(cent_obs, dict):
            raise ValueError("Centralized observation must be a dictionary")
        
        processed_obs = {}
        for key in cent_obs:
            if not torch.is_tensor(cent_obs[key]):
                processed_obs[key] = torch.FloatTensor(cent_obs[key])
            else:
                processed_obs[key] = cent_obs[key]
            processed_obs[key] = processed_obs[key].to(**self.tpdv)
        
        critic_features = self.base(processed_obs)


        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
