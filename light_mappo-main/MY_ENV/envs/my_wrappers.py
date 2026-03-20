import gym
import numpy as np
from gym import spaces

# class FlattenObservation(gym.ObservationWrapper):
#     """
#     针对 TargetSearch 环境的专用包装器。
#     作用：将 Dict 类型的观测压平为一维向量，以便存入 Buffer。
#     """
#     def __init__(self, env):
#         super(FlattenObservation, self).__init__(env)
        
#         # 1. 获取原始观测空间 (假设所有智能体同构，取第一个)
#         # 注意：这里兼容 MultiAgentObservationSpace 是列表的情况
#         if isinstance(env.observation_space, list):
#             old_space = env.observation_space[0]
#         else:
#             old_space = env.observation_space

#         # 2. 获取各部分形状
#         self.phd_shape = old_space['phd_heatmap'].shape
#         self.pos_shape = old_space['self_pos'].shape
        
#         # 3. 计算压平后的维度
#         self.phd_dim = int(np.prod(self.phd_shape))
#         self.pos_dim = int(np.prod(self.pos_shape))
        
#         total_dim = self.phd_dim +  self.pos_dim
        
#         # 4. 定义新的观测空间 (Box)
#         self.observation_space = [
#             spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
#             for _ in range(env.n_agents)
#         ]
        
#         self.share_observation_space = self.observation_space

#     def observation(self, observation):
#         """
#         将环境返回的 List[Dict] 转换为 List[Array] (即压平)
#         """
#         flat_obs_n = []
#         for obs_dict in observation:
#             # 展平各个组件
#             phd_flat = obs_dict['phd_heatmap'].flatten()
#             pos_flat = obs_dict['self_pos'].flatten()
            
#             # 拼接
#             flat_vec = np.concatenate([phd_flat, pos_flat])
#             flat_obs_n.append(flat_vec)
            
#         # 返回 numpy 数组，符合 light_mappo 接口要求
#         return np.array(flat_obs_n)
    


class FlattenObservation(gym.ObservationWrapper):
    """
    将字典观测空间 {'polar_grid': (U,V,C), 'self_state': (3,)} 
    压平为一维向量，方便 MAPPO 的 ReplayBuffer 存储。

    压平后的结构: [ --- polar_grid flat --- | --- self_state --- ]
    """
    def __init__(self, env):
        super().__init__(env)
        
        # 获取原始空间维度
        self.grid_shape = env.observation_space[0]['polar_grid'].shape # (16, 16, 4)
        self.state_shape = env.observation_space[0]['self_state'].shape # (3,)
        
        self.grid_dim = np.prod(self.grid_shape)
        self.state_dim = np.prod(self.state_shape)
        self.total_dim = self.grid_dim + self.state_dim
        self.max_agents = env.n_agents
        
        # 定义新的平铺观测空间
        self.observation_space = []
        for agent_id in range(env.n_agents):
            self.observation_space.append(
                gym.spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(self.total_dim,), 
                    dtype=np.float32
                )
            )

        # Critic 输入约定：拼接 [obs_i, active_mask_i]，总维度 (obs_dim + 1) * max_agents
        self.share_total_dim = (self.total_dim + 1) * self.max_agents

        self.share_observation_space = []
        for _ in range(env.n_agents):
            self.share_observation_space.append(
                gym.spaces.Box(
                    low=-np.inf, 
                    high=np.inf, 
                    shape=(self.share_total_dim,), 
                    dtype=np.float32
                )
            )       

    def observation(self, observation):
        """将列表中的每个字典观测压平"""
        flat_obs_n = []
        for obs_dict in observation:
            # 1. 压平 Grid: (16, 16, 4) -> (1024,)
            grid_flat = obs_dict['polar_grid'].flatten()
            
            # 2. 获取 State: (3,)
            state_vec = obs_dict['self_state']
            
            # 3. 拼接
            flat_vec = np.concatenate([grid_flat, state_vec])
            flat_obs_n.append(flat_vec)
            
        return flat_obs_n
    
    def step(self, actions):
            # 1. 执行环境步
            obs_n, rewards, dones, infos = self.env.step(actions)

            # 2. 返回压平后的局部观测
            flat_obs_n = self.observation(obs_n)
            
            return flat_obs_n, rewards, dones, infos

    # --- 重写 reset 以初始化 share_obs (虽然 Runner 可能主要用 step 的 info) ---
    def reset(self):
        obs_n = self.env.reset()
        
        # 生成初始的全局观测 (尽管 reset 只能返回 obs, 无法返回 info)
        # 但为了逻辑完整性，我们可以计算它，或者如果需要在 reset 后立即获取 share_obs，
        # 我们可以通过 hack 的方式 (例如返回一个 tuple) 但这会破坏 Gym 接口。
        # 
        # 在 light_mappo 中，第一帧的 share_obs 通常是通过 runner 中的 warm_up 或者 
        # 单独调用环境接口获取的。如果不想修改 runner，可以暂不处理 reset 的 share_obs，
        # 因为 buffer 中第一帧通常是全 0 或者随机初始化，真正训练从第一步 step 后开始。
        
        # 这里仅返回压平的局部观测
        return self.observation(obs_n)    