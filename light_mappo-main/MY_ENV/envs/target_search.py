import copy
import logging
import random
import gym
import numpy as np

from MY_ENV.utils.action_space import MultiAgentActionSpace
from MY_ENV.utils.observation_space import MultiAgentObservationSpace
from MY_ENV.utils.draw import draw_grid, fill_cell, write_cell_text

from gym import spaces
from gym.utils import seeding
from MY_ENV.envs.target_model2 import model, targets, target_CV, observe_Fov
from PHD import PHD
from PHD import State_extraction

class target_search(gym.Env):
    def __init__(self, x_min = -1000, x_max = 1000, y_min = -1000, y_max = 1000, n_agent = 3, n_target = 5, step = 100):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.n_agents = n_agent
        self.n_targets = n_target
        self.step = step
        self.total_rewad = None
        self.channel = 3
        self.obs_Rnums = 4
        self.obs_dim = 8 * self.channel * self.obs_Rnums + 100
        self.state = [
            0,                      # 权重
            np.zeros((1, 4)),       # 均值
            np.zeros((4, 4)),       # 协方差
            0                       # 粒子数量
        ]
        self.action_space = MultiAgentActionSpace(
                            [spaces.Discrete(9) for _ in range(self.n_agents)])
        
        self.agent_pos = {_: None for _ in range(self.n_agents)}

        # 1. 初始化观测上下界
        self.obs_low = np.zeros(self.obs_dim, dtype=np.float32)
        self.obs_high = np.ones(self.obs_dim, dtype=np.float32)

        # 2. 分通道设置上下界
        section = 8 * self.obs_Rnums

        # 通道1：目标数量 ∈ [0, n_targets]
        self.obs_high[0:section] = self.n_targets

        # 通道2：目标x均值 ∈ [x_min, x_max]
        self.obs_low[section:2 * section] = self.x_min
        self.obs_high[section:2 * section] = self.x_max

        # 通道3：目标y均值 ∈ [y_min, y_max]
        self.obs_low[2 * section:3 * section] = self.y_min
        self.obs_high[2 * section:3 * section] = self.y_max

        # 3. 搜索效用图 ∈ [0,1]
        self.obs_low[-self.utility_map_dim:] = 0
        self.obs_high[-self.utility_map_dim:] = 1

        self.observation_space = MultiAgentObservationSpace(
                                [spaces.Box(self.obs_low, self.obs_high, dtype=np.float32) for _ in range(self.n_agents)])
        
        targets_birth_time, targets_death_time, targets_start = targets()
        # print(targets_death_time)
        self.trajectories, self.targets_tracks = target_CV(targets_birth_time, targets_death_time, targets_start, self.step, self.x_min, self.x_max, self.y_min, self.y_max, 
                          noise=True)


    def get_action_meanings(self, agent_i=None):
        action_meaning = []
        for _ in range(self.n_agents):
            meaning = [ACTION_MEANING[i] for i in range(9)]
            action_meaning.append(meaning)
        if agent_i is not None:
            assert isinstance(agent_i, int)   #确保 agent_i 是一个整数，并且在合法范围内
            assert 0 <= agent_i <= self.n_agents

            return action_meaning[agent_i]
        else:
            return action_meaning
        
    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            model_data = model(pos[0], pos[1])
            z_polar = observe_Fov(model_data, self.trajectories)
            
            #每次phd需要前两个时刻的直角坐标观测数据和现在时刻的极坐标观测数据来生成新生分量
            phd = PHD(model_data, Z_dicaer[i - 2], Z_dicaer[i - 1], z_polar, self.state)  # 调用PHD函数进行处理
            X_now = phd.predict_update()
            state_draw, num_draw  = State_extraction(X_now)




#     def full_target_mea(self):
#         targets_birth_time, targets_death_time, targets_start = targets()
#         trajectories, targets_tracks = target_CV(targets_birth_time, targets_death_time, targets_start, self.step, 
#                                                 self.x_min, self.x_max, self.y_min, self.y_max, noise=True)
#         return trajectories, targets_tracks          

    
#     def get_agent_mea(self):
        

#     def reset(self):



ACTION_MEANING = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "LEFT_UP",
    5: "RIGHT_UP",
    6: "LEFT_DOWN",
    7: "RIGHT_DOWN",
    8: "NOOP",
}