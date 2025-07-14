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
from MY_ENV.envs.target_model2 import model, targets, target_CV, observe_Fov, polar2dicaer
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
        self.step_count = None
        self.total_rewad = None
        self.agent_step_size = 7
        self.channel = 3
        self.obs_Rnums = 4
        self.grid_M = 20  
        self.grid_N = 20  
        self.utility_map_dim = self.grid_M * self.grid_N
        self.obs_dim = 8 * self.channel * self.obs_Rnums + self.utility_map_dim

        self.state = [
            [
                0,                      # 权重
                np.zeros((1, 4)),       # 均值
                np.zeros((4, 4)),       # 协方差
                0                       # 粒子数量
            ] for _ in range(self.n_agents)
        ]
        # 为每个agent维护一个搜索效用图，初始值为1
        self.agent_utility_maps = [
            np.ones((self.grid_M, self.grid_N), dtype=np.float32)
            for _ in range(self.n_agents)
        ]

        self.action_space = MultiAgentActionSpace(
                            [spaces.Discrete(9) for _ in range(self.n_agents)])
        
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.agent_Z_dicaer = [[[], []] for _ in range(self.n_agents)]

        # 1. 初始化观测上下界
        self.obs_low = np.zeros(self.obs_dim, dtype=np.float32)
        self.obs_high = np.ones(self.obs_dim, dtype=np.float32)

        section = 8 * self.obs_Rnums  # 2. 分通道设置上下界
        
        self.obs_high[0:section] = self.n_targets  # 通道1：目标数量 ∈ [0, n_targets]

        self.obs_low[section:2 * section] = self.x_min  # 通道2：目标x均值 ∈ [x_min, x_max]
        self.obs_high[section:2 * section] = self.x_max

        self.obs_low[2 * section:3 * section] = self.y_min  # 通道3：目标y均值 ∈ [y_min, y_max]
        self.obs_high[2 * section:3 * section] = self.y_max

        self.obs_low[-self.utility_map_dim:] = 0   # 3. 搜索效用图 ∈ [0,1]
        self.obs_high[-self.utility_map_dim:] = 1

        self.observation_space = MultiAgentObservationSpace(
                                [spaces.Box(self.obs_low, self.obs_high, dtype=np.float32) for _ in range(self.n_agents)])
        
    def full_init_pos(self):
        """
        获得初始化agent位置和目标位置、用于生成新生分量的观测数据
        """
        targets_birth_time, targets_death_time, targets_start = targets()
        trajectories, targets_tracks = target_CV(targets_birth_time, targets_death_time, targets_start, self.step, self.x_min, self.x_max, self.y_min, self.y_max, 
                          noise=True)
        for agent_i in range(self.n_agents):             #初始化每个agent的位置
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            self.agent_pos[agent_i] = np.array([x, y])
            # 初始化每个agent的观测数据（用于生成新生分量）
            self.agent_Z_dicaer[agent_i] = [[], []]
            pos = self.agent_pos[agent_i]
            model_data = model(pos[0], pos[1])
            for i in range(2):
                z_polar = observe_Fov(model_data, trajectories[i])
                Z_dicaer = polar2dicaer(z_polar, model_data)
                self.agent_Z_dicaer[agent_i][i] = Z_dicaer
        #return trajectories 
        
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
        
    def is_in_region(self, x, y, model_data, region_index, agent_pos):
        """
        判断目标是否在指定的区域内,以agent为中心划分圆形区域。
        :return: 是否在区域内
        """
        # 假设圆形划分为4个圆，每个圆分为8个扇形区域
        raduis = model_data["obverser_d"]
        radius_step = raduis / 4  # 每个圆的半径范围
        angle_step = 360 / 8  # 每个扇形的角度范围

        circle_index = region_index // 8  # 第几个圆
        sector_index = region_index % 8  # 第几个扇形

        # 计算当前区域的半径范围
        r_min = circle_index * radius_step
        r_max = (circle_index + 1) * radius_step

        # 计算当前区域的角度范围
        theta_min = sector_index * angle_step
        theta_max = (sector_index + 1) * angle_step

        # 转换目标坐标为以agent为中心的极坐标
        x_agent, y_agent = agent_pos
        dx, dy = x - x_agent, y - y_agent
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx) * 180 / np.pi  # 转换为角度
        if theta < 0:
            theta += 360

        # 判断目标是否在区域内
        return r_min <= r < r_max and theta_min <= theta < theta_max
        
    def get_agent_obs(self, agent_i):
        """
        获取agent 的观测数据,包括通过phd获得状态估计和更新地图。
        """
        pos = self.agent_pos[agent_i]
        model_data = model(pos[0], pos[1])
        z_polar = observe_Fov(model_data, self.trajectories)
        Z_dicaer = polar2dicaer(z_polar, model_data)  #返回的是列表

        self.agent_Z_dicaer[agent_i][0] = self.agent_Z_dicaer[agent_i][1]  # 上一时刻 ← 当前时刻
        self.agent_Z_dicaer[agent_i][1] = Z_dicaer  # 当前时刻 ← 新观测

        #每次phd需要前两个时刻的直角坐标观测数据和现在时刻的极坐标观测数据来生成新生分量
        phd = PHD(model_data, self.agent_Z_dicaer[agent_i][1], self.agent_Z_dicaer[agent_i][0], z_polar, self.state[agent_i])  # 调用PHD函数进行处理、
        X_now, cov = phd.predict_update()
        self.state[agent_i] = X_now    #更新状态
        state_draw, num_draw  = State_extraction(X_now)

        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # 填充通道1：目标数量
        section = 8 * self.obs_Rnums
        region_x_means = np.zeros(section, dtype=np.float32)  # 用于存储每个区域的目标x均值
        region_y_means = np.zeros(section, dtype=np.float32)
        region_counts = np.zeros(section, dtype=np.int32)
        for i in range(section):
            # 根据目标状态估计的集合，统计每个区域的目标数量
            region_count = 0
            for target in state_draw:
                x, y = target[0], target[1]  # 假设目标状态包含x和y坐标
                if self.is_in_region(x, y, model_data, region_index=i, agent_pos=pos):  # 判断目标是否在第i个区域
                    region_count += 1
                    region_x_means[i] += x
                    region_y_means[i] += y
                    region_counts[i] += 1
            obs[i] = region_count

            # 填充通道2：目标x均值
        for i in range(section):
            if region_counts[i] > 0:
                region_x_means[i] /= region_counts[i]  # 计算均值
        obs[section:2 * section] = region_x_means

        # 填充通道3：目标y均值
        for i in range(section):
            if region_counts[i] > 0:
                region_y_means[i] /= region_counts[i]  # 计算均值
        obs[2 * section:3 * section] = region_y_means
        
        #更新地图:需要维护的搜索效用图和提取后的观测数据情况



 

        return 



    def reset(self):
        self.step_count = 0
        self.total_rewad = [0 for _ in range(self.n_agents)]
        self.trajectories = self.full_init_pos()
        # 重置每个agent的效用图
        for i in range(self.n_agents):
            self.agent_utility_maps[i][:] = 1




    def update_agent_pos(self, agent_i, action):
        x, y = self.agent_pos[agent_i]

        # 设置 agent 的步长，为目标最大速度的 50%~100%
        step_size = getattr(self, "agent_step_size", 5)  # 可通过 self.agent_step_size 配置

        # 方向映射表
        action_map = {
            0: (0, 1),    # UP
            1: (0, -1),   # DOWN
            2: (-1, 0),   # LEFT
            3: (1, 0),    # RIGHT
            4: (-1, 1),   # LEFT_UP
            5: (1, 1),    # RIGHT_UP
            6: (-1, -1),  # LEFT_DOWN
            7: (1, -1),   # RIGHT_DOWN
            8: (0, 0),    # NOOP
        }

        dx, dy = action_map.get(action, (0, 0))

        # 如果是斜对角移动（如左上），对 sqrt(2) 步长归一化，保持速度一致
        if dx != 0 and dy != 0:
            norm = np.sqrt(2)
            dx /= norm
            dy /= norm

        # 更新位置
        new_x = x + dx * step_size
        new_y = y + dy * step_size

        # 加边界限制
        x_min, x_max = -1000, 1000
        y_min, y_max = -1000, 1000

        new_x = np.clip(new_x, x_min, x_max)
        new_y = np.clip(new_y, y_min, y_max)

        # 更新位置
        self.agent_pos[agent_i] = (new_x, new_y)

    
    def step(self):
        """
        对每个agent获取观测,根据观测和状态获得动作,并更新位置、计算奖励。
        (这里只需要输入动作即可)
        观测包括目标状态估计、协方差矩阵和搜索效用
        返回观测、奖励、结束标志和额外信息。
        """
        self.step_count += 1

        #obs = self.get_agent_obs()
        rewards = [0 for _ in range(self.n_agents)]
        dones = [False for _ in range(self.n_agents)]
        infos = [{} for _ in range(self.n_agents)]

        # 更新每个 agent 的位置、观测，计算奖励和是否终止
        for agent_i in range(self.n_agents):
            obs = self.get_agent_obs(agent_i)


            
        return obs, rewards, dones, infos




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