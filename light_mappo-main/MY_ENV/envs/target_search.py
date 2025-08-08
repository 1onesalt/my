import copy
import logging
import random
import gym
import numpy as np

from gym.spaces import Dict, Box

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
        self.x_min = x_min          #区域范围
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.n_agents = n_agent
        self.n_targets = n_target
        self.step = step
        self.step_count = None
        self.total_rewad = None
        self.agent_step_size = 7
        self.done = False

        self.state = [
            [
                0,                      # 权重
                np.zeros((1, 4)),       # 均值
                np.zeros((4, 4)),       # 协方差
                0                       # 粒子数量
            ] for _ in range(self.n_agents)
        ]

        # 智能体位置和PHD观测数据初始化
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.agent_Z_dicaer = [[[], []] for _ in range(self.n_agents)]

        self.action_space = MultiAgentActionSpace(
                            [spaces.Discrete(9) for _ in range(self.n_agents)])
        
        #观测空间初始化信息
        self.utility_channel = 2
        self.utilityMap_M = 20     #效用图划分
        self.utilityMap_N = 20 
        self.utility_obs_shape = (self.utility_channel, self.utilityMap_M, self.utilityMap_N)

        self.channel = 3           #视域内目标是否讯在、x轴平均位置、y轴平均位置、x轴平均速度、y轴平均速度         
        self.ring_num = 4     
        self.target_obs_shape = 8 * self.channel * self.ring_num 
        self.observation_space = MultiAgentObservationSpace([
            Dict({
                'target': Box(    #目标观测信息
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.target_obs_shape,),
                    dtype=np.float32
                ),
                'utility': Box(   #效用图观测信息
                    low=0.0,
                    high=1.0,
                    shape=self.utility_obs_shape,
                    dtype=np.float32
                ),
                'self_pos': Box(  # agent自身位置
                    low=np.array([self.x_min, self.y_min]),
                    high=np.array([self.x_max, self.y_max]),
                    shape=(2,),
                    dtype=np.float32
                )
            }) for _ in range(self.n_agents)]
        )

    #主要步骤代码
    def reset(self):
        self.done = False
        self.step_count = 0
        self.total_rewad = [0.0 for _ in range(self.n_agents)]
        self.agent_pos = {i: None for i in range(self.n_agents)}
        self.agent_Z_dicaer = [[[], []] for _ in range(self.n_agents)]
        self.trajectories = self.init_targrt()    #得到rajectories所有时间步的目标状态、agent位置和前两个时刻的直角坐标观测

         # 初始化每个 agent 的存储的目标的PHD状态
        self.state = [
            [
                0,                      # 权重
                np.zeros((1, 4)),       # 均值
                np.zeros((4, 4)),       # 协方差
                0                       # 粒子数量
            ] for _ in range(self.n_agents)
        ]
        #前两个时刻的直角坐标观测用于PHD、剩下的要转为规定的智能体观测形式
        observations = []
        for i in range(self.n_agents):
            # 获取当前位置和模型数据
            pos = self.agent_pos[i]
            model_data = model(pos[0], pos[1])

            for j in range(2):
                z_polar = observe_Fov(model_data, self.trajectories[j])   #极坐标观测数据
                Z_dicaer = polar2dicaer(z_polar, model_data)         #转化为直角坐标观测数据
                self.agent_Z_dicaer[i][j] = Z_dicaer           #获得智能体前两个时刻的直角坐标观测

            # 调用PHD进行目标状态估计
            phd = PHD(model_data, self.agent_Z_dicaer[i][0], self.agent_Z_dicaer[i][1], z_polar, self.state[i])
            X_now = phd.predict_update()
            self.state[i] = X_now
            state_draw, num_draw = State_extraction(X_now)

            target_obs = np.zeros(self.target_obs_shape, dtype=np.float32)
            utility_obs = np.zeros(self.utility_obs_shape, dtype=np.float32)
            self_pos = np.array(self.agent_pos[i], dtype=np.float32)  # shape: (2,)

            obs = {
                'target': target_obs,
                'utility': utility_obs,
                'self_pos': self_pos
            }
            observations.append(obs)
        #进行一步PHD更新观测数据、并更新搜索效用图数据


        return 
    
    def step(self, actions):
        """
        对每个agent获取观测,根据观测和状态获得动作,并更新位置、计算奖励。
        (这里只需要输入动作即可)
        观测包括目标状态估计、协方差矩阵和搜索效用
        返回观测、奖励、结束标志和额外信息。
        """
        self.step_count += 1
        obs_n = []
        rewards = []
        dones = []
        infos = []

        for agent_i in range(self.n_agents):
            action = actions[agent_i]

            # 更新 agent 位置
            self.update_agent_pos(agent_i, action)

            # 获取新观测（结构化）
            obs = self.get_agent_obs(agent_i)
            obs_n.append(obs)

            # 计算奖励（可自定义逻辑）
            reward = self.compute_reward(agent_i, obs)
            rewards.append(reward)
            self.total_rewad[agent_i] += reward

            # done 标志：默认仅在最大步数时终止
            done = self.step_count >= self.step
            dones.append(done)

            # info（可扩展）
            infos.append({
                'step': self.step_count
            })

        return obs_n, rewards, dones, infos
    def get_reward(self, agent_i, obs):
        return
    

    #第一步初始化
    def init_targrt(self): #得到rajectories所有时间步的目标状态、agent位置和前两个时刻的直角坐标观测
        """
        获得初始化agent位置和目标位置、用于生成新生分量的观测数据
        """
        targets_birth_time, targets_death_time, targets_start = targets(self.n_targets)
        trajectories, _ = target_CV(targets_birth_time, targets_death_time, targets_start, 
                                                self.step, 
                                                self.x_min, self.x_max, 
                                                self.y_min, self.y_max, 
                                                noise=True)                    #trajectories是k时间步所有目标状态，targets_tracks是第i个目标所有时间的状态
        for agent_i in range(self.n_agents):             #初始化每个agent的位置
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            self.agent_pos[agent_i] = np.array([x, y])
            # 初始化每个agent的观测数据（用于生成新生分量）
            self.agent_Z_dicaer[agent_i] = [[], []]
            model_data = model(x, y)
            for i in range(2):
                z_polar = observe_Fov(model_data, trajectories[i])   #极坐标观测数据
                Z_dicaer = polar2dicaer(z_polar, model_data)         #转化为直角坐标观测数据
                self.agent_Z_dicaer[agent_i][i] = Z_dicaer           #获得智能体前两个时刻的直角坐标观测
        return trajectories 

    #更新智能体位置和观测
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

        # ===== 构建目标状态观测向量 =====
        section = 8 * self.obs_Rnums
        region_counts = np.zeros(section, dtype=np.int32)
        region_x_sums = np.zeros(section, dtype=np.float32)
        region_y_sums = np.zeros(section, dtype=np.float32)

        for i in range(section):
            count = 0
            for target in state_draw:
                x, y = target[0], target[1]
                if self.is_in_region(x, y, model_data, region_index=i, agent_pos=pos):
                    count += 1
                    region_x_sums[i] += x
                    region_y_sums[i] += y
                    region_counts[i] += 1

        # 避免除 0 的处理
        region_x_means = np.where(region_counts > 0, region_x_sums / region_counts, 0.0)
        region_y_means = np.where(region_counts > 0, region_y_sums / region_counts, 0.0)

        # 拼接目标状态向量
        target_obs = np.concatenate([region_counts, region_x_means, region_y_means], dtype=np.float32)

        # ===== 效用图部分 =====
        utility_obs = self.agent_utility_maps[agent_i].flatten().astype(np.float32)

        # ===== 返回结构化观测 =====
        return {
            'target': target_obs,
            'utility': utility_obs
        }

    #下面都是功能函数
    def build_utility_obs(self, agent_pos, utility_map): #叠加智能体位置到utility_obs
        """
        构造 shape=(2, 20, 20) 的 utility 观测，
        通道0: 效用图
        通道1: agent位置的 one-hot mask
        """
        utility = np.clip(utility_map, 0.0, 1.0)  # (20, 20)

        # 构建 agent mask
        agent_mask = np.zeros_like(utility)       # (20, 20)
        x_idx = int((agent_pos[0] - self.x_min) / (self.x_max - self.x_min) * self.utilityMap_M)
        y_idx = int((agent_pos[1] - self.y_min) / (self.y_max - self.y_min) * self.utilityMap_N)

        # 边界检查
        x_idx = np.clip(x_idx, 0, self.utilityMap_M - 1)
        y_idx = np.clip(y_idx, 0, self.utilityMap_N - 1)
        agent_mask[y_idx, x_idx] = 1.0  # 注意行是 y，列是 x

        # 拼接通道
        utility_obs = np.stack([utility, agent_mask], axis=0)  # shape=(2, 20, 20)
        return utility_obs
    def get_action_meanings(self, agent_i=None):         #根据action索引返回对应的含义
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
    def is_in_region(self, x, y, model_data, region_index, agent_pos):   #判断目标是否在指定的区域内
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