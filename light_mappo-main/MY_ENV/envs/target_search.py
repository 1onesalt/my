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
    def __init__(self, x_min = -1000, x_max = 1000, y_min = -1000, y_max = 1000, n_agent = 3, n_target = 5, max_steps = 100):
        # 环境参数
        self.x_min = x_min          
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.n_agents = n_agent
        self.n_targets = n_target
        self.max_steps = max_steps
        self.agent_step_size = 7

        # 初始化为 None 的变量
        self.step_count = None
        self.total_rewad = None
        self.done = None
        self.trajectories = None

        # 初始化为空的数据结构
        self.state = None
        self.agent_pos = None
        self.agent_Z_dicaer = None
        self.utility_map = None
        self.target_obs = None

        # 观测空间相关参数
        self.utility_channel = 2
        self.utilityMap_M = 20     
        self.utilityMap_N = 20 
        self.utility_obs_shape = (self.utility_channel, self.utilityMap_M, self.utilityMap_N)

        self.channel = 3                  
        self.ring_num = 4     
        self.sector_num = 8
        self.target_obs_shape = (self.channel, self.ring_num, self.sector_num) 

        # 动作空间和观测空间
        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(9) for _ in range(self.n_agents)]
        )
        
        self.observation_space = MultiAgentObservationSpace([
            Dict({
                'target': Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.target_obs_shape,
                    dtype=np.float32
                ),
                'utility': Box(
                    low=0.0,
                    high=1.0,
                    shape=self.utility_obs_shape,
                    dtype=np.float32
                ),
                'self_pos': Box(
                    low=np.array([self.x_min, self.y_min]),
                    high=np.array([self.x_max, self.y_max]),
                    shape=(2,),
                    dtype=np.float32
                )
            }) for _ in range(self.n_agents)]
        )

    #主要步骤代码
    def reset(self):
        # 重置环境状态
        self.done = False
        self.step_count = 0
        self.total_rewad = [0.0 for _ in range(self.n_agents)]

        # 重置智能体位置和观测数据
        self.agent_pos = {i: None for i in range(self.n_agents)}
        self.agent_Z_dicaer = [[[], []] for _ in range(self.n_agents)]

        # 重置效用图和目标观测
        self.utility_map = {
            i: np.zeros((self.utility_channel, self.utilityMap_M, self.utilityMap_N), dtype=np.float32) 
            for i in range(self.n_agents)
        }
        self.target_obs = {
            i: np.zeros((self.channel, self.ring_num, self.sector_num), dtype=np.float32)
            for i in range(self.n_agents)
        }


        # 重置 PHD 状态
        self.state = [
            [
                0,                      # 权重
                np.zeros((1, 4)),      # 均值
                np.zeros((4, 4)),      # 协方差
                0                      # 粒子数量
            ] for _ in range(self.n_agents)
        ]
        print("初始化PHD状态:", self.state)
        # 获取轨迹并初始化
        self.trajectories = self.init_targrt()

        observations = []
        for i in range(self.n_agents):
            # 获取当前位置和模型数据
            pos = self.agent_pos[i]
            model_data = model(pos[0], pos[1])

            for j in range(2):
                z_polar = observe_Fov(model_data, self.trajectories[j])   #极坐标观测数据
                Z_dicaer = polar2dicaer(z_polar, model_data)         #转化为直角坐标观测数据
                self.agent_Z_dicaer[i][j] = Z_dicaer           #获得智能体前两个时刻的直角坐标观测

            # PHD 更新
            phd = PHD(model_data, self.agent_Z_dicaer[i][0], self.agent_Z_dicaer[i][1], z_polar, self.state[i])
            X_now = phd.predict_update()
            print( "第{}个智能体的目标状态: {}".format(i, X_now) )
            self.state[i] = X_now
        
            state_draw, num_draw = State_extraction(X_now)    #这里得到的就是粒子状态和粒子数量

            # 构建观测
            target_obs = self.build_target_obs(pos, state_draw, self.ring_num, self.sector_num, self.channel, model_data["obverser_d"])
            self.target_obs[i] = target_obs


            # 更新效用图
            self.utility_map[i] = self.update_utility_map_vectorized(self.utility_map[i], pos, model_data["obverser_d"], 0.8,
                                  self.x_min, self.x_max, self.y_min, self.y_max)
            # 然后叠加agent位置mask，生成最终观测的utility通道输入
            utility_obs = self.build_utility_obs(pos, self.utility_map[i])
            
            obs = {
                'target': target_obs,  
                'utility': utility_obs,  
                'self_pos': np.array(pos, dtype=np.float32)
            }
            observations.append(obs)
        print("环境重置完成，所有智能体已初始化。")
        return observations
    
    def step(self, actions):
        """
        输入: actions - 每个智能体的动作列表   actions 是一个长度等于智能体数量的列表（或元组），每个元素是该智能体在当前时间步执行的离散动作索引
        返回: (obs_n, rewards, dones, infos)
        """
        self.step_count += 1
        obs_n = []
        rewards = []
        dones = []
        infos = []

        # 遍历每个智能体
        for i in range(self.n_agents):
            # 1. 更新智能体位置
            self.update_agent_pos(i, actions[i])
            pos = self.agent_pos[i]
            
            # 2. 获取观测数据
            model_data = model(pos[0], pos[1])
            z_polar = observe_Fov(model_data, self.trajectories[self.step_count])
            Z_dicaer = polar2dicaer(z_polar, model_data)

            # 3. 更新智能体的观测历史
            self.agent_Z_dicaer[i][0] = self.agent_Z_dicaer[i][1]
            self.agent_Z_dicaer[i][1] = Z_dicaer

            # 4. PHD 滤波更新
            phd = PHD(model_data, self.agent_Z_dicaer[i][0], self.agent_Z_dicaer[i][1], 
                    z_polar, self.state[i])
            X_now = phd.predict_update()
            self.state[i] = X_now
            state_draw, num_draw = State_extraction(X_now)

            # 5. 构建目标观测
            target_obs = self.build_target_obs(
                pos, state_draw, self.ring_num, self.sector_num, 
                self.channel, model_data["obverser_d"]
            )
            self.target_obs[i] = target_obs

            # 6. 更新效用图
            self.utility_map[i] = self.update_utility_map_vectorized(
                self.utility_map[i], pos, model_data["obverser_d"], 0.8,
                self.x_min, self.x_max, self.y_min, self.y_max
            )
            utility_obs = self.build_utility_obs(pos, self.utility_map[i])

            # 7. 构建完整观测
            obs = {
                'target': target_obs,
                'utility': utility_obs,
                'self_pos': np.array(pos, dtype=np.float32)
            }
            obs_n.append(obs)

            # 8. 计算奖励
            reward = self.compute_reward(i, X_now, state_draw, actions[i])
            rewards.append(reward)
            self.total_rewad[i] += reward

            # 9. 判断是否结束
            done = (self.step_count >= self.max_steps)
            dones.append(done)

            # 10. 额外信息
            info = {
                'step': self.step_count,
                'agent_pos': pos,
                'num_targets': num_draw
            }
            infos.append(info)

        # 11. 更新环境完成状态
        self.done = all(dones)

        return obs_n, rewards, dones, infos

    def compute_reward(self, agent_i, X_now, state_draw, action):
        """
        agent_i: 智能体索引
        X_now: 当前时刻PHD滤波输出的目标估计列表，格式包含状态和协方差，如 [(w, m, P, j), ...]
        prev_X: 上一时刻目标估计，用于计算新目标发现和丢失
        action: 当前智能体采取的动作索引

        返回：标量奖励
        """
        pos = np.array(self.agent_pos[agent_i])  # 智能体当前位置

        dist_rewards = []
        for target_state in state_draw:
            target_pos = np.array([target_state[0], target_state[2]])  # 取 x,y 坐标
            dist = np.linalg.norm(pos - target_pos)
            dist_reward = np.exp(-0.1 * dist)
            dist_rewards.append(dist_reward)
        dist_reward_total = np.sum(dist_rewards)

        # 解包 X_now
        weights = X_now[0]  # 权重列表
        means = X_now[1]    # 均值列表
        covs = X_now[2]     # 协方差列表
        num = X_now[3]      # 粒子数量

        # 1. 智能体与目标距离奖励
        dist_rewards = []
        cov_rewards = []

        for i in range(len(weights)):
            if weights[i] > 0:  # 只考虑有效粒子
                P = covs[i]     # 协方差
           
                # 协方差奖励
                pos_cov = np.array([[P[0,0], P[0,2]], 
                                [P[2,0], P[2,2]]])  # 提取位置相关的协方差
                cov_trace = np.trace(pos_cov)
                cov_reward = np.exp(-0.5 * cov_trace)
                cov_rewards.append(cov_reward)
                #detected_ids.add(i)  # 使用索引作为ID

        cov_reward_total = np.sum(cov_rewards)

        # 2. 长期跟踪奖励
        # if not hasattr(self, "tracking_steps"):
        #     self.tracking_steps = {i: {} for i in range(self.n_agents)}

        # detected_ids = set(range(len(state_draw))) 
        # for tid in detected_ids:
        #     self.tracking_steps[agent_i][tid] = self.tracking_steps[agent_i].get(tid, 0) + 1
        # # 没检测到的目标重置计数
        # for tid in list(self.tracking_steps[agent_i].keys()):
        #     if tid not in detected_ids:
        #         self.tracking_steps[agent_i][tid] = 0

        # longtrack_reward = sum(0.05 * self.tracking_steps[agent_i][tid] for tid in detected_ids)

        # 3. 搜索效用奖励
        utility_map = self.utility_map[agent_i]
        M, N = utility_map.shape[1], utility_map.shape[2]
        x_idx = int((pos[0] - self.x_min) / (self.x_max - self.x_min) * M)
        y_idx = int((pos[1] - self.y_min) / (self.y_max - self.y_min) * N)
        x_idx = np.clip(x_idx, 0, M - 1)
        y_idx = np.clip(y_idx, 0, N - 1)
        utility_reward = np.sum(utility_map[:, x_idx, y_idx])

        # 4. 动作代价
        action_cost = 0.0 if action == 8 else -0.05

        # 6. 使用提取状态计算发现/丢失目标
        num_targets_now = len(state_draw)
        if hasattr(self, 'prev_num_targets'):
            num_targets_prev = self.prev_num_targets.get(agent_i, 0)
        else:
            self.prev_num_targets = {}
            num_targets_prev = 0
            
        self.prev_num_targets[agent_i] = num_targets_now
        new_targets = max(num_targets_now - num_targets_prev, 0)
        lost_targets = max(num_targets_prev - num_targets_now, 0)
        discovery_reward = new_targets * 0.5
        lost_penalty = -lost_targets * 0.5

        # 组合总奖励
        total_reward = (
            dist_reward_total * 1.0 +
            cov_reward_total * 0.5 +
            # longtrack_reward +
            utility_reward * 0.3 +
            action_cost +
            discovery_reward +
            lost_penalty
        )

        return total_reward
            
    #第一步初始化
    def init_targrt(self): #得到rajectories所有时间步的目标状态、agent位置和前两个时刻的直角坐标观测
        """
        获得初始化agent位置和目标位置、用于生成新生分量的观测数据
        """
        targets_birth_time, targets_death_time, targets_start = targets(self.n_targets)
        trajectories, _ = target_CV(targets_birth_time, targets_death_time, targets_start, 
                                                self.max_steps, 
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
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

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
        # 第一个通道保持不变（效用值）
        utility_map[0] = np.clip(utility_map[0], 0.0, 1.0)

        # 第二个通道（智能体位置 mask）
        utility_map[1].fill(0.0)  # 清空第二个通道
        x_idx = int((agent_pos[0] - self.x_min) / (self.x_max - self.x_min) * self.utilityMap_N)
        y_idx = int((agent_pos[1] - self.y_min) / (self.y_max - self.y_min) * self.utilityMap_M)
        x_idx = np.clip(x_idx, 0, self.utilityMap_N - 1)
        y_idx = np.clip(y_idx, 0, self.utilityMap_M - 1)
        utility_map[1, y_idx, x_idx] = 1.0
        return utility_map
    
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

    def build_target_obs(self, agent_pos, target_positions, ring_num, sector_num, channel, view_radius):#初始化观测中的目标
        """
        agent_pos: (x, y) 智能体位置
        targets: [(x, y), ...] 目标位置列表
        fov_radius: 视域半径
        ring_num: 环数
        channel: 每个扇区的通道数
        """
        # 8 扇区，每圈 8 个格子
        target_obs = np.zeros((channel, ring_num, sector_num), dtype=np.float32)

        count_map = np.zeros((ring_num, sector_num), dtype=np.float32)              #目标数量
        dist_sum_map = np.zeros((ring_num, sector_num), dtype=np.float32)           #距离
        min_dist_map = np.full((ring_num, sector_num), np.inf, dtype=np.float32)    #最小距离

        for target_state in target_positions:
            tx, ty = target_state[0], target_state[2] 
            dx = tx - agent_pos[0]
            dy = ty - agent_pos[1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > view_radius or dist == 0:
                continue

            ring_idx = int(dist / (view_radius / ring_num))
            ring_idx = min(ring_idx, ring_num - 1)

            angle = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
            sector_idx = int(angle / (2 * np.pi / sector_num))

            count_map[ring_idx, sector_idx] += 1
            dist_sum_map[ring_idx, sector_idx] += dist
            min_dist_map[ring_idx, sector_idx] = min(min_dist_map[ring_idx, sector_idx], dist)

        # 填充 target_obs
        for s in range(sector_num):
            for r in range(ring_num):
                cnt = count_map[r, s]
                if cnt > 0:
                    target_obs[0, r, s] = cnt  # 目标数量
                    target_obs[1, r, s] = (dist_sum_map[r, s] / cnt) / view_radius  # 归一化平均距离
                    target_obs[2, r, s] = min_dist_map[r, s] / view_radius  # 归一化最小距离

        return target_obs
    
    def update_utility_map_vectorized(self, utility_map, agent_pos, view_radius, decay_factor, #初始化搜索概率图
                                  x_min, x_max, y_min, y_max):
        """
        向量化更新搜索概率图
        utility_map: 概率图 (C, M, N)
        agent_pos: (x, y) 智能体位置
        view_radius: 视域半径
        decay_factor: 衰减因子 (0 < decay_factor <= 1)
        """
        M, N = self.utilityMap_M, self.utilityMap_N

        # 生成网格坐标
        # 1. 计算每个网格的实际大小
        cell_size_x = (x_max - x_min) / N
        cell_size_y = (y_max - y_min) / M
        
        # 2. 生成网格中心坐标
        x_centers = np.linspace(x_min + cell_size_x/2, x_max - cell_size_x/2, N)
        y_centers = np.linspace(y_min + cell_size_y/2, y_max - cell_size_y/2, M)
        X, Y = np.meshgrid(x_centers, y_centers)

        # 计算到智能体的距离
        dist_map = np.sqrt((X - agent_pos[0])**2 + (Y - agent_pos[1])**2)

        # 更新第一个通道（效用图）
        mask_in_view = dist_map <= view_radius
        utility_map[0][mask_in_view] = 1.0
        mask_decay = ~mask_in_view & (utility_map[0] > 0)
        utility_map[0][mask_decay] *= decay_factor
        
        print(f"Agent pos: {agent_pos}")
        print(f"View radius: {view_radius}")
        print(f"Cells in view: {np.sum(mask_in_view)}")
        print(f"Non-zero cells: {np.sum(utility_map[0] > 0)}")

        return utility_map
    
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