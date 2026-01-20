import copy
import logging
import random
import gym
import numpy as np
import torch
from gym.spaces import Dict, Box

from MY_ENV.utils.action_space import MultiAgentActionSpace
from MY_ENV.utils.observation_space import MultiAgentObservationSpace
from MY_ENV.utils.draw import draw_grid, fill_cell, write_cell_text

from gym import spaces
from gym.utils import seeding
from MY_ENV.envs.target_model2 import model, targets, target_CV, observe_Fov, polar2dicaer
from MY_ENV.envs.PHD import PHD, State_extraction, generate
from MY_ENV.envs.phd_utils import PHDFeatureExtractor

class target_search(gym.Env):
    def __init__(self, x_min = -1000, x_max = 1000, y_min = -1000, y_max = 1000, n_agent = 3, n_target = 3, max_steps = 100):
        # 环境参数
        self.x_min = x_min          #环境边界
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.n_agents = n_agent
        self.n_targets = n_target
        self.max_steps = max_steps

        # 模型参数 (从 target_model 中获取或定义)
        self.model_params = model()
        self.sensor_r = self.model_params["obverser_d"]  # 视域半径 d
        self.agent_v = 10.0

        # # --- PHD 特征提取器配置 ---
        # self.phd_h = 64
        # self.phd_w = 64
        # phd_config = {
        #     'cnn_h': self.phd_h,       # 张量高度
        #     'cnn_w': self.phd_w,       # 张量宽度
        #     'r_max': 800.0,            # 视域半径
        #     'max_speed': 10.0          # 估计的最大目标速度，用于归一化
        # }
        # self.feature_extractor = PHDFeatureExtractor(phd_config, device='cpu')

        # --- 2. 极坐标网格参数 (Section 4.1) ---
        self.grid_U = 16    # 极径方向划分数量
        self.grid_V = 16    # 极角方向划分数量
        self.grid_C = 4     # 通道数: [强度, 径向速度, 切向速度, 协方差]    

        self.n_actions = 7 # 例如:大幅左转, 小幅左转, 直行, 小幅右转, 大幅右转等，或均匀分布
        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(self.n_actions) for _ in range(self.n_agents)]
        )    
        self.angle_adjustments = np.linspace(-np.pi/4, np.pi/4, self.n_actions) # 示例: -45度 到 45度

        
        # 历史目标基数窗口 (用于 r_new [cite: 79])
        self.L_window = 10
        #self.cardinality_history = collections.deque(maxlen=self.L_window)
        self.max_history_cardinality = 0

        # 初始化为 None 的变量
        self.step_count = None
        self.total_rewad = None
        self.done = None
        self.trajectories = None

        # 初始化为空的数据结构
        self.state = None
        self.agent_pos = None
        self.agent_Z_dicaer = None

        self.observation_space = MultiAgentObservationSpace([
                    Dict({
                        'polar_grid': Box(
                            low=-np.inf, high=np.inf,
                            shape=(self.grid_U, self.grid_V, self.grid_C),
                            dtype=np.float32
                        ),
                        'self_state': Box(
                            low=-np.inf, high=np.inf,
                            shape=(3,), # x, y, theta
                            dtype=np.float32
                        )
                    }) for _ in range(self.n_agents)]
                )

        # self.shared_observation_space = MultiAgentObservationSpace([
        #     Dict({
        #         'phd_heatmap': Box(
        #             low=-np.inf, 
        #             high=np.inf, # log变换后范围可能较大，或者使用 low=-1, high=1 如果只看速度通道
        #             shape=self.heatmap_shape,
        #             dtype=np.float32
        #         ),
        #         'self_pos': Box(  
        #             low=np.array([self.x_min, self.y_min]),
        #             high=np.array([self.x_max, self.y_max]),
        #             shape=(2 * self.n_agents,),
        #             dtype=np.float32
        #         )
        #     }) for _ in range(self.n_agents)]
        # )       

        # 奖励因子 (公式 29)
        self.lambda1 = 1.0  # r_track
        self.lambda2 = 2.0  # r_new
        self.lambda3 = 0.5  # r_overlap
        self.lambda4 = 0.5  # r_bound
        self.rho_star = 0.1 # 理想重叠率
        self.delta_overlap = 1.0 # 重叠奖励权重
    
    def reset(self):
        """
        重置环境状态:
        1、目标轨迹的重置
        2、智能体位置的重置(状态)
        3、智能体观测的重置(热力图观测(phd状态)、自身位置)，并获取初始观测
        4、回合数、步数、奖励重置
        """

        #回合数、步数、奖励重置
        self.done = False
        self.step_count = 0
        self.total_rewad = [0.0 for _ in range(self.n_agents)]

        # 智能体观测的重置
        self.agent_pos = {i: None for i in range(self.n_agents)}
        self.agent_headings = {}
        self.agent_Z_dicaer = [[[], []] for _ in range(self.n_agents)]

        self.state = [
            [
                0,                      # 权重
                np.zeros((1, 4)),      # 均值
                np.zeros((4, 4)),      # 协方差
                0                      # 粒子数量
            ] for _ in range(self.n_agents)
        ]

        # 获取轨迹并初始化
        self.trajectories = self.init_targrt()
        for agent_i in range(self.n_agents):             #初始化每个agent的位置
            self.agent_pos[i] = np.array([
                random.uniform(self.x_min + 100, self.x_max - 100),
                random.uniform(self.y_min + 100, self.y_max - 100)
            ])
            self.agent_headings[i] = random.uniform(-np.pi, np.pi)
            self.cardinality_history[i] = [0] * self.history_window # 重置历史基数

        return self._get_observations()
    
        # #获取初始观测
        # observations = []
        # for i in range(self.n_agents):
        #     # 获取当前位置和模型数据
        #     pos = self.agent_pos[i]
        #     heading = self.agent_headings[i]
        #     model_data = model(pos[0], pos[1], self.max_steps)

        #     for j in range(2):
        #         z_polar = observe_Fov(model_data, self.trajectories[j])   #极坐标观测数据
        #         Z_dicaer = polar2dicaer(z_polar, model_data)         #转化为直角坐标观测数据
        #         self.agent_Z_dicaer[i][j] = Z_dicaer           #获得智能体前两个时刻的直角坐标观测

        #     # PHD 更新
        #     phd = PHD(model_data)
        #     phd.init_params(self.agent_Z_dicaer[i][0], self.agent_Z_dicaer[i][1], z_polar, self.state[i])
        #     phd.w_new, phd.m_new, phd.P_new, phd.J_new = generate(phd.z_lastD, phd.nums_z_lastD, phd.z_nowD, phd.nums_z_nowD, phd.Vx_thre, phd.Vy_thre)
        #     X_now = phd.predict_update()
        #     print( "第{}个智能体的目标状态: {}".format(i, X_now) )
        #     self.state[i] = X_now

        #     # --- 1. 生成新的 PHD 热力图观测 (CNN Input) ---
        #     agent_pose = [pos[0], pos[1], heading]
        #     obs_tensor = self.feature_extractor.process(X_now, agent_pose)  # process 返回的是 Torch Tensor (3, 64, 64)
        #     phd_heatmap_np = obs_tensor.cpu().numpy()  # 转为 Numpy 数组以符合 Gym 规范
            
        #     obs = {
        #         'phd_heatmap': phd_heatmap_np,  
 
        #         'self_pos': np.array(pos, dtype=np.float32)
        #     }
        #     observations.append(obs)

        # print("环境重置完成，所有智能体已初始化。")
        # return observations
    
    def step(self, actions):
        """
        输入: actions - 每个智能体的动作列表   actions 是一个长度等于智能体数量的列表（或元组），每个元素是该智能体在当前时间步执行的离散动作索引
        返回: (obs_n, rewards, dones, infos)
        """
        # 防止越界读取轨迹
        if self.step_count < self.max_steps - 1:
            self.step_count += 1
        obs_n = []
        share_obs_n = []
        rewards = []
        dones = []
        infos = []

        # 临时存储所有智能体的PHD状态，用于生成全局热力图
        all_agents_phd_components = {
            'w': [], 'm': [], 'P': []
        }   


        # 遍历每个智能体
        for i in range(self.n_agents):
            action = actions[i] 

            # 1. 更新智能体位置
            self.update_agent_pos(i, actions[i])
            #self._update_heading(i, action) #更新朝向

            pos = self.agent_pos[i]
            heading = self.agent_headings[i]
            
            # 2. 获取观测数据
            model_data = model(pos[0], pos[1], self.max_steps)
            z_polar = observe_Fov(model_data, self.trajectories[self.step_count])
            Z_dicaer = polar2dicaer(z_polar, model_data)

            # 3. 更新智能体的观测历史
            self.agent_Z_dicaer[i][0] = self.agent_Z_dicaer[i][1]
            self.agent_Z_dicaer[i][1] = Z_dicaer

            # 4. PHD 滤波更新
            phd = PHD(model_data)
            phd.init_params(self.agent_Z_dicaer[i][0], self.agent_Z_dicaer[i][1], z_polar, self.state[i])
            phd.w_new, phd.m_new, phd.P_new, phd.J_new = generate(phd.z_lastD, phd.nums_z_lastD, phd.z_nowD, phd.nums_z_nowD, phd.Vx_thre, phd.Vy_thre)
            X_now = phd.predict_update()
            self.state[i] = X_now

            state_draw, num_draw = State_extraction(X_now)

            # --- 5. 生成 PHD 热力图观测 (Embedding Point) ---
            agent_pose = [pos[0], pos[1], heading]
            # 生成 Tensor 并转 Numpy
            obs_tensor = self.feature_extractor.process(X_now, agent_pose)
            phd_heatmap_np = obs_tensor.cpu().numpy()


            self.utility_map[i], search_utility_reward = self.update_utility_map_vectorized(
                self.utility_map[i], pos, model_data["obverser_d"], 0.8, # 0.8参数在这里失效了，但为了接口兼容先留着
                self.x_min, self.x_max, self.y_min, self.y_max
            )
            utility_obs = self.build_utility_obs(pos, self.utility_map[i])

            # 7. 构建完整观测
            obs = {
                'phd_heatmap': phd_heatmap_np,
                'utility': utility_obs,
                'self_pos': np.array(pos, dtype=np.float32)
            }
            obs_n.append(obs)

            # 8. 计算奖励
            reward = self.compute_reward(i, X_now, state_draw, actions[i], search_utility_reward)
            rewards.append(reward)
            self.total_rewad[i] += reward

            # 9. 判断是否结束
            done = (self.step_count >= self.max_steps)
            dones.append(done)

            # 10. 额外信息
            info = {
                'step': self.step_count,
                'agent_pos': pos,
                'agent_heading': heading,
                'num_targets': num_draw
            }
            infos.append(info)

        # 11. 更新环境完成状态
        self.done = all(dones)

        return obs_n, rewards, dones, infos

    def _update_heading(self, agent_i, action):
            """
            根据动作更新智能体的朝向 (Heading)
            只有在移动时才改变朝向，NOOP 保持原朝向
            """
            # 动作定义参考 ACTION_MEANING
            # 0: UP (90度)
            # 1: DOWN (-90度 或 270度)
            # 2: LEFT (180度)
            # 3: RIGHT (0度)
            # 4: LEFT_UP (135度)
            # 5: RIGHT_UP (45度)
            # 6: LEFT_DOWN (-135度 或 225度)
            # 7: RIGHT_DOWN (-45度 或 315度)
            # 8: NOOP (保持不变)
            
            heading_map = {
                0: 90.0,
                1: -90.0,
                2: 180.0,
                3: 0.0,
                4: 135.0,
                5: 45.0,
                6: -135.0,
                7: -45.0
            }
            
            if action in heading_map:
                self.agent_headings[agent_i] = heading_map[action]


    def _compute_reward(self, agent_i, phd_state):
        """
        实现论文 4.3 节的奖励函数
        R = lambda1 * r_track + lambda2 * r_new + lambda3 * r_overlap + lambda4 * r_bound
        """
        weights, _, covs, n_est = phd_state
        
        # 1. 跟踪精度奖励 r_track (公式 13)
        # r_track = - sum(tr(P)) + beta * (N_k+1 - N_k)
        # 注意：论文公式是 -sum(tr)，表示惩罚不确定性。
        # 这里的 N_k+1 - N_k 项，如果是为了防止丢失目标，当 N 减小时应惩罚。
        # 这里实现为：如果不确定性低，奖励高；如果基数增加，奖励高。
        trace_sum = 0
        for w, P in zip(weights, covs):
            if w > 0.1: # 仅统计有效分量
                trace_sum += w * (P[0,0] + P[2,2])
        
        prev_N = self.cardinality_history[agent_i][-1]
        delta_N = n_est - prev_N
        beta = 1.0 # 权重系数，需调优
        r_track = - trace_sum + beta * delta_N
        
        # 2. 新目标发现奖励 r_new (公式 24)
        history_max = max(self.cardinality_history[agent_i]) if self.cardinality_history[agent_i] else 0
        r_new = 0
        if n_est > history_max:
            r_new = 1.0 # alpha, 固定正向奖励
        
        # 更新历史
        self.cardinality_history[agent_i].append(n_est)
        if len(self.cardinality_history[agent_i]) > self.history_window:
            self.cardinality_history[agent_i].pop(0)

        # 3. 视域重叠度奖励 r_overlap (公式 26, 27)
        # 计算该智能体与其他智能体的重叠率
        r_overlap = 0
        current_pos = self.agent_pos[agent_i]
        for j in range(self.n_agents):
            if i == j: continue
            other_pos = self.agent_pos[j]
            dist = np.linalg.norm(current_pos - other_pos)
            
            # 计算两个圆的重叠面积
            # d = sensor_r
            d = self.sensor_r
            if dist >= 2 * d:
                overlap_area = 0
            else:
                # 圆重叠面积公式
                angle1 = 2 * np.arccos(dist / (2 * d))
                overlap_area = 0.5 * d**2 * (angle1 - np.sin(angle1)) * 2 # 对称
            
            area_i = np.pi * d**2
            rho_i = overlap_area / area_i
            
            # 高斯函数奖励/惩罚
            # 旨在维持一个理想的重叠率 rho_star
            sigma = 0.1
            r_overlap_j = self.delta_overlap * np.exp( - (rho_i - self.rho_star)**2 / (2 * sigma**2) )
            r_overlap += r_overlap_j

        # 4. 边界惩罚 r_bound (公式 28)
        # 计算到最近边界的距离
        x, y = current_pos
        d_ib = min(x - self.x_min, self.x_max - x, y - self.y_min, self.y_max - y)
        d_sensor = self.sensor_r
        
        if d_ib <= d_sensor:
            r_bound = -0.5 * (d_sensor - d_ib) / d_sensor
        else:
            r_bound = 0
            
        # 总奖励 (公式 29)
        total_reward = (self.lambda1 * r_track + 
                        self.lambda2 * r_new + 
                        self.lambda3 * r_overlap + 
                        self.lambda4 * r_bound)
                        
        return total_reward
            
    #第一步初始化
    def init_targrt(self): #得到rajectories所有时间步的目标状态、agent位置和前两个时刻的直角坐标观测
        """
        获得初始化agent位置和目标位置、用于生成新生分量的观测数据
        """
        targets_birth_time, targets_death_time, targets_start = targets(self.n_targets, self.max_steps)
        trajectories, _ = target_CV(targets_birth_time, targets_death_time, targets_start, 
                                                self.max_steps, 
                                                self.x_min, self.x_max, 
                                                self.y_min, self.y_max, 
                                                noise=True)                    #trajectories是k时间步所有目标状态，targets_tracks是第i个目标所有时间的状态

        return trajectories 
    
    def _get_observations(self):
        obs_n = []
        for i in range(self.n_agents):
            # 构建极坐标网格特征 (Section 4.1.2)
            polar_grid = self._build_polar_grid(self.state[i], self.agent_pos[i], self.agent_headings[i])
            
            obs = {
                'polar_grid': polar_grid,
                'self_state': np.array([self.agent_pos[i][0], self.agent_pos[i][1], self.agent_headings[i]], dtype=np.float32)
            }
            obs_n.append(obs)
        return obs_n

    def _build_polar_grid(self, phd_state, agent_pos, agent_heading):
        """
        根据公式 (16)-(21) 构建 U x V x C 的特征张量
        phd_state: [weights, means, covs, num]
        """
        weights, means, covs, _ = phd_state
        
        grid = np.zeros((self.grid_U, self.grid_V, self.grid_C), dtype=np.float32)
        
        # 如果没有目标分量，直接返回零张量
        if len(weights) == 0:
            return grid
            
        # 预计算网格参数
        r_step = self.sensor_r / self.grid_U
        theta_step = 2 * np.pi / self.grid_V
        
        for w, m, P in zip(weights, means, covs):
            if w < 1e-4: continue # 忽略极小权重
            
            # 目标状态 m = [x, vx, y, vy] (假设 PHD 输出格式)
            # 相对位置
            dx = m[0] - agent_pos[0]
            dy = m[2] - agent_pos[1]
            
            # 转换到极坐标 (公式 14)
            rho = np.sqrt(dx**2 + dy**2)
            # 相对航向角的角度
            theta_global = np.arctan2(dy, dx)
            theta_local = theta_global - agent_heading
            # 归一化到 [0, 2pi) 用于索引
            theta_local = (theta_local + 2*np.pi) % (2*np.pi)
            
            if rho >= self.sensor_r: continue
            
            # 计算网格索引 u, v
            u = int(rho / r_step)
            v = int(theta_local / theta_step)
            u = np.clip(u, 0, self.grid_U - 1)
            v = np.clip(v, 0, self.grid_V - 1)
            
            # 累加权重 (公式 16) - Channel 0: Intensity
            grid[u, v, 0] += w
            
            # 计算局部坐标系速度 (公式 17, 18)
            # 旋转矩阵 R^T * v_global
            vx_global, vy_global = m[1], m[3]
            c, s = np.cos(agent_heading), np.sin(agent_heading)
            # 旋转矩阵 R = [[c, -s], [s, c]]
            # 逆旋转(投影到本体轴): v_local = R^T * v
            v_long = c * vx_global + s * vy_global # 纵向 (沿航向)
            v_lat = -s * vx_global + c * vy_global # 横向 (垂直航向)
            
            # 加权累加速度 - Channel 1 & 2 (暂时只累加分子，最后归一化)
            grid[u, v, 1] += w * v_long
            grid[u, v, 2] += w * v_lat
            
            # 协方差迹 (公式 21) - Channel 3
            # 提取位置相关的协方差子矩阵的迹
            pos_cov_trace = P[0,0] + P[2,2] 
            grid[u, v, 3] += w * pos_cov_trace

        # 归一化 (公式 19, 20, 21)
        # 利用 Channel 0 (Intensity) 作为分母
        epsilon = 1e-6
        denominator = grid[:, :, 0] + epsilon
        
        grid[:, :, 1] /= denominator
        grid[:, :, 2] /= denominator
        grid[:, :, 3] /= denominator # 协方差也做加权平均
        
        return grid

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

    def get_agent_obs(self, agent_i):   #？
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
        mask_in_view = dist_map <= view_radius

        # 3. 计算【归一化】的覆盖奖励
        # 3a. 计算理论最大格子数 (这是一个常数，可以预计算)
        avg_resolution = (cell_size_x + cell_size_y) / 2.0
        radius_in_grid = view_radius / avg_resolution
        max_cells_in_view = np.pi * (radius_in_grid ** 2)
        max_cells_in_view = max(max_cells_in_view, 1.0) # 防止除0

        # 3b. 计算原始和
        raw_reward_sum = np.sum(utility_map[0][mask_in_view])

        # 3c. 归一化 (得到 0.0 ~ 1.0 之间的值)
        normalized_reward = raw_reward_sum / max_cells_in_view

        # 4. 更新地图
        # A. 观测区：不确定性消除，重置为 0
        utility_map[0][mask_in_view] = 0.0

        # B. 未观测区：不确定性随着时间增长 (Time-Aging)
        mask_not_in_view = ~mask_in_view
        growth_rate = 0.01  # 每一步增长 0.01
        utility_map[0][mask_not_in_view] += growth_rate
        
        # 限制最大值为 1.0
        utility_map[0] = np.clip(utility_map[0], 0.0, 1.0)

        # 【修复】这里只返回一次，且返回的是归一化后的奖励
        return utility_map, normalized_reward
    
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