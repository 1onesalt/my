import logging
import random
import gym
import numpy as np
from gym import spaces
from gym.spaces import Dict, Box

from MY_ENV.utils.action_space import MultiAgentActionSpace
from MY_ENV.utils.observation_space import MultiAgentObservationSpace
from MY_ENV.envs.target_model2 import model, targets, target_CV, observe_Fov, polar2dicaer
from MY_ENV.envs.PHD import PHD, State_extraction, generate

class TargetSearchEnv(gym.Env):
    def __init__(self, x_min=-1000, x_max=1000, y_min=-1000, y_max=1000, n_agent=3, n_target=5, max_steps=100):
        # --- 1. 环境基础参数 ---
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.n_agents = n_agent
        self.n_targets = n_target
        self.max_steps = max_steps
        
        # 模型参数 (从 target_model 中获取或定义)
        self.model_params = model()
        self.sensor_r = self.model_params["obverser_d"]  # 视域半径 d
        self.agent_v = 10.0  # 智能体恒定速率 v (假设值，需根据实际情况调整)
        
        # --- 2. 极坐标网格参数 (Section 4.1) ---
        self.grid_U = 16    # 极径方向划分数量
        self.grid_V = 16    # 极角方向划分数量
        self.grid_C = 4     # 通道数: [强度, 径向速度, 切向速度, 协方差]
        
        # --- 3. 动作空间 (Section 4.2) ---
        # 动作是航向角的调整量。假设离散化为 Na 个动作
        # ai = -pi + 2*na*pi / Na
        self.n_actions = 7 # 例如:大幅左转, 小幅左转, 直行, 小幅右转, 大幅右转等，或均匀分布
        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(self.n_actions) for _ in range(self.n_agents)]
        )
        # 定义动作对应的角度调整值 (这里简化为在 [-pi/2, pi/2] 范围内调整，或全向调整)
        # 根据论文公式 12: ai = -pi + 2*na*pi / Na，这意味着动作直接决定绝对航向还是相对调整？
        # 通常航向控制是相对调整 (delta)，但公式 12 看起来像是绝对航向的选择。
        # 结合上下文 "控制智能体的运动"，这里实现为相对调整量 delta_theta 更符合平滑运动控制
        # 或者严格按照公式12实现为"设定当前时刻的航向角"。这里采用相对调整，更符合动力学。
        self.angle_adjustments = np.linspace(-np.pi/4, np.pi/4, self.n_actions) # 示例: -45度 到 45度

        # --- 4. 观测空间 (Section 4.1) ---
        # 包含: 极坐标特征张量 + 自身状态
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

        # 奖励因子 (公式 29)
        self.lambda1 = 1.0  # r_track
        self.lambda2 = 2.0  # r_new
        self.lambda3 = 0.5  # r_overlap
        self.lambda4 = 0.5  # r_bound
        self.rho_star = 0.1 # 理想重叠率
        self.delta_overlap = 1.0 # 重叠奖励权重
        
        # 历史基数记录 (用于 r_new)
        self.history_window = 5
        self.cardinality_history = {i: [] for i in range(self.n_agents)}

        # 初始化变量
        self.state = None
        self.agent_pos = None
        self.agent_headings = None
        self.agent_Z_dicaer = None
        self.trajectories = None
        self.step_count = 0
        self.total_reward = None
        self.done = None

    def reset(self):
        self.step_count = 0
        self.done = False
        self.total_reward = [0.0 for _ in range(self.n_agents)]
        
        # 初始化位置和朝向
        self.agent_pos = {}
        self.agent_headings = {}
        for i in range(self.n_agents):
            self.agent_pos[i] = np.array([
                random.uniform(self.x_min + 100, self.x_max - 100),
                random.uniform(self.y_min + 100, self.y_max - 100)
            ])
            self.agent_headings[i] = random.uniform(-np.pi, np.pi)
            self.cardinality_history[i] = [0] * self.history_window # 重置历史基数

        # 初始化 PHD 状态和观测缓存
        self.state = [
            [0, np.zeros((1, 4)), np.zeros((4, 4)), 0] 
            for _ in range(self.n_agents)
        ]
        self.agent_Z_dicaer = [[[], []] for _ in range(self.n_agents)]
        
        # 初始化目标轨迹
        self.trajectories = self._init_targets()

        # 生成初始观测
        return self._get_observations()

    def step(self, actions):
        if self.step_count < self.max_steps - 1:
            self.step_count += 1
            
        obs_n = []
        rewards = []
        dones = []
        infos = []

        # 1. 智能体运动更新 (Section 2-4)
        for i in range(self.n_agents):
            action_idx = actions[i]
            # 动作转化为航向角调整
            delta_theta = self.angle_adjustments[action_idx]
            self.agent_headings[i] += delta_theta
            # 归一化角度到 [-pi, pi]
            self.agent_headings[i] = (self.agent_headings[i] + np.pi) % (2 * np.pi) - np.pi
            
            # 位置更新 (公式 7)
            dx = self.agent_v * np.cos(self.agent_headings[i])
            dy = self.agent_v * np.sin(self.agent_headings[i])
            self.agent_pos[i][0] = np.clip(self.agent_pos[i][0] + dx, self.x_min, self.x_max)
            self.agent_pos[i][1] = np.clip(self.agent_pos[i][1] + dy, self.y_min, self.y_max)

        # 2. 获取新的观测并更新 PHD
        # 这里先统一更新所有智能体的 PHD 状态，再计算奖励，因为奖励可能依赖全局信息(如重叠)
        # 注意: 论文提及 CTDE 和 AGM 融合。为简化，这里暂保留单智能体 PHD 逻辑，
        # 但奖励计算时使用了 agent 间的相对距离计算重叠。
        
        current_phd_states = []
        for i in range(self.n_agents):
            # 获取观测
            model_data = model()
            # 修正: model 中需要动态更新 agent 位置
            model_data['x_agent'] = self.agent_pos[i][0]
            model_data['y_agent'] = self.agent_pos[i][1]
            
            z_polar = observe_Fov(model_data, self.trajectories[self.step_count])
            Z_dicaer = polar2dicaer(z_polar, model_data)
            
            # 更新观测历史
            self.agent_Z_dicaer[i][0] = self.agent_Z_dicaer[i][1]
            self.agent_Z_dicaer[i][1] = Z_dicaer
            
            # PHD 更新
            phd = PHD(model_data)
            phd.init_params(self.agent_Z_dicaer[i][0], self.agent_Z_dicaer[i][1], z_polar, self.state[i])
            # 注意: generate 函数需要正确实现新生目标逻辑
            phd.w_new, phd.m_new, phd.P_new, phd.J_new = generate(
                phd.z_lastD, phd.nums_z_lastD, phd.z_nowD, phd.nums_z_nowD, phd.Vx_thre, phd.Vy_thre
            )
            X_now = phd.predict_update()
            self.state[i] = X_now
            current_phd_states.append(X_now)

        # 3. 构建观测张量和计算奖励
        obs_n = self._get_observations()
        
        for i in range(self.n_agents):
            reward = self._compute_reward(i, current_phd_states[i])
            rewards.append(reward)
            self.total_reward[i] += reward
            dones.append(self.step_count >= self.max_steps)
            infos.append({'step': self.step_count})

        self.done = all(dones)
        return obs_n, rewards, dones, infos

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

    def _init_targets(self):
        # 封装 target_model 中的初始化逻辑
        targets_birth_time, targets_death_time, targets_start = targets(self.max_steps)
        # 注意: target_CV 的 model 参数传递
        # 这里由于 target_CV 内部使用了 model['num_scans'] 等，需确保匹配
        dummy_model = model()
        dummy_model['num_scans'] = self.max_steps
        trajectories, _ = target_CV(
            dummy_model, targets_birth_time, targets_death_time, targets_start, noise=True
        )
        return trajectories

    def _get_global_observation(self):
        """
        构建全局融合的信念状态 (Critic 专用)
        
        返回:
        1. global_grid (16, 16, 4): 覆盖全域的 PHD 强度/速度/协方差图
        2. joint_state (N * 3): 所有智能体的 [x, y, theta]
        """
        # --- 1. 初始化全局网格 ---
        # 注意：这里我们复用 grid_U, grid_V (16x16)，意味着全局图的分辨率较低，但视野覆盖全图
        # 如果显存允许，也可以定义单独的 global_grid_size (如 32x32)
        g_u, g_v, g_c = self.grid_U, self.grid_V, self.grid_C
        global_grid = np.zeros((g_u, g_v, g_c), dtype=np.float32)
        
        # 计算全局网格的物理步长 (覆盖 x_min ~ x_max)
        step_x = (self.x_max - self.x_min) / g_u
        step_y = (self.y_max - self.y_min) / g_v
        
        # --- 2. 融合所有智能体的 PHD 状态 ---
        # 遍历每个智能体 i
        for i in range(self.n_agents):
            weights, means, covs, _ = self.state[i]
            
            # 遍历该智能体维护的所有高斯分量
            for w, m, P in zip(weights, means, covs):
                if w < 1e-4: continue # 忽略极小权重
                
                # m = [x, vx, y, vy] (全局坐标)
                gx, gy = m[0], m[2]
                
                # 判断是否在监测区域内
                if not (self.x_min <= gx <= self.x_max and self.y_min <= gy <= self.y_max):
                    continue

                # 计算在全局网格中的索引
                idx_x = int((gx - self.x_min) / step_x)
                idx_y = int((gy - self.y_min) / step_y)
                
                # 边界保护
                idx_x = np.clip(idx_x, 0, g_u - 1)
                idx_y = np.clip(idx_y, 0, g_v - 1)
                
                # --- 通道叠加 ---
                # Channel 0: 强度 (Intensity) - 直接累加权重，表示该区域目标存在的可能性
                global_grid[idx_x, idx_y, 0] += w
                
                # Channel 1 & 2: 速度 (Velocity) - 累加动量 (w * v)
                # 注意：Critic 看到的是全局绝对速度，不需要像局部观测那样旋转坐标系
                global_grid[idx_x, idx_y, 1] += w * m[1] # vx
                global_grid[idx_x, idx_y, 2] += w * m[3] # vy
                
                # Channel 3: 不确定性 (Covariance Trace) - 累加加权迹
                global_grid[idx_x, idx_y, 3] += w * (P[0,0] + P[2,2])

        # --- 3. 归一化处理 ---
        # 对速度和协方差进行归一化 (除以总强度)
        # 防止网格内粒子叠加导致数值过大，同时也让特征具有物理意义(平均速度、平均不确定性)
        epsilon = 1e-6
        denominator = global_grid[:, :, 0:1] + epsilon # 保持维度以便广播
        
        global_grid[:, :, 1] /= denominator[:, :, 0] # vx_avg
        global_grid[:, :, 2] /= denominator[:, :, 0] # vy_avg
        global_grid[:, :, 3] /= denominator[:, :, 0] # cov_avg

        # --- 4. 构建联合状态向量 ---
        joint_state = []
        for i in range(self.n_agents):
            # 将每个智能体的 [x, y, theta] 加入列表
            joint_state.extend([
                self.agent_pos[i][0], 
                self.agent_pos[i][1], 
                self.agent_headings[i]
            ])
            
        return global_grid, np.array(joint_state, dtype=np.float32)
