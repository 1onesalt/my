import logging
import random
import gym
import numpy as np
from gym import spaces
from gym.spaces import Dict, Box
import copy
from MY_ENV.utils.action_space import MultiAgentActionSpace
from MY_ENV.utils.observation_space import MultiAgentObservationSpace
from MY_ENV.envs.target_model import model, targets, target_CV, observe_Fov, polar2dicaer
from MY_ENV.envs.PHD import PHD, State_extraction, generate
from MY_ENV.envs.phd_fusion import AGMFusionCenter

class SensorAgent:
    """
    封装单个传感器的：位置、PHD滤波器、历史状态
    """
    def __init__(self, agent_id, x, y, heading, track_params):
        self.id = agent_id
        
        # 1. 独立的模型参数 (深拷贝)
        self.model_data = copy.deepcopy(track_params)
        self.model_data['x_agent'] = x
        self.model_data['y_agent'] = y
        self.model_data['heading_agent'] = heading
        
        # 2. 独立的 PHD 滤波器实例
        self.phd = PHD(self.model_data)
        
        # 3. 状态初始化 [weights, means, covs, n_components]
        self.state = [[], [], [], 0]
        
        # 4. 观测历史 (用于两帧差分新生检测)
        self.z_cart_history = [] 

    def update_position(self, x, y, heading):
        """RL每一面更新位置后，同步更新模型参数"""
        self.model_data['x_agent'] = x
        self.model_data['y_agent'] = y
        self.model_data['heading_agent'] = heading
        self.heading = heading

    def process_measurement(self, z_polar_noisy):
        """
        处理一帧量测：更新历史 -> PHD初始化 -> 新生检测 -> 预测更新
        Input: z_polar_noisy (当前时刻的极坐标量测，由环境生成)
        Output: X_local (局部估计)
        """
        z_polar_global = []
        
        # 1. 坐标系对齐 (相对 -> 全局)
        for z in z_polar_noisy:
            r = z[0]
            theta_rel = z[1]
            
            # [关键] 加上自身的朝向 self.heading
            theta_global = theta_rel + self.heading
            # 归一化到 [-pi, pi]
            theta_global = (theta_global + np.pi) % (2 * np.pi) - np.pi
            z_polar_global.append(np.array([r, theta_global]))

        # 1. 极坐标转直角坐标 (用于新生检测)
        if len(z_polar_global) > 0:
            z_cart = polar2dicaer(z_polar_global, self.model_data)
        else:
            z_cart = []
            
        # 2. 维护历史 (滑动窗口)
        self.z_cart_history.append(z_cart)
        if len(self.z_cart_history) > 3: # 保持队列短一点，节省内存
            self.z_cart_history.pop(0)
            
        # 3. 如果历史数据不足，直接返回空或上一时刻状态
        if len(self.z_cart_history) < 2:
            return self.phd.predict_update() # 或者 return self.state

        # 4. 准备 PHD 输入数据
        z_lastD = self.z_cart_history[-2] # 上一帧直角
        z_nowD = self.z_cart_history[-1]  # 当前帧直角
        
        # 5. PHD 流程
        # 初始化 (使用上一时刻融合反馈回来的 self.state)
        self.phd.init_params(z_lastD, z_nowD, z_polar_noisy, self.state)
        
        # 新生检测
        w_new, m_new, P_new, J_new = generate(
            self.phd.z_lastD, len(self.phd.z_lastD),
            self.phd.z_nowD, len(self.phd.z_nowD),
            self.model_data['Vx_thre'], self.model_data['Vy_thre']
        )
        self.phd.w_new, self.phd.m_new, self.phd.P_new, self.phd.J_new = w_new, m_new, P_new, J_new
        
        # 预测更新
        X_local = self.phd.predict_update()
        return X_local


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

        self.base_model_params = model()
        self.sensor_r = self.base_model_params["obverser_d"]  # 视域半径 d
        self.agent_v = 10.0  # 智能体恒定速率 v (假设值，需根据实际情况调整)

        # 初始化融合中心
        self.fusion_module = AGMFusionCenter(self.base_model_params)
        self.tracking_agents = []
        
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

        local_obs_dim = self.grid_U * self.grid_V * self.grid_C + 3
        share_obs_dim = local_obs_dim * self.n_agents

        self.share_observation_space = MultiAgentObservationSpace([
            Box(
                low=-np.inf, high=np.inf,
                shape=(share_obs_dim,),
                dtype=np.float32
            ) for _ in range(self.n_agents)
        ])

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
        
        
        self.targets = self._init_targets()                        # 初始化目标轨迹
        self.sensor_states = self._init_sensors()                  # 初始化传感器位置和朝向
        self.sensor_pos = self.sensor_states[:, 0:2]

        self.tracking_agents = []   #agent实例列表
        
        for i in range(self.n_agents):
            init_x, init_y, init_heading = self.sensor_states[i]
            agent = SensorAgent(i, init_x, init_y, init_heading, self.base_model_params)  #封装单个传感器的：位置、PHD滤波器、历史状态
            self.tracking_agents.append(agent)
            
        self.fusion_center = AGMFusionCenter(self.base_model_params)

        # 生成初始观测
        return self._get_observations()

    def step(self, actions):
        self.step_count += 1
        obs_n = []
        rewards = []
        dones = []
        infos = []

        # 1. 智能体运动更新 (Section 2-4)
        for i in range(self.n_agents):
            # --- 动作解析 ---
            action_in = actions[i]    
        if hasattr(action_in, 'shape') and len(action_in.shape) > 0 and action_in.size > 1:
            # 如果是 One-hot 向量 (例如 [0, 0, 1, 0...])
            action_idx = np.argmax(action_in)
        elif hasattr(action_in, 'item'):
            action_idx = int(action_in.item())
        else:
            action_idx = int(action_in)

            # --- 运动更新 ---
            # 更新朝向
            delta_theta = self.angle_adjustments[action_idx]
            self.agent_headings[i] += delta_theta
            self.agent_headings[i] = (self.agent_headings[i] + np.pi) % (2 * np.pi) - np.pi
            
            # 位置更新 
            dx = self.agent_v * np.cos(self.agent_headings[i])
            dy = self.agent_v * np.sin(self.agent_headings[i])
            self.agent_pos[i][0] = np.clip(self.agent_pos[i][0] + dx, self.x_min, self.x_max)
            self.agent_pos[i][1] = np.clip(self.agent_pos[i][1] + dy, self.y_min, self.y_max)

        # 2. 获取新的观测并更新 PHD
        current_phd_states = []
        sensor_configs = []

        for i in range(self.n_agents):
            sensor_configs.append({
                'x': self.agent_pos[i][0],
                'y': self.agent_pos[i][1],
                'range': self.sensor_r,
            })

            # 获取观测
            model_data = self.base_model_params.copy()
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
            current_phd_states.append(X_now)
        
        self.global_phd_state = self.fusion_module.run(current_phd_states, sensor_configs)

        feedback_states = self.fusion_module.distribute_to_sensors(self.global_phd_state, sensor_configs)

        for i in range(self.n_agents):
            self.state[i] = feedback_states[i]

        # 3. 构建观测张量和计算奖励
        obs_n, share_obs_n = self._get_observations_dual()
        episode_done = (self.step_count >= self.max_steps)

        for i in range(self.n_agents):
            reward_val = self._compute_reward(i, self.state[i])
            rewards.append([reward_val])
            self.total_reward[i] += reward_val
            dones.append(episode_done)
            infos.append({
                'step': self.step_count,
                'share_obs': share_obs_n[i]
            })

        self.done = all(dones)
        return obs_n, rewards, dones, infos

    def _get_observations_dual(self):
        obs_n = []
        share_obs_n = []
        for i in range(self.n_agents):
            # Actor: 看局部反馈结果
            local_grid = self._build_polar_grid(self.state[i], self.agent_pos[i], self.agent_headings[i])
            
            # Critic: 看全局融合结果 (投影到局部坐标系)
            global_grid = self._build_polar_grid(self.global_phd_state, self.agent_pos[i], self.agent_headings[i])
            
            obs = {
                'polar_grid': local_grid,
                'self_state': np.array([self.agent_pos[i][0], self.agent_pos[i][1], self.agent_headings[i]], dtype=np.float32)
            }
            obs_n.append(obs)
            
            # Flatten Global Grid for Critic
            grid_flat = global_grid.flatten()
            state_vec = obs['self_state']
            share_obs_vec = np.concatenate([grid_flat, state_vec])
            share_obs_n.append(share_obs_vec)
            
        return obs_n, share_obs_n

    def _get_observations(self):
        obs, _ = self._get_observations_dual()
        return obs

    def _build_polar_grid(self, phd_state, agent_pos, agent_heading):
        """
        根据公式 (16)-(21) 构建 U x V x C 的特征张量
        phd_state: [weights, means, covs, num]
        """
        weights, means, covs, _ = phd_state
        
        grid = np.zeros((self.grid_U, self.grid_V, self.grid_C), dtype=np.float32)
        
        # 如果没有目标分量，直接返回零张量
        if not weights or len(weights) == 0:
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
            if agent_i == j: continue
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
    
    def _init_sensors(self):
        """
        初始化传感器状态
        Returns:
            np.array, shape=(n_agents, 3) -> [[x, y, heading], ...]
            - x, y: 坐标 (米)
            - heading: 朝向 (弧度, [-pi, pi])
        """
        # 模式 1: 训练时使用随机初始化 (Training Mode)
        # 假设地图范围是 [-2000, 2000]，你可以根据 args.map_size 修改
        limit = 2000.0 
        
        # 1. 生成随机位置 (x, y)
        # 这里全图随机，你也可以限制在某个“起飞区”（例如 x < -1000）
        pos_x = np.random.uniform(-limit, limit, self.n_agents)
        pos_y = np.random.uniform(-limit, limit, self.n_agents)
        
        # 2. 生成随机朝向 (heading) -> [-pi, pi]
        heading = np.random.uniform(-np.pi, np.pi, self.n_agents)
        
        # 3. 堆叠成 (N, 3) 矩阵
        sensor_states = np.stack([pos_x, pos_y, heading], axis=1)
        
        # ---------------------------------------------------------
        # 模式 2: 测试/验证时的固定初始化 (Validation Mode)
        # 如果你想复现 validate_tracking.py 的场景，请取消下面注释
        # ---------------------------------------------------------
        # if self.args.use_fixed_spawn: # 假设你在参数里加了这个开关
        #     # 这是一个固定的 10 机编队 (参考之前的代码)
        #     fixed_pos = np.array([
        #         [0, 0], [-500, 0], [500, 800], [1000, 800], [1000, 0],
        #         [500, -800], [-500, -800], [-1500, 0], [-1000, 800], [-1000, 1600]
        #     ])
        #     
        #     # 如果智能体数量不对，进行裁剪或填充
        #     current_n = self.n_agents
        #     if len(fixed_pos) >= current_n:
        #         use_pos = fixed_pos[:current_n]
        #     else:
        #         # 不够的话就随机补
        #         pad = np.random.uniform(-500, 500, (current_n - len(fixed_pos), 2))
        #         use_pos = np.vstack([fixed_pos, pad])
        #     
        #     # 固定朝向 (例如全部朝北 90度/1.57rad，或者随机)
        #     # fixed_heading = np.full((current_n, 1), np.pi/2) 
        #     fixed_heading = np.random.uniform(-np.pi, np.pi, (current_n, 1))
        #
        #     sensor_states = np.hstack([use_pos, fixed_heading])
        
        return sensor_states

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
