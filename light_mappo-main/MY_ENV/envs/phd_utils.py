import numpy as np
import torch

class PHDFeatureExtractor:
    def __init__(self, config, device='cpu'):
        """
        参数 config 字典需包含:
        - cnn_h, cnn_w: CNN输入张量的高度和宽度 (例如 64, 64)
        - r_max: 观测半径 (例如 800)
        - max_speed: 用于归一化的最大速度 (例如 20 m/s)
        """
        self.H = config.get('cnn_h', 64)   # 对应距离分辨率
        self.W = config.get('cnn_w', 64)   # 对应角度分辨率
        self.r_max = config.get('r_max', 800)
        self.max_speed = config.get('max_speed', 20.0)
        self.device = device

        # --- 1. 预计算极坐标网格 (Polar Grid) ---
        # 距离 r: [0, r_max]
        dr = self.r_max / self.H
        r_vals = np.linspace(dr/2, self.r_max - dr/2, self.H)
        
        # 角度 theta: [-pi, pi] (360度全圆)
        dtheta = (2 * np.pi) / self.W
        theta_vals = np.linspace(-np.pi + dtheta/2, np.pi - dtheta/2, self.W)
        
        # 生成网格矩阵 (H, W)
        # self.R_grid[i, j] 是第 (i,j) 个格子的距离
        # self.Theta_grid[i, j] 是第 (i,j) 个格子的局部角度
        self.R_grid, self.Theta_grid = np.meshgrid(r_vals, theta_vals, indexing='ij')
        
        # 预计算单元面积 Jacobian (Area = r * dr * dtheta)
        # 这一点非常重要：远处的格子物理面积大，必须乘上 r
        self.Cell_Area = self.R_grid * dr * dtheta
        
    def process(self, phd_output, agent_pose):
        """
        将 PHD 输出转换为 MAPPO 观测张量
        
        输入:
        - phd_output: PHD.predict_update() 的返回值 [W, M, P, len]
        - agent_pose: [x, y, heading_degree] 
                      智能体的绝对位置和朝向(角度制)，如果智能体不旋转，heading传0即可
        
        返回:
        - tensor: shape (3, H, W) -> [Occupancy, Radial_Vel, Tangential_Vel]
        """
        weights = phd_output[0]
        means = phd_output[1]
        covs = phd_output[2]
        n_components = phd_output[3]
        
        ax, ay, a_head_deg = agent_pose
        a_head_rad = np.radians(a_head_deg)

        # 初始化特征图 (H, W, 3)
        feature_map = np.zeros((self.H, self.W, 3), dtype=np.float32)
        
        # 用于计算平均速度的分母 (总密度)
        total_intensity_div = np.ones((self.H, self.W), dtype=np.float32) * 1e-9

        # --- 2. 准备网格的全局坐标 ---
        # 全局角度 = 局部网格角度 + 智能体朝向
        Theta_global = self.Theta_grid + a_head_rad
        
        # 网格中心在世界坐标系下的 (x, y)
        Grid_X = ax + self.R_grid * np.cos(Theta_global)
        Grid_Y = ay + self.R_grid * np.sin(Theta_global)
        
        # 堆叠为 (H, W, 2) 方便后续向量化计算
        Grid_Pos = np.dstack((Grid_X, Grid_Y)) 

        # --- 3. 遍历 PHD 高斯分量 ---
        if n_components > 0:
            for w, m, P in zip(weights, means, covs):
                # m: [x, vx, y, vy] -> 提取位置 [x, y] 和 速度 [vx, vy]
                m_pos = np.array([m[0], m[2]])
                m_vel = np.array([m[1], m[3]])
                
                # P: 4x4 -> 提取位置部分的协方差 2x2
                P_pos = np.array([[P[0,0], P[0,2]], 
                                  [P[2,0], P[2,2]]])
                
                # --- 优化：距离门控 (Distance Gating) ---
                # 如果目标距离超过 观测半径 + 3倍标准差，直接跳过不画
                dist_to_agent = np.linalg.norm(m_pos - np.array([ax, ay]))
                sigma_max = np.sqrt(max(P[0,0], P[2,2]))
                if dist_to_agent > self.r_max + 3 * sigma_max:
                    continue

                # --- 计算高斯 PDF ---
                diff = Grid_Pos - m_pos # shape (H, W, 2)
                
                try:
                    inv_P = np.linalg.inv(P_pos)
                    det_P = np.linalg.det(P_pos)
                except np.linalg.LinAlgError:
                    continue 

                # 计算马氏距离 (Vectorized Mahalanobis Distance)
                # Einsum 公式: (x-u)^T * S^-1 * (x-u)
                mahalanobis = np.einsum('ijk,kl,ijl->ij', diff, inv_P, diff)
                
                # 概率密度 PDF
                norm_const = 1.0 / (2 * np.pi * np.sqrt(det_P))
                pdf_val = norm_const * np.exp(-0.5 * mahalanobis)
                
                # --- 核心：计算期望目标数 (Occupancy Mass) ---
                # Mass = Weight * PDF * Area (Jacobian)
                local_mass = w * pdf_val * self.Cell_Area
                
                # --- 累加到特征图 ---
                # Channel 0: Occupancy
                feature_map[:, :, 0] += local_mass
                
                # 累加动量 (用于后续求平均速度)
                # Global Velocity Momentum
                feature_map[:, :, 1] += local_mass * m_vel[0] # Vx component
                feature_map[:, :, 2] += local_mass * m_vel[1] # Vy component
                
                total_intensity_div += local_mass

            # --- 4. 速度场投影 (Global -> Local Polar) ---
            # 先求平均全局速度
            avg_vx_global = feature_map[:, :, 1] / total_intensity_div
            avg_vy_global = feature_map[:, :, 2] / total_intensity_div
            
            # 投影到 径向 (Radial) 和 切向 (Tangential)
            # Vr = Vx cos(theta) + Vy sin(theta)
            # Vt = -Vx sin(theta) + Vy cos(theta)
            # 注意：这里的 Theta_global 是网格点的角度
            
            Vr = avg_vx_global * np.cos(Theta_global) + avg_vy_global * np.sin(Theta_global)
            Vt = -avg_vx_global * np.sin(Theta_global) + avg_vy_global * np.cos(Theta_global)
            
            feature_map[:, :, 1] = Vr
            feature_map[:, :, 2] = Vt
        
        # --- 5. 数值归一化 (Normalization) ---
        # Ch0: Log 变换 (压缩数值范围)
        feature_map[:, :, 0] = np.log1p(feature_map[:, :, 0])
        
        # Ch1 & Ch2: 速度归一化 [-1, 1]
        feature_map[:, :, 1] = np.clip(feature_map[:, :, 1] / self.max_speed, -1.0, 1.0)
        feature_map[:, :, 2] = np.clip(feature_map[:, :, 2] / self.max_speed, -1.0, 1.0)

        # --- 6. 转为 PyTorch Tensor ---
        # Numpy (H, W, C) -> Torch (C, H, W)
        tensor = torch.from_numpy(feature_map).float().permute(2, 0, 1)
        
        return tensor.to(self.device)