import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import block_diag

class AGMFusionCenter:
    """
    带视域感知（FoV-Aware）的中心化 AGM 融合算法
    核心功能：在融合时利用视域信息防止“数据乱伦”和错误稀释
    """
    def __init__(self, model_params):
        self.model_params = model_params
        self.T_merge = 40.0   # 合并/匹配阈值 (马氏距离)
        self.Jmax = 100       # 最大目标数限制
        
    def run(self, list_of_local_phds, sensor_configs):
        """
        执行融合
        :param list_of_local_phds: 各智能体的 PHD 状态 [[w,m,P,n], ...]
        :param sensor_configs: 传感器配置列表 [{'x':, 'y':, 'range':, 'fov_angle':}, ...]
        :return: 全局 PHD 状态 [W, M, P, N]
        """
        num_sensors = len(list_of_local_phds)
        if num_sensors == 0:
            return [[], [], [], 0]

        # --- 1. 建立粒子簇 (Clustering) ---
        # 将所有传感器上传的粒子按空间位置归类
        clusters = self._cluster_particles(list_of_local_phds, num_sensors)
        
        if not clusters:
            return [[], [], [], 0]

        # --- 2. 簇内视域感知 AGM 融合 ---
        W_fused, M_fused, P_fused = [], [], []

        for cluster in clusters:
            components = cluster['components'] # list of (w, m, P, sensor_id)
            
            # A. 几何融合 (Geometric): 融合位置与协方差
            # 利用 GCI (Generalized Covariance Intersection) 提升精度
            # 公式: P_f = inv( sum(inv(P_i)) ), m_f = P_f * sum(inv(P_i)*m_i)
            
            inf_mat_sum = np.zeros_like(components[0][2]) # 信息矩阵和
            weighted_mean_sum = np.zeros_like(components[0][1])
            
            for (_, m, P, _) in components:
                try:
                    inv_P = np.linalg.inv(P)
                except:
                    inv_P = np.linalg.pinv(P)
                inf_mat_sum += inv_P
                weighted_mean_sum += inv_P @ m
            
            try:
                P_f = np.linalg.inv(inf_mat_sum)
            except:
                P_f = np.linalg.pinv(inf_mat_sum)
            m_f = P_f @ weighted_mean_sum

            # B. 算术融合 (Arithmetic) + 视域校验 (FoV Check)
            # 这是防止数据乱伦的关键步骤
            
            w_accumulated = 0.0
            valid_sensor_count = 0 # 分母：只有视域覆盖了目标的传感器才有资格投票
            
            target_pos = m_f[[0, 2]] # 假设状态是 [x, vx, y, vy]
            
            for i in range(num_sensors):
                # 检查传感器 i 是否在 cluster 中贡献了粒子
                w_contrib = 0.0
                has_detection = False
                
                for (w, _, _, sid) in components:
                    if sid == i:
                        w_contrib = w
                        has_detection = True
                        break
                
                # 视域逻辑核心：
                if has_detection:
                    # Case 1: 看到了 -> 累加权重
                    w_accumulated += w_contrib
                    valid_sensor_count += 1
                else:
                    # 没看到，需要判断原因
                    sensor_pos = np.array([sensor_configs[i]['x'], sensor_configs[i]['y']])
                    dist = np.linalg.norm(target_pos - sensor_pos)
                    
                    if dist <= sensor_configs[i]['range']:
                        # Case 2: 在视域内但没看到 (漏检/确认消失) -> 累加 0，分母+1
                        # 这会显著拉低该目标的平均权重
                        w_accumulated += 0.0
                        valid_sensor_count += 1
                    else:
                        # Case 3: 不在视域内 -> 不知情
                        # 不累加权重，也不增加分母 (相当于它弃权)
                        pass 

            # 如果没有任何传感器覆盖该区域（不太可能，因为至少有一个传感器看到了），兜底处理
            if valid_sensor_count == 0:
                valid_sensor_count = 1
            
            # 计算最终权重
            w_f = w_accumulated / valid_sensor_count
            
            # 只有经过确认依然存在的粒子才保留
            if w_f > 1e-4:
                W_fused.append(w_f)
                M_fused.append(m_f)
                P_fused.append(P_f)

        # --- 3. 数量限制 ---
        if len(W_fused) > self.Jmax:
            indices = np.argsort(W_fused)[::-1][:self.Jmax]
            W_fused = [W_fused[i] for i in indices]
            M_fused = [M_fused[i] for i in indices]
            P_fused = [P_fused[i] for i in indices]
            
        return [W_fused, M_fused, P_fused, sum(W_fused)]

    def _cluster_particles(self, list_of_local_phds, num_sensors):
        """
        辅助函数：使用匈牙利算法进行序贯聚类
        """
        # 1. 找基准 (Base)
        base_idx = -1
        for i in range(num_sensors):
            if list_of_local_phds[i][0]:
                base_idx = i
                break
        if base_idx == -1: return []

        clusters = []
        w_base, m_base, P_base = list_of_local_phds[base_idx][0], list_of_local_phds[base_idx][1], list_of_local_phds[base_idx][2]
        
        # 初始化簇
        for j in range(len(w_base)):
            clusters.append({
                'components': [(w_base[j], m_base[j], P_base[j], base_idx)]
            })

        # 2. 遍历其他传感器进行匹配
        for i in range(num_sensors):
            if i == base_idx: continue
            w_new, m_new, P_new = list_of_local_phds[i][0], list_of_local_phds[i][1], list_of_local_phds[i][2]
            if not w_new: continue

            # 构建代价矩阵
            n_clusters = len(clusters)
            n_particles = len(w_new)
            cost_matrix = np.full((n_clusters, n_particles), 1e9)

            for r in range(n_clusters):
                # 简化计算：只比对簇中第一个粒子
                target_m = clusters[r]['components'][0][1] 
                target_P = clusters[r]['components'][0][2]
                try:
                    inv_P = np.linalg.inv(target_P)
                except:
                    inv_P = np.linalg.pinv(target_P)

                for c in range(n_particles):
                    diff = m_new[c] - target_m
                    dist = diff.T @ inv_P @ diff
                    if dist < self.T_merge:
                        cost_matrix[r, c] = dist

            # 匈牙利匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_cols = set()
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.T_merge:
                    clusters[r]['components'].append((w_new[c], m_new[c], P_new[c], i))
                    matched_cols.add(c)
            
            # 未匹配的粒子新建簇
            for c in range(n_particles):
                if c not in matched_cols:
                    clusters.append({
                        'components': [(w_new[c], m_new[c], P_new[c], i)]
                    })
                    
        return clusters
    
    def distribute_to_sensors(self, global_state, sensor_configs):
        """
        对应 MATLAB 中的 FoV_divide
        将全局融合结果根据各传感器的视场进行分配
        
        输入:
            global_state: [W, M, P, N] (融合后的全局状态)
            sensor_configs: 传感器配置列表
        输出:
            feedback_states: 列表，每个元素是分发给对应智能体的新局部状态
        """
        W_g, M_g, P_g, _ = global_state
        num_sensors = len(sensor_configs)
        feedback_states = []

        if not W_g:
            # 如果全局都没目标，大家也都别想看到
            empty_state = [[], [], [], 0]
            return [empty_state for _ in range(num_sensors)]

        for i in range(num_sensors):
            # 为传感器 i 准备新的状态容器
            w_local = []
            m_local = []
            p_local = []
            
            sensor_pos = np.array([sensor_configs[i]['x'], sensor_configs[i]['y']])
            sensor_r = sensor_configs[i]['range']
            
            # 遍历全局所有粒子
            for k in range(len(W_g)):
                target_pos = M_g[k][[0, 2]] # 假设状态是 [x, vx, y, vy]
                dist = np.linalg.norm(target_pos - sensor_pos)
                
                # === 视场门控 (FoV Gating) ===
                # 逻辑：只有在传感器视场内的全局目标，才会被反馈给该传感器
                # 这避免了传感器被视场外的“幽灵目标”带偏 (Anti-Incest 的一种手段)
                if dist <= sensor_r:
                    w_local.append(W_g[k])
                    m_local.append(M_g[k])
                    p_local.append(P_g[k])
            
            n_est = sum(w_local)
            feedback_states.append([w_local, m_local, p_local, n_est])
            
        return feedback_states