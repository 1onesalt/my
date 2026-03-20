import numpy as np
from scipy.optimize import linear_sum_assignment

class AGMFusionCenter:
    """
    带视域感知（FoV-Aware）的中心化 AGM 融合算法
    核心功能：在融合时利用视域信息防止“数据乱伦”和错误稀释
    """
    def __init__(self, model_params):
        self.model_params = model_params
        self.T_merge = float(model_params.get("fusion_t_merge", 40.0))   # 合并/匹配阈值 (马氏距离)
        self.Jmax = 100       # 最大目标数限制
        self.unmatched_part2_scale = float(model_params.get("fusion_unmatched_scale", 0.5))
        
    def run(self, list_of_local_phds, sensor_configs):
        """按 Matlab 版序贯 AGM 逻辑进行集中融合。"""
        num_sensors = len(list_of_local_phds)
        if num_sensors == 0:
            return [[], [], [], 0]

        # 1) 选择第一个非空局部后验作为序贯融合基准
        base_idx = None
        for i in range(num_sensors):
            if list_of_local_phds[i][3] != 0:
                base_idx = i
                break
        if base_idx is None:
            return [[], [], [], 0]

        W_ref = [float(w) for w in list_of_local_phds[base_idx][0]]
        M_ref = [np.array(m, dtype=float) for m in list_of_local_phds[base_idx][1]]
        P_ref = [np.array(P, dtype=float) for P in list_of_local_phds[base_idx][2]]
        t_f = 1

        # 2) 依次融合其他有分量的传感器
        for i in range(base_idx + 1, num_sensors):
            w2, m2, p2, n2 = list_of_local_phds[i]
            if n2 == 0:
                continue

            W2 = [float(w) for w in w2]
            M2 = [np.array(m, dtype=float) for m in m2]
            P2 = [np.array(P, dtype=float) for P in p2]
            t_f += 1
            pi_1 = 1.0 - 1.0 / t_f
            pi_2 = 1.0 / t_f

            match_map, mat_match = self._match_components(M_ref, P_ref, M2)

            W_new, M_new, P_new = [], [], []

            # (a) 匹配组 AGM 融合
            for j, k in enumerate(match_map):
                if k < 0:
                    continue
                P1_inv = self._safe_inv(P_ref[j])
                P2_inv = self._safe_inv(P2[k])
                info = pi_1 * P1_inv + pi_2 * P2_inv
                P_f = self._safe_inv(info)
                m_f = P_f @ (pi_1 * P1_inv @ M_ref[j] + pi_2 * P2_inv @ M2[k])
                P_f = 0.5 * (P_f + P_f.T)

                W_new.append(pi_1 * W_ref[j] + pi_2 * W2[k])
                M_new.append(m_f)
                P_new.append(P_f)

            # (b) part1 未匹配：FoV 内衰减、FoV 外保持
            sensor_pos = np.array([sensor_configs[i]['x'], sensor_configs[i]['y']], dtype=float)
            sensor_r = float(sensor_configs[i]['range'])
            for j in range(len(W_ref)):
                if np.any(mat_match[j, :] == 1):
                    continue
                target_pos = np.array([M_ref[j][0], M_ref[j][2]])
                in_fov = np.linalg.norm(target_pos - sensor_pos) <= sensor_r
                w_tmp = pi_1 * W_ref[j] if in_fov else W_ref[j]
                W_new.append(w_tmp)
                M_new.append(M_ref[j])
                P_new.append(P_ref[j])

            # (c) part2 未匹配：直接引入（新发现）
            for k in range(len(W2)):
                if np.any(mat_match[:, k] == 1):
                    continue
                # 对未匹配新分量保守并入，抑制多传感器重复计数导致的基数偏高。
                W_new.append(self.unmatched_part2_scale * W2[k])
                M_new.append(M2[k])
                P_new.append(P2[k])

            W_ref, M_ref, P_ref = W_new, M_new, P_new

        # 3) 最多保留 Jmax 个高权重分量
        if len(W_ref) > self.Jmax:
            keep = np.argsort(W_ref)[::-1][:self.Jmax]
            W_ref = [W_ref[idx] for idx in keep]
            M_ref = [M_ref[idx] for idx in keep]
            P_ref = [P_ref[idx] for idx in keep]

        return [W_ref, M_ref, P_ref, len(W_ref)]

    def _safe_inv(self, M):
        try:
            return np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)

    def _match_components(self, M1, P1, M2):
        """
        Matlab `ALG3_Match` 的 Python 实现：
        - 构造距离矩阵
        - 阈值截断
        - 匈牙利匹配
        - 超阈值配对置零
        """
        n1 = len(M1)
        n2 = len(M2)
        if n1 == 0 or n2 == 0:
            return np.full(n1, -1, dtype=int), np.zeros((n1, n2), dtype=int)

        cost = np.full((n1, n2), self.T_merge, dtype=float)
        for i in range(n1):
            invP = self._safe_inv(P1[i])
            for j in range(n2):
                diff = M2[j] - M1[i]
                d = float(diff.T @ invP @ diff)
                cost[i, j] = min(self.T_merge, d)

        row_ind, col_ind = linear_sum_assignment(cost)
        match_map = np.full(n1, -1, dtype=int)
        mat_match = np.zeros((n1, n2), dtype=int)

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.T_merge:
                match_map[r] = c
                mat_match[r, c] = 1

        return match_map, mat_match
    
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