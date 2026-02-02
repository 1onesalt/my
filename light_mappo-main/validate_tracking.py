import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ==========================================
# 1. 导入已有模块
# ==========================================
from MY_ENV.envs.PHD import PHD, generate, State_extraction
from MY_ENV.envs.phd_fusion import AGMFusionCenter
# 导入工具函数
from MY_ENV.envs.test_model import model, polar2dicaer, observe_Fov
from MY_ENV.envs.target_model import target_CV

# ==========================================
# 2. 定义 SensorAgent 类 (解决混淆的核心)
# ==========================================
class SensorAgent:
    """
    封装单个传感器的所有属性：位置、模型参数、PHD滤波器、历史状态
    """
    def __init__(self, sensor_id, x, y, base_params):
        self.id = sensor_id
        
        # 1. 独立的模型参数 (深拷贝，防止共用修改)
        self.model_data = copy.deepcopy(base_params)
        self.model_data['x_agent'] = x
        self.model_data['y_agent'] = y
        
        # 2. 独立的 PHD 滤波器实例
        self.phd = PHD(self.model_data)
        
        # 3. 状态初始化 (权重, 均值, 协方差, 粒子数)
        # 对应您参考代码中的 state1 = [0, zeros, zeros, 0]
        self.state = [
            [],             # weights
            [],             # means
            [],             # covs
            0               # Jk (number of components)
        ]
        
        # 4. 观测历史 (用于两帧差分)
        # 对应 reference code 中 Z_dicaer1[i-2], Z_dicaer1[i-1]
        self.z_cart_history = [] 

    def step_filter(self, current_step, trajectories):
        """
        单步滤波逻辑：观测 -> 初始化参数 -> 新生检测 -> 预测更新
        """
        # --- A. 获取观测 (对应 observe_Fov) ---
        # 这里的 trajectories 是所有时刻的，我们取当前时刻 i 的真值进行观测生成
        # 注意：observe_Fov 内部已经加了杂波，但没加高斯噪声，这里为了严谨复现，
        # 我们手动添加符合 model_data['obverser_R'] 的噪声
        
        # 1. 获取理想观测 (r, theta)
        # target_CV 生成的 trajectories 是 list[list[state]]
        # state 是 [x, vx, y, vy], observe_Fov 需要 [x, y]
        current_targets = trajectories[current_step]
        
        z_polar_ideal = observe_Fov(self.model_data, current_targets)
        
        # 2. 添加噪声 (匹配您的 model定义)
        z_polar_noisy = []
        R = self.model_data['obverser_R']
        std_r = np.sqrt(R[0,0])
        std_theta = np.sqrt(R[1,1])
        
        for z in z_polar_ideal:
            r = z[0] + np.random.randn() * std_r
            theta = z[1] + np.random.randn() * std_theta
            z_polar_noisy.append(np.array([r, theta]))
            
        # 3. 转直角坐标 (对应 polar2dicaer)
        if len(z_polar_noisy) > 0:
            # [修复] 去掉外层的 [] 包装和末尾的 [0] 索引
            # 直接传入观测列表，适配您当前的 test_model.py
            z_cart = polar2dicaer(z_polar_noisy, self.model_data)
        else:
            z_cart = []
            
        # --- B. 更新历史数据 ---
        self.z_cart_history.append(z_cart)
        
        # 需要至少3帧数据才能开始(i-2)，或者我们做边界处理
        # 您的参考代码从 i=2 开始 (即第3帧)
        if len(self.z_cart_history) < 3:
            # 数据不够，不做更新，保持空状态或上一时刻状态
            return self.state 

        # 获取 t-2, t-1, t 时刻的数据
        z_lastD = self.z_cart_history[-2] # t-1 (上一次)
        z_nowD = self.z_cart_history[-1]  # t   (这一次)
        z_polar_now = z_polar_noisy       # t   (这一次极坐标)
        
        # --- C. PHD 核心步骤 (完全对应您的参考代码) ---
        
        # 1. 初始化参数
        # 注意：self.state 是上一时刻(t-1)经过反馈后的状态
        # 参数: (Z_dicaer[i-1], Z_dicaer[i], z_polar[i], state_prev)
        # 修正：generate 需要的是 z_lastD(i-2) 和 z_nowD(i-1) 还是 i-1和i？
        # 根据您的参考代码: 
        # init_params(Z[i-2], Z[i-1], z_polar[i], state) -> 这里似乎有一点时间戳的混淆
        # 通常两帧差分是 Z[k-1] 和 Z[k]。
        # 让我们按照标准的 PHD 逻辑：
        # z_lastD: 上一帧直角坐标观测
        # z_nowD: 当前帧直角坐标观测
        
        self.phd.init_params(z_lastD, z_nowD, z_polar_now, self.state)
        
        # 2. 新生目标检测
        self.phd.w_new, self.phd.m_new, self.phd.P_new, self.phd.J_new = generate(
            self.phd.z_lastD, len(self.phd.z_lastD),
            self.phd.z_nowD, len(self.phd.z_nowD),
            self.model_data['Vx_thre'], self.model_data['Vy_thre']
        )
        
        # 3. 预测与更新
        X_now = self.phd.predict_update()
        
        # 此时 X_now 是局部估计结果，暂时不赋值给 self.state
        # 因为我们要拿去融合，融合后的结果再赋值回来
        return X_now

# ==========================================
# 3. 辅助函数
# ==========================================
def get_fixed_targets():
    """生成表3.6的固定轨迹参数"""
    # 必须使用 float64 防止 numpy 类型报错
    s1 = np.array([400, -10, 100, -10], dtype=np.float64)
    s2 = np.array([-1500, 0, -500, 10], dtype=np.float64)
    s3 = np.array([-1000, 15, 1700, 0], dtype=np.float64)
    s4 = np.array([-1000, 25, -2000, 8], dtype=np.float64)
    s5 = np.array([-800, -10, 400, 10], dtype=np.float64)
    s6 = np.array([-800, 0, 300, -10], dtype=np.float64)
    s7 = np.array([600, -15, 1000, -10], dtype=np.float64)
    s8 = np.array([-300, 10, -500, -10], dtype=np.float64)
    s9 = np.array([1000, 5, 2000, -15], dtype=np.float64)
    s10 = np.array([-1400, -10, 1500, -20], dtype=np.float64)
    s11 = np.array([-500, 15, 250, 0], dtype=np.float64)

    targets_start = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
    targets_birth = [20, 1, 1, 50, 20, 1, 20, 20, 1, 20, 1]
    targets_death = [100, 80, 100, 100, 60, 100, 100, 80, 80, 80, 100]
    return targets_birth, targets_death, targets_start

def calculate_ospa(X_est_list, X_true_list, p=2, c=200):
    """OSPA计算工具"""
    # X_est_list: [ [x,y,vx,vy], ... ] -> 我们只取 [x,y]
    if len(X_est_list) > 0:
        est_pos = [np.array([x[0], x[2]]) for x in X_est_list] # 提取位置
    else:
        est_pos = []
    
    true_pos = [np.array([x[0], x[2]]) for x in X_true_list] # 提取位置
    
    m = len(est_pos)
    n = len(true_pos)
    
    if m == 0 and n == 0: return 0
    if m > n: return calculate_ospa(X_true_list, X_est_list, p, c) # 交换
    
    cost_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d = np.linalg.norm(est_pos[i] - true_pos[j])
            cost_matrix[i, j] = min(d, c) ** p
            
    if m > 0:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_cost = cost_matrix[row_ind, col_ind].sum()
    else:
        matched_cost = 0
        
    term1 = matched_cost
    term2 = (c ** p) * (n - m)
    return ((term1 + term2) / n) ** (1/p)

# ==========================================
# 4. 主程序 (Main Loop)
# ==========================================
def main():
    num_steps = 100
    mc_runs = 20
    
    # 1. 准备基础参数
    base_model_params = model()
    # 强制覆盖为测试场景参数
    base_model_params["obverser_d"] = 1500.0
    base_model_params["obverser_R"] = np.diag([10.0, 0.005])
    base_model_params["Vx_thre"] = 50.0
    base_model_params["Vy_thre"] = 50.0
    base_model_params["Pd"] = 0.98
    base_model_params["Zr"] = 2
    
    # 2. 生成固定的真实轨迹 (Ground Truth)
    birth, death, start = get_fixed_targets()
    # trajectories[t] 包含了 t 时刻所有存活目标的 [x, vx, y, vy]
    trajectories, _ = target_CV(base_model_params, birth, death, start, num_steps)
    
    # 3. 定义传感器位置 (图a)
    sensor_positions = [
        [0, 0], [-500, 0], [500, 800], [1000, 800], [1000, 0],
        [500, -800], [-500, -800], [-1500, 0], [-1000, 800], [-1000, 1600]
    ]
    
    # 结果统计容器
    ospa_mc = np.zeros((mc_runs, num_steps))
    card_mc = np.zeros((mc_runs, num_steps))
    
    print(f"Start Simulation: {mc_runs} runs, {num_steps} steps.")

    # --- 蒙特卡洛循环 ---
    for mc in tqdm(range(mc_runs)):
        
        # A. 初始化所有传感器智能体
        agents = []
        sensor_configs = [] # 用于融合中心的视域判断
        for idx, pos in enumerate(sensor_positions):
            # 创建智能体：自动处理模型参数拷贝、PHD初始化
            agent = SensorAgent(idx, pos[0], pos[1], base_model_params)
            agents.append(agent)
            
            sensor_configs.append({
                'x': pos[0], 'y': pos[1], 'range': base_model_params["obverser_d"]
            })
            
        # B. 初始化融合中心
        fusion_center = AGMFusionCenter(base_model_params)
        
        # --- 时间步循环 ---
        for t in range(num_steps):
            # trajectories索引从0开始，对应第1步
            current_gt = trajectories[t]
            
            # 容器：存储所有传感器这一步的局部估计
            local_estimates_list = []
            
            # --- Step 1 & 2: 观测 + 局部滤波 ---
            for agent in agents:
                # 调用封装好的单步滤波函数
                # 注意：agent.state 此时存储的是上一时刻融合反馈回来的结果
                X_local = agent.step_filter(t, trajectories)
                local_estimates_list.append(X_local)
            
            # --- Step 3: 中心化融合 (AGM) ---
            # X_local 格式: [w, m, P, n]
            global_state = fusion_center.run(local_estimates_list, sensor_configs)
            
            # --- Step 4: 结果反馈 (Feedback) ---
            # 将全局结果切分，强制覆盖回每个传感器，供 t+1 时刻使用
            feedback_list = fusion_center.distribute_to_sensors(global_state, sensor_configs)
            
            for i, agent in enumerate(agents):
                fb = feedback_list[i]
                # 显式更新 agent 的状态 [w, m, P, n]
                agent.state = [fb[0], fb[1], fb[2], len(fb[0])]
            
            # --- Step 5: 统计误差 ---
            # 使用全局融合结果计算 OSPA
            # State_extraction 返回 (estimates_list, num_estimates)
            # estimates_list 里的元素是 [x, vx, y, vy]
            global_estimates, n_est = State_extraction(global_state)
            
            ospa_val = calculate_ospa(global_estimates, current_gt, c=200)
            card_err = abs(n_est - len(current_gt))
            
            ospa_mc[mc, t] = ospa_val
            card_mc[mc, t] = card_err
            
    # --- 绘图 ---
    plot_results(ospa_mc, card_mc)

def plot_results(ospa, card_err):
    time_axis = np.arange(1, ospa.shape[1] + 1)
    mean_ospa = np.mean(ospa, axis=0)
    mean_card = np.mean(card_err, axis=0)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(time_axis, mean_ospa, 'b-', label='GM-PHD + AGM Fusion')
    plt.title('Average OSPA Distance')
    plt.xlabel('Time Step (s)')
    plt.ylabel('OSPA (m)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(time_axis, mean_card, 'r-', label='GM-PHD + AGM Fusion')
    plt.title('Average Cardinality Error')
    plt.xlabel('Time Step (s)')
    plt.ylabel('Error Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()