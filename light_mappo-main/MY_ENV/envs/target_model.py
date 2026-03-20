import numpy as np
import random
from typing import Tuple
from math import atan2, degrees
from scipy.linalg import sqrtm

# --- 1. 参数配置 ---
def model():
    model_params = {}
    # 观测参数
    model_params["obverser_d"] = 500        # 观测半径
    # 距离相关噪声参数（与 chapter4 文档一致）
    model_params["meas_sigma_r0"] = 10.0                 # m
    model_params["meas_sigma_theta0_deg"] = 0.5          # deg
    model_params["meas_sigma_r_eta"] = 0.02              # m/m
    model_params["meas_sigma_theta_eta_deg"] = 0.001     # deg/m
    # 兼容旧代码保留常量协方差（角度单位：deg）
    model_params["obverser_R"] = np.diag([
        model_params["meas_sigma_r0"] ** 2,
        model_params["meas_sigma_theta0_deg"] ** 2,
    ])
    model_params["Zr"] = 5                  # 杂波泊松参数
    model_params["Pd"] = 0.95               # 检测概率
    
    # 区域参数
    x_min, x_max = -2000, 2000
    y_min, y_max = -2000, 2000
    model_params['surveillance_region'] = np.array([[x_min, x_max], [y_min, y_max]])
    
    # 占位符 (Step中动态更新)
    model_params['x_agent'] = 0
    model_params['y_agent'] = 0
    
    return model_params


def distance_dependent_meas_sigma(distance, model_params, angle_unit="deg"):
    """
    按传感器-目标距离计算测量标准差。
    angle_unit: "deg" 或 "rad"
    """
    d = float(max(0.0, distance))
    sigma_r0 = float(model_params.get("meas_sigma_r0", 10.0))
    sigma_theta0_deg = float(model_params.get("meas_sigma_theta0_deg", 0.5))
    eta_r = float(model_params.get("meas_sigma_r_eta", 0.0))
    eta_theta_deg = float(model_params.get("meas_sigma_theta_eta_deg", 0.0))

    sigma_r = sigma_r0 + eta_r * d
    sigma_theta_deg = sigma_theta0_deg + eta_theta_deg * d
    sigma_r = max(1e-6, sigma_r)
    sigma_theta_deg = max(1e-6, sigma_theta_deg)

    if angle_unit == "rad":
        return sigma_r, np.deg2rad(sigma_theta_deg)
    return sigma_r, sigma_theta_deg

# --- 2. 目标生成 ---
# 这里的参数顺序必须适配 target_search.py 的调用: targets(self.max_steps)
# 即第一个参数必须是 num_of_scans
def targets(num_of_scans, n_target=5, x_range=(-2000, 2000), y_range=(-2000, 2000), v_range=(-20, 20)):
    # 随机出生时间 (0 到 总时长的20%)
    targets_birth_time = np.random.randint(0, int(num_of_scans * 0.2) + 1, size=n_target).tolist()
    targets_death_time = [num_of_scans] * n_target

    targets_start = []
    for _ in range(n_target):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        vx = np.random.uniform(*v_range)
        vy = np.random.uniform(*v_range)
        targets_start.append(np.array([x, vx, y, vy]))
        
    return targets_birth_time, targets_death_time, targets_start

# --- 3. 轨迹生成 (关键修复) ---
# 必须接收 model_params 作为第一个参数
def target_CV(model_params, targets_birth_time, targets_death_time, targets_start, noise=True):
    region = model_params['surveillance_region']
    x_min, x_max = region[0]
    y_min, y_max = region[1]
    
    # 从 model_params 中获取步数，或者使用默认值
    num_of_scans = model_params.get('num_scans', 200)

    T = 1
    A = np.array([[1, T, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, T],
                  [0, 0, 0, 1]])
    Q = np.diag([1, 0.1, 1, 0.1]) 

    trajectories = [[] for _ in range(num_of_scans)]
    targets_tracks = {}

    for i, start in enumerate(targets_start):
        target_state = start
        targets_tracks[i] = []
        
        # 安全检查：确保 death_time 不超过 num_of_scans
        end_time = min(targets_death_time[i], num_of_scans)
        
        for k in range(targets_birth_time[i], end_time):
            target_state = A @ target_state
            if noise:
                target_state += np.random.multivariate_normal(np.zeros(4), Q)
            
            # 越界处理
            if (target_state[0] < x_min or target_state[0] > x_max or 
                target_state[2] < y_min or target_state[2] > y_max):
                break
            
            if k < num_of_scans: # 二次检查索引
                trajectories[k].append(target_state)
                targets_tracks[i].append(target_state)
    
    return trajectories, targets_tracks

# --- 4. 辅助函数 ---
def compute_r_theta_2d(x, y, x_radar, y_radar):
    dx, dy = x - x_radar, y - y_radar
    r = np.sqrt(dx**2 + dy**2)
    theta = degrees(atan2(dy, dx)) % 360
    return r, theta

def generate_clutter_2d(x_radar, y_radar, r_detect, n_clutter):
    if n_clutter <= 0:
        return []
    
    # 使用泊松分布生成杂波数量
    num = np.random.poisson(n_clutter)
    if num == 0: 
        return []

    angles = np.random.uniform(0, 2*np.pi, num)
    radii = r_detect * np.sqrt(np.random.uniform(0, 1, num))

    x_clutter = x_radar + radii * np.cos(angles)
    y_clutter = y_radar + radii * np.sin(angles)

    clutter = []
    for x_c, y_c in zip(x_clutter, y_clutter):
        r, theta = compute_r_theta_2d(x_c, y_c, x_radar, y_radar)
        clutter.append(np.array([r, theta]))

    return clutter

# --- 5. 观测生成 (适配 step 函数) ---
def observe_Fov(model_params, targets_current_step):
    obverser_d = model_params["obverser_d"]
    x_agent = model_params['x_agent']
    y_agent = model_params['y_agent']
    Zr = model_params["Zr"] 
    Pd = model_params["Pd"] 

    z_polar = []
    
    # 真实目标
    for target_state in targets_current_step:
        x, y = target_state[0], target_state[2]
        d = np.sqrt((x - x_agent)**2 + (y - y_agent)**2)

        if d <= obverser_d and np.random.rand() < Pd:
            r, theta = compute_r_theta_2d(x, y, x_agent, y_agent)
            std_r, std_theta_deg = distance_dependent_meas_sigma(
                d, model_params, angle_unit="deg"
            )
            z_polar.append(np.array([
                r + np.random.randn() * std_r,
                (theta + np.random.randn() * std_theta_deg) % 360,
            ]))

    # 杂波
    z_polar.extend(generate_clutter_2d(x_agent, y_agent, obverser_d, Zr))
    
    return z_polar

def polar2dicaer(z_polar, model_params):
    x_agent = model_params['x_agent']
    y_agent = model_params['y_agent']

    z_cartesian = []
    for obs in z_polar:
        r = obs[0]
        theta = np.radians(obs[1])
        x = r * np.cos(theta) + x_agent
        y = r * np.sin(theta) + y_agent
        z_cartesian.append(np.array([x, y]))

    return z_cartesian



























# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from typing import Tuple
# from math import atan2, degrees
# from scipy.linalg import sqrtm

# def model():
#     model = {}
#     model["obverser_d"] = 800             #观测半径
#     model['x_agent'] = 0
#     model['y_agent'] = 0
#     x_min, x_max = -1000, 1000
#     y_min, y_max = -1000, 1000
#     model['surveillance_region'] = np.array([[x_min, x_max], [y_min, y_max]])
#     model["obverser_R"] = np.diag([0.5, 0.1])   #观测噪声协方差矩阵
#     model["Zr"] = 2                         #杂波
#     model["Pd"] = 0.98                   #检测概率
 
#     return model


# def targets(n_target, num_of_scans, x_range=(-300, 300), y_range=(-300, 300), v_range=(-6, 6)):
#     """
#     生成目标初始状态及出生死亡时间

#     输入:目标数量n_target
#     输出:targets_birth_time: 目标出生时间列表
#         targets_death_time: 目标死亡时间列表
#         targets_start: 目标初始状态列表
#     """
#     targets_birth_time = np.random.randint(0, 10,  size=n_target).tolist()   #0-20时刻内出生共n_target个目标
#     targets_death_time = [num_of_scans] * n_target

#     targets_start = []
#     for _ in range(n_target):
#         x = np.random.uniform(*x_range)
#         vx = np.random.uniform(*v_range)
#         y = np.random.uniform(*y_range)
#         vy = np.random.uniform(*v_range)
#         targets_start.append(np.array([x, vx, y, vy]))
#     return targets_birth_time, targets_death_time, targets_start

# def target_CV(targets_birth_time, targets_death_time, targets_start, num_steps, x_min=-1000, x_max=1000, y_min=-1000, y_max=1000, 
#                           noise=True):
#     T = 1
#     A = np.array([[1, T, 0, 0],
#                 [0, 1, 0, 0],
#                 [0, 0, 1, T],
#                 [0, 0, 0, 1]])
#     Q = np.diag([1, 0.01, 1, 0.01])

#     num_of_scans = num_steps
#     trajectories = []                #轨迹
#     for i in range(num_of_scans):
#         trajectories.append([])
    
#     targets_tracks = {}
#     for i, start in enumerate(targets_start):
#         target_state = start
#         targets_tracks[i] = []
#         for k in range(targets_birth_time[i], min(targets_death_time[i], num_of_scans)):
#             target_state = A @ target_state
#             if noise:
#                 target_state += sqrtm(Q) @ np.random.randn(4)
            
#             if target_state[0] < x_min or target_state[0] > x_max or target_state[2] < y_min or target_state[2] > y_max:
#                 targets_death_time[i] = k - 1
#                 break
#             trajectories[k].append(target_state)
#             targets_tracks[i].append(target_state)
    
#     return trajectories, targets_tracks

# def compute_r_theta_2d(
#     x: float, 
#     y: float, 
#     x_radar: float, 
#     y_radar: float
# ) -> Tuple[float, float]:
#     """
#     计算2D极坐标（距离和方位角）
    
#     参数:
#         x, y: 目标直角坐标
#         x_radar, y_radar: 传感器坐标
        
#     返回:
#         (距离, 方位角(度)) 方位角范围[0,360)
#     """
#     dx, dy = x - x_radar, y - y_radar
#     r = np.sqrt(dx**2 + dy**2)
        
#     # 使用atan2直接处理所有象限情况
#     theta = degrees(atan2(dy, dx)) % 360  # 转换为0-360度范围
    
#     return r, theta


# def generate_clutter_2d(x_radar, y_radar, r_detect, n_clutter):
#     """
#     返回形式：[array([r, θ]), array([r, θ]), ...]
#     输入:传感器的x、y位置, 探测半径, 杂波数量
#     """
#     if n_clutter == 0:
#         return []

#     angles = np.random.uniform(0, 2*np.pi, n_clutter)
#     radii = r_detect * np.sqrt(np.random.uniform(0, 1, n_clutter))

#     x_clutter = x_radar + radii * np.cos(angles)
#     y_clutter = y_radar + radii * np.sin(angles)

#     clutter = []
#     for x_c, y_c in zip(x_clutter, y_clutter):
#         r, theta = compute_r_theta_2d(x_c, y_c, x_radar, y_radar)
#         clutter.append(np.array([r, theta]))

#     return clutter

# def observe_Fov(model, targets_t):
#     obverser_d = model["obverser_d"]
#     x_agent = model['x_agent']
#     y_agent = model['y_agent']
#     obverser_R = model["obverser_R"] 
#     Zr = model["Zr"] 
#     Pd = model["Pd"] 

#     targets_t = np.array(targets_t)

#     if targets_t.ndim == 1:
#         targets_t = targets_t[np.newaxis, :]

#     z_polar = []
#     for i, target_state in enumerate(targets_t):

#         x, y = target_state[0], target_state[2]  # 假设状态中第 0 和第 2 是 x 和 y 坐标
#         d = np.sqrt((x - x_agent)**2 + (y - y_agent)**2)

#         if d <= obverser_d and Pd > np.random.rand():
#     # 计算极坐标观测并添加噪声
#             r, theta = compute_r_theta_2d(x, y, x_agent, y_agent)
#             noisy_obs = np.array([r, theta]) + np.linalg.cholesky(obverser_R) @ np.random.randn(2)
#             noisy_obs[1] = noisy_obs[1] % 360
#             z_polar.append(noisy_obs)

#         # 生成杂波（圆形均匀分布）
#         clutter = generate_clutter_2d(x_agent, y_agent, obverser_d, Zr)
#         z_polar.extend(clutter)

#     return z_polar


# def polar2dicaer(z_polar, model):
#     """
#     将每个时刻的极坐标观测信息转换为直角坐标信息。
#     返回:
#         z_cartesian: 每个时刻的直角坐标观测信息
#     """
#     x_agent = model['x_agent']
#     y_agent = model['y_agent']

#     z_cartesian = []  # 每个时刻的直角坐标结果

#     # for t, obs_list in enumerate(z_polar):   # 遍历每个时刻
#     #     cartesian_obs_list = []              # 当前时刻的所有观测
#     #     if len(obs_list) > 0:
#     for obs in z_polar:             # 遍历该时刻的每个观测
#         r = obs[0]
#         theta = np.radians(obs[1])
#         x = r * np.cos(theta) + x_agent
#         y = r * np.sin(theta) + y_agent
#         #cartesian_obs_list.append(np.array([x, y]))
#         # 即使没有观测，也要保留空列表
#         z_cartesian.append(np.array([x, y]))

#     return z_cartesian
