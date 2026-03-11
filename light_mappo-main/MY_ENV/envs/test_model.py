import numpy as np
import random
from typing import Tuple
from math import atan2, degrees
from scipy.linalg import sqrtm

# --- 1. 参数配置 ---
def model():
    model_params = {}
    # 观测参数
    model_params["obverser_d"] = 200        # 观测半径
    model_params["meas_sigma_r0"] = 10.0                 # m
    model_params["meas_sigma_theta0_deg"] = 0.5          # deg
    model_params["meas_sigma_r_eta"] = 0.02              # m/m
    model_params["meas_sigma_theta_eta_deg"] = 0.001     # deg/m
    model_params["obverser_R"] = np.diag([
        model_params["meas_sigma_r0"] ** 2,
        model_params["meas_sigma_theta0_deg"] ** 2,
    ])
    model_params["Zr"] = 5                  # 杂波泊松参数
    model_params["Pd"] = 0.95               # 检测概率
    
    # 区域参数
    x_min, x_max = -1000, 1000
    y_min, y_max = -1000, 1000
    model_params['surveillance_region'] = np.array([[x_min, x_max], [y_min, y_max]])
    
    # 占位符 (Step中动态更新)
    model_params['x_agent'] = 0
    model_params['y_agent'] = 0
    
    return model_params


def distance_dependent_meas_sigma(distance, model_params, angle_unit="deg"):
    d = float(max(0.0, distance))
    sigma_r0 = float(model_params.get("meas_sigma_r0", 10.0))
    sigma_theta0_deg = float(model_params.get("meas_sigma_theta0_deg", 0.5))
    eta_r = float(model_params.get("meas_sigma_r_eta", 0.0))
    eta_theta_deg = float(model_params.get("meas_sigma_theta_eta_deg", 0.0))

    sigma_r = max(1e-6, sigma_r0 + eta_r * d)
    sigma_theta_deg = max(1e-6, sigma_theta0_deg + eta_theta_deg * d)
    if angle_unit == "rad":
        return sigma_r, np.deg2rad(sigma_theta_deg)
    return sigma_r, sigma_theta_deg

# --- 2. 目标生成 ---
# 这里的参数顺序必须适配 target_search.py 的调用: targets(self.max_steps)
# 即第一个参数必须是 num_of_scans
def targets(num_of_scans, n_target=5, x_range=(-800, 800), y_range=(-800, 800), v_range=(-15, 15)):
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