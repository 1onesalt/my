import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Tuple
from math import atan2, degrees
from scipy.linalg import sqrtm

def model():
    model = {}
    model["obverser_d"] = 800             #观测半径
    # model['num_scans'] = num_scans
    # model['x_agent'] = x_agent
    # model['y_agent'] = y_agent

    model["obverser_R"] = np.diag([0.5, 0.1])   #观测噪声协方差矩阵
    model["Zr"] = 2                         #杂波
    model["Pd"] = 0.98                   #检测概率
 
    return model


def targets(n_target, num_of_scans, x_range=(-300, 300), y_range=(-300, 300), v_range=(-6, 6)):
    """
    生成目标初始状态及出生死亡时间

    输入:目标数量n_target
    输出:targets_birth_time: 目标出生时间列表
        targets_death_time: 目标死亡时间列表
        targets_start: 目标初始状态列表
    """
    targets_birth_time = np.random.randint(0, 1,  size=n_target).tolist()   #0-20时刻内出生共n_target个目标
    targets_death_time = [num_of_scans] * n_target

    targets_start = []
    for _ in range(n_target):
        x = np.random.uniform(*x_range)
        vx = np.random.uniform(*v_range)
        y = np.random.uniform(*y_range)
        vy = np.random.uniform(*v_range)
        targets_start.append(np.array([x, vx, y, vy]))
    return targets_birth_time, targets_death_time, targets_start

def target_CV(targets_birth_time, targets_death_time, targets_start, num_steps, x_min=-1000, x_max=1000, y_min=-1000, y_max=1000, 
                          noise=True):
    T = 1
    A = np.array([[1, T, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, T],
                [0, 0, 0, 1]])
    Q = np.diag([1, 0.01, 1, 0.01])

    num_of_scans = num_steps
    trajectories = []                #轨迹
    for i in range(num_of_scans):
        trajectories.append([])
    
    targets_tracks = {}
    for i, start in enumerate(targets_start):
        target_state = start
        targets_tracks[i] = []
        for k in range(targets_birth_time[i], min(targets_death_time[i], num_of_scans)):
            target_state = A @ target_state
            if noise:
                target_state += sqrtm(Q) @ np.random.randn(4)
            
            if target_state[0] < x_min or target_state[0] > x_max or target_state[2] < y_min or target_state[2] > y_max:
                targets_death_time[i] = k - 1
                break
            trajectories[k].append(target_state)
            targets_tracks[i].append(target_state)
    
    return trajectories, targets_tracks

def compute_r_theta_2d(
    x: float, 
    y: float, 
    x_radar: float, 
    y_radar: float
) -> Tuple[float, float]:
    """
    计算2D极坐标（距离和方位角）
    
    参数:
        x, y: 目标直角坐标
        x_radar, y_radar: 传感器坐标
        
    返回:
        (距离, 方位角(度)) 方位角范围[0,360)
    """
    dx, dy = x - x_radar, y - y_radar
    r = np.sqrt(dx**2 + dy**2)
        
    # 使用atan2直接处理所有象限情况
    theta = degrees(atan2(dy, dx)) % 360  # 转换为0-360度范围
    
    return r, theta


def generate_clutter_2d(x_radar, y_radar, r_detect, n_clutter):
    """
    返回形式：[array([r, θ]), array([r, θ]), ...]
    输入:传感器的x、y位置, 探测半径, 杂波数量
    """
    if n_clutter == 0:
        return []

    angles = np.random.uniform(0, 2*np.pi, n_clutter)
    radii = r_detect * np.sqrt(np.random.uniform(0, 1, n_clutter))

    x_clutter = x_radar + radii * np.cos(angles)
    y_clutter = y_radar + radii * np.sin(angles)

    clutter = []
    for x_c, y_c in zip(x_clutter, y_clutter):
        r, theta = compute_r_theta_2d(x_c, y_c, x_radar, y_radar)
        clutter.append(np.array([r, theta]))

    return clutter

def observe_Fov(model, targets_t):
    obverser_d = model["obverser_d"]
    x_agent = model['x_agent']
    y_agent = model['y_agent']
    obverser_R = model["obverser_R"] 
    Zr = model["Zr"] 
    Pd = model["Pd"] 

    targets_t = np.array(targets_t)

    if targets_t.ndim == 1:
        targets_t = targets_t[np.newaxis, :]

    z_polar = []
    for i, target_state in enumerate(targets_t):

        x, y = target_state[0], target_state[2]  # 假设状态中第 0 和第 2 是 x 和 y 坐标
        d = np.sqrt((x - x_agent)**2 + (y - y_agent)**2)

        if d <= obverser_d and Pd > np.random.rand():
    # 计算极坐标观测并添加噪声
            r, theta = compute_r_theta_2d(x, y, x_agent, y_agent)
            noisy_obs = np.array([r, theta]) + np.linalg.cholesky(obverser_R) @ np.random.randn(2)
            noisy_obs[1] = noisy_obs[1] % 360
            z_polar.append(noisy_obs)

        # 生成杂波（圆形均匀分布）
        clutter = generate_clutter_2d(x_agent, y_agent, obverser_d, Zr)
        z_polar.extend(clutter)

    return z_polar


def polar2dicaer(z_polar, model):
    """
    将每个时刻的极坐标观测信息转换为直角坐标信息。
    返回:
        z_cartesian: 每个时刻的直角坐标观测信息
    """
    x_agent = model['x_agent']
    y_agent = model['y_agent']

    z_cartesian = []  # 每个时刻的直角坐标结果

    # for t, obs_list in enumerate(z_polar):   # 遍历每个时刻
    #     cartesian_obs_list = []              # 当前时刻的所有观测
    #     if len(obs_list) > 0:
    for obs in z_polar:             # 遍历该时刻的每个观测
        r = obs[0]
        theta = np.radians(obs[1])
        x = r * np.cos(theta) + x_agent
        y = r * np.sin(theta) + y_agent
        #cartesian_obs_list.append(np.array([x, y]))
        # 即使没有观测，也要保留空列表
        z_cartesian.append(np.array([x, y]))

    return z_cartesian
