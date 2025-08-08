import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from math import atan2, degrees, sqrt, pi, sin, cos

def model(x_agent, y_agent):
    model = {}
    model["obverser_d"] = 100
    model['num_scans'] = 100
    model['x_agent'] = x_agent
    model['y_agent'] = y_agent

    model["obverser_R"] = np.diag([10, 1])
    model["Zr"] = 5
    model["Pd"] = 0.98

    return model

def targets(n_target, x_range=(-500, 500), y_range=(-500, 500), v_range=(-10, 10), num_of_scans=100):
    targets_birth_time = np.random.randint(0, num_of_scans // 5, size=n_target).tolist()   #0-20时刻内出生
    targets_death_time = [num_of_scans, num_of_scans, num_of_scans, num_of_scans, num_of_scans,
                          num_of_scans]
    targets_start = []
    for _ in range(n_target):
        x = np.random.uniform(*x_range)
        vx = np.random.uniform(*v_range)
        y = np.random.uniform(*y_range)
        vy = np.random.uniform(*v_range)
        targets_start.append(np.array([x, vx, y, vy]))
    return targets_birth_time, targets_death_time, targets_start

def target_CV(targets_birth_time, targets_death_time, targets_start, step, x_min, x_max, y_min, y_max, 
                          noise=True):
    T = 1
    A = np.array([[1, T, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, T],
                [0, 0, 0, 1]])
    Q = np.diag([1, 0.01, 1, 0.01])

    num_of_scans = step
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
                target_state += np.random.multivariate_normal(np.zeros(target_state.size), Q)
            if target_state[0] < x_min or target_state[0] > \
                    x_max or target_state[2] < y_min or \
                    target_state[2] > y_max:
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

    z_polar = []
    for target_state in targets_t:
        x, y = target_state[0], target_state[2]  # 假设状态中第 0 和第 2 是 x 和 y 坐标
        d = np.sqrt((x - x_agent)**2 + (y - y_agent)**2)
        if d <= obverser_d and Pd > np.random.rand():
        # 计算极坐标观测并添加噪声
            r, theta = compute_r_theta_2d(x, y, x_agent, y_agent)
            noisy_obs = np.array([r, theta]) + np.linalg.cholesky(obverser_R) @ np.random.randn(2)
            noisy_obs[1] = noisy_obs[1] % 360
            # observed_targets.append(noisy_obs)
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

    z_cartesian = []  # 用于存储每个时刻的直角坐标信息

    cartesian_obs_list = []
    for obs in z_polar:
        r = obs[0]
        theta = np.radians(obs[1])
        x = r * np.cos(theta) + x_agent
        y = r * np.sin(theta) + y_agent
        cartesian_obs_list.append(np.array([x, y]))
        
    return cartesian_obs_list
