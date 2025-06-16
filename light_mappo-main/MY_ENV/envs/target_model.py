import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from math import atan2, degrees, sqrt, pi, sin, cos

def model():
    model = {}

    model["obverser_d"] = 100
    model['num_scans'] = 100
    model['x_agent'] = 0
    model['y_agent'] = 0

    model["obverser_R"] = np.diag([10, 1])
    model["Zr"] = 5
    model["Pd"] = 0.98

    model["N"] = 100

    x_min = -1000
    x_max = 1000
    y_min = -1000
    y_max = 1000
    model['surveillance_region'] = np.array([[x_min, x_max], [y_min, y_max]])

    return model

def targets(num_of_scans=100):
    # targets_birth_time = [1, 1, 1, 20, 20, 20, 40, 40, 60, 60, 80, 80]
    # targets_birth_time = (np.array(targets_birth_time) - 1).tolist()
    # targets_death_time = [70, num_of_scans, 70, num_of_scans, num_of_scans, num_of_scans,
    #                       num_of_scans, num_of_scans, num_of_scans, num_of_scans, num_of_scans,
    #                       num_of_scans]
    # targets_start = [np.array([0., 0., 0., -10.]),
    #                  np.array([400., -10., -600., 5.]),
    #                  np.array([-800., 20., -200., -5.]),

    #                  np.array([400., -7., -600., -4.]),
    #                  np.array([400., -2.5, -600., 10.]),
    #                  np.array([0., 7.5, 0., -5.]),

    #                  np.array([-800., 12., -200., 7.]),
    #                  np.array([-200., 15., 800., -10.]),

    #                  np.array([-800., 3., -200., 15.]),
    #                  np.array([-200., -3., 800., -15.]),

    #                  np.array([0., -20., 0., -15.]),
    #                  np.array([-200., 15., 800., -5.])]

    targets_birth_time = [1, 1, 1, 20, 20, 20]
    targets_birth_time = (np.array(targets_birth_time) - 1).tolist()
    targets_death_time = [
                          num_of_scans, num_of_scans, num_of_scans, num_of_scans, num_of_scans,
                          num_of_scans]
    targets_start = [np.array([0., 0., 0., -10.]),
                     np.array([400., -10., -600., 5.]),
                     np.array([-800., 20., -200., -5.]),

                     np.array([400., -7., -600., -4.]),
                     np.array([400., -2.5, -600., 10.]),
                     np.array([0., 7.5, 0., -5.]),
                    ]
    return targets_birth_time, targets_death_time, targets_start

def target_CV(model, targets_birth_time, targets_death_time, targets_start, targets_spw_time_brttgt_vel=[],
                          noise=True):
    T = 1
    A = np.array([[1, T, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, T],
                [0, 0, 0, 1]])
    Q = np.diag([1, 0.01, 1, 0.01])

    num_of_scans = model['num_scans']
    trajectories = []
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
            if target_state[0] < model['surveillance_region'][0][0] or target_state[0] > \
                    model['surveillance_region'][0][1] or target_state[2] < model['surveillance_region'][1][0] or \
                    target_state[2] > model['surveillance_region'][1][1]:
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

def observe_Fov(model, trajectories):
    obverser_d = model["obverser_d"]
    x_agent = model['x_agent']
    y_agent = model['y_agent']
    obverser_R = model["obverser_R"] 
    Zr = model["Zr"] 
    Pd = model["Pd"] 
    N = model["N"]

    z_polar = [[] for _ in range(N)]  # 初始化N个空观测集

    for t in range(N):
        current_targets = trajectories[t]
        n_targets = len(current_targets)
        observed_targets = []
        
        # 处理真实目标
        for i in range(n_targets):
            target_state = current_targets[i]  # 取出第 i 个目标的状态
            x, y = target_state[0], target_state[2]  # 假设状态中第 0 和第 2 是 x 和 y 坐标
            d = np.sqrt((x - x_agent)**2 + (y - y_agent)**2)
            
            if d <= obverser_d and Pd > np.random.rand():
                # 计算极坐标观测并添加噪声
                r, theta = compute_r_theta_2d(x, y, x_agent, y_agent)
                noisy_obs = np.array([r, theta]) + np.linalg.cholesky(obverser_R) @ np.random.randn(2)
                noisy_obs[1] = noisy_obs[1] % 360
                # observed_targets.append(noisy_obs)
                z_polar[t].append(noisy_obs)
        
        # 生成杂波（圆形均匀分布）
        clutter = generate_clutter_2d(x_agent, y_agent, obverser_d, Zr)
        
        # # 合并真实观测和杂波
        # z_polar[t] = np.hstack([obs_array, clutter]) if obs_array.size > 0 else clutter
        z_polar[t].extend(clutter)
    
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

    for t in range(len(z_polar)):
        polar_obs_list = z_polar[t]  # 列表，每个元素是np.array([r, θ])

        cartesian_obs_list = []
        for obs in polar_obs_list:
            r = obs[0]
            theta = np.radians(obs[1])
            x = r * np.cos(theta) + x_agent
            y = r * np.sin(theta) + y_agent
            cartesian_obs_list.append(np.array([x, y]))

        z_cartesian.append(cartesian_obs_list)  # 直接append列表
        
    return z_cartesian


# model = model()
# targets_birth_time, targets_death_time, targets_start = targets()
# trajectories, targets_tracks = target_CV(model, targets_birth_time, targets_death_time, targets_start, targets_spw_time_brttgt_vel=[],
#                           noise=True) #trajectories某一时刻内所有目标的状态、 targets_tracks 目标在所有时刻的轨迹

# # 调用 observe_Fov 函数
# z_polar = observe_Fov(model, trajectories)

# # 打印某一时刻的观测结果
# t = 50  # 选择第一个时间步
# print(f"Time step {50}:")
# print("Polar observations (r, theta):")
# print(z_polar[50])

# # 创建图形
# plt.figure(figsize=(10, 8))

# # 绘制每个目标的轨迹
# for i, target_trajectory in targets_tracks.items():
#     # 将目标的轨迹转换为 numpy 数组，方便处理
#     target_trajectory = np.array(target_trajectory)
    
#     # 获取目标轨迹的 x 和 y 坐标
#     x_trajectory = target_trajectory[:, 0]  # 假设目标的 x 位置在第一列
#     y_trajectory = target_trajectory[:, 2]  # 假设目标的 y 位置在第三列
    
#     # 绘制目标的轨迹
#     plt.plot(x_trajectory, y_trajectory, label=f"Target {i+1}")

# # 设置图形属性
# plt.title("Target Trajectories")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.xlim(model['surveillance_region'][0][0], model['surveillance_region'][0][1])
# plt.ylim(model['surveillance_region'][1][0], model['surveillance_region'][1][1])
# plt.legend()
# plt.grid(True)
# plt.show()
