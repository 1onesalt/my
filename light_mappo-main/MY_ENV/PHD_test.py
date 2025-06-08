import numpy as np
import random
import matplotlib.pyplot as plt

from target_model import model, targets, target_CV, observe_Fov
from target_model import generate_clutter_2d, polar2dicaer
from PHD import PHD
from PHD import State_extraction

model_data = model()
targets_birth_time, targets_death_time, targets_start = targets()
# print(targets_death_time)
trajectories, targets_tracks = target_CV(model_data, targets_birth_time, targets_death_time, targets_start, targets_spw_time_brttgt_vel=[],
                          noise=True) #trajectories某一时刻内所有目标的状态、 targets_tracks 目标在所有时刻的轨迹
# print(targets_tracks)
z_polar = observe_Fov(model_data, trajectories)
Z_dicaer = polar2dicaer(z_polar, model_data)  #返回的是列表


state = [
    0,                      # 权重
    np.zeros((1, 4)),       # 均值
    np.zeros((4, 4)),       # 协方差
    0                       # 粒子数量
]

# print(Z_dicaer[0])
# print(Z_dicaer[1])
# print(z_polar[2])
for i in range(2, 100):
    phd = PHD(model_data, Z_dicaer[i - 2], Z_dicaer[i - 1], z_polar[i], state)  # 调用PHD函数进行处理
    X_now = phd.predict_update()
    state_draw, num_draw  = State_extraction(X_now)

    print(state_draw)
    print(num_draw)
    #(self, params, z_lastD, z_nowD, z_nowP, X_laxt):

