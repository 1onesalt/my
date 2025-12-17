import numpy as np
from target_model2 import model, targets, target_CV, observe_Fov, polar2dicaer
from PHD import PHD
from PHD import State_extraction, generate
from OSPA import ospa
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def main():
    num_steps = 100
    model1_data = model(0,0, num_steps)
    model2_data = model(400,400, num_steps)
    targets_birth_time, targets_death_time, targets_start = targets(2, num_steps)

    trajectories, targets_tracks = target_CV( targets_birth_time, targets_death_time, targets_start, num_steps,
                            noise=False) #trajectories某一时刻内所有目标的状态、 targets_tracks 目标在所有时刻的轨迹

    state1 = [
        0,                      # 权重
        np.zeros((1, 4)),       # 均值
        np.zeros((4, 4)),       # 协方差
        0                       # 粒子数量
    ]

    state2 = [
        0,                      # 权重
        np.zeros((1, 4)),       # 均值
        np.zeros((4, 4)),       # 协方差
        0                       # 粒子数量
    ]

    ospa1 = np.full(num_steps, np.nan)
    ospa2 = np.full(num_steps, np.nan)
    estimated_states1 = []  # 保存传感器1的估计状态
    estimated_states2 = []  # 保存传感器2的估计状态

    phd1 = PHD(model1_data)
    phd2 = PHD(model2_data)


    for i in range(2, num_steps):

        z_polar1 = observe_Fov(model1_data, trajectories)#视域内并加杂波
        Z_dicaer1 = polar2dicaer(z_polar1, model1_data)  #转直角坐标

        z_polar2 = observe_Fov(model2_data, trajectories)
        Z_dicaer2 = polar2dicaer(z_polar2, model2_data)  #返回的是列表

        phd1.init_params(Z_dicaer1[i - 2], Z_dicaer1[i - 1], z_polar1[i], state1)
        phd1.w_new, phd1.m_new, phd1.P_new, phd1.J_new = generate(phd1.z_lastD, phd1.nums_z_lastD, phd1.z_nowD, phd1.nums_z_nowD, phd1.Vx_thre, phd1.Vy_thre)

        phd2.init_params(Z_dicaer2[i - 2], Z_dicaer2[i - 1], z_polar2[i], state2)
        phd2.w_new, phd2.m_new, phd2.P_new, phd2.J_new = generate(phd2.z_lastD, phd2.nums_z_lastD, phd2.z_nowD, phd2.nums_z_nowD, phd2.Vx_thre, phd2.Vy_thre)

        # 执行 predict_update
        X_now1 = phd1.predict_update()
        X_now2 = phd2.predict_update()

        # 提取并保存
        state_draw1, num_draw1 = State_extraction(X_now1)
        state_draw2, num_draw2 = State_extraction(X_now2)

        # 更新 state 以供下一步使用
        state1 = X_now1
        state2 = X_now2

        # 用每个时刻的真实观测（trajectories[i]）和估计结果计算OSPA并保存
        ospa1[i] = ospa(trajectories[i], state_draw1)
        ospa2[i] = ospa(trajectories[i], state_draw2)

        estimated_states1.append(state_draw1)
        estimated_states2.append(state_draw2)

        #print("时刻", i, "目标轨迹", trajectories[i], "直角坐标观测", Z_dicaer1[i])

    plot_results(ospa1, ospa2, num_steps)
    plot_position_comparison(estimated_states1, estimated_states2, trajectories,sensor1_pos=(0, 0), sensor2_pos=(400, 400), detection_range=500)
    # plot_targets_tracks(targets_tracks)
    # plot_observations(Z_dicaer1, Z_dicaer2)
    #animate_observations(Z_dicaer1, Z_dicaer2, targets_tracks)


    #plot_environment(trajectories, z_polar1, z_polar2, model1_data, model2_data)

def plot_position_comparison(est_states1, est_states2, true_states,sensor1_pos=(-200, -200), sensor2_pos=(200, 200), detection_range=500):
    """绘制所有时刻的位置对比图"""
    plt.figure(figsize=(12, 8))
    
    # 提取所有真实位置
    all_true_x = []
    all_true_y = []
    for states in true_states:
        for state in states:
            if len(state) >= 4:  # [x, vx, y, vy]
                all_true_x.append(state[0])
                all_true_y.append(state[2])
    
    # 提取传感器1估计位置
    all_est1_x = []
    all_est1_y = []
    for states in est_states1:
        for state in states:
            if len(state) >= 4:
                all_est1_x.append(state[0])
                all_est1_y.append(state[2])
    
    # 提取传感器2估计位置
    all_est2_x = []
    all_est2_y = []
    for states in est_states2:
        for state in states:
            if len(state) >= 4:
                all_est2_x.append(state[0])
                all_est2_y.append(state[2])
    
    # 绘制传感器检测范围
    circle1 = patches.Circle(sensor1_pos, detection_range, fill=False, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='传感器1检测范围')
    circle2 = patches.Circle(sensor2_pos, detection_range, fill=False, color='red', linestyle='--', linewidth=2, alpha=0.5, label='传感器2检测范围')
    
    ax = plt.gca()
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # 绘制散点图
    plt.scatter(all_true_x, all_true_y, c='green', marker='o', s=50, label='真实位置', alpha=0.7)
    plt.scatter(all_est1_x, all_est1_y, c='blue', marker='x', s=40, label='传感器1估计', alpha=0.7)
    plt.scatter(all_est2_x, all_est2_y, c='red', marker='+', s=40, label='传感器2估计', alpha=0.7)
    
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.title('所有时刻位置估计对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # 打印简单统计
    print(f"真实位置点数: {len(all_true_x)}")
    print(f"传感器1估计点数: {len(all_est1_x)}")
    print(f"传感器2估计点数: {len(all_est2_x)}")


def plot_results(ospa1, ospa2, num_steps):
    """绘制结果图表"""
    t = np.arange(num_steps)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # OSPA距离图
    ax1.plot(t, ospa1, 'b-', linewidth=2, label='传感器1 OSPA')
    ax1.plot(t, ospa2, 'r-', linewidth=2, label='传感器2 OSPA')
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('OSPA 距离')
    ax1.set_title('OSPA距离随时间变化')
    ax1.legend()
    ax1.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("\n=== 性能统计 ===")
    print(f"传感器1 - 平均OSPA: {np.nanmean(ospa1[2:]):.4f}")
    print(f"传感器2 - 平均OSPA: {np.nanmean(ospa2[2:]):.4f}")


def plot_environment(trajectories, z_polar1, z_polar2, model1, model2):  #绘制环境图
    """
    绘制环境图，包括：
    - 真实目标轨迹（所有时刻）
    - 传感器位置
    - 传感器视域范围
    - 杂波分布（所有时刻所有杂波）
    """

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # ----------- 1. 绘制真实轨迹 -----------
    for target_traj in trajectories:
        for st in target_traj:
            x, y = st[0], st[2]  # 你的状态结构 [x, vx, y, vy]
            plt.scatter(x, y, c="green", s=10, alpha=0.5)

    # ----------- 2. 绘制传感器位置 -----------
    x1, y1 = model1["x_agent"], model1["y_agent"]
    x2, y2 = model2["x_agent"], model2["y_agent"]
    plt.scatter(x1, y1, c="blue", marker="o", s=120, label="传感器1")
    plt.scatter(x2, y2, c="red", marker="o", s=120, label="传感器2")

    # ----------- 3. 绘制传感器视域圆 -----------
    r1 = model1["obverser_d"]
    r2 = model2["obverser_d"]

    circle1 = patches.Circle((x1, y1), r1, fill=False, linestyle="--", color="blue", alpha=0.5, label="传感器1视域")
    circle2 = patches.Circle((x2, y2), r2, fill=False, linestyle="--", color="red", alpha=0.5, label="传感器2视域")

    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # ----------- 4. 绘制杂波点 -----------
    clutter_x = []
    clutter_y = []

    # z_polar1 / z_polar2 内容结构：一系列 [ [r, θ], [r, θ], ... ]（包含目标和杂波）
    def extract_clutter(z_polar, model):
        x_radar, y_radar = model["x_agent"], model["y_agent"]
        r_detect = model["obverser_d"]
        out_x, out_y = [], []

        for t in z_polar:
            for obs in t:
                r, theta = obs

                # theta 单位为度
                theta_rad = np.radians(theta)
                x = x_radar + r * np.cos(theta_rad)
                y = y_radar + r * np.sin(theta_rad)

                # 判定是否为杂波
                # 判断观测点是否不在真实目标附近
                out_x.append(x)
                out_y.append(y)
        return out_x, out_y

    cx1, cy1 = extract_clutter(z_polar1, model1)
    cx2, cy2 = extract_clutter(z_polar2, model2)

    plt.scatter(cx1, cy1, c="blue", s=10, alpha=0.3, label="传感器1杂波")
    plt.scatter(cx2, cy2, c="red", s=10, alpha=0.3, label="传感器2杂波")

    # ----------- 图形设置 -----------
    plt.title("真实轨迹 + 传感器 + 视域 + 杂波分布图")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.show()

def plot_targets_tracks(Z_dicaer):
    plt.figure(figsize=(8, 8))
    plt.title("轨迹", fontsize=16)

    # 为每个目标绘制轨迹
    for target_id, track in Z_dicaer.items():
        if len(track) == 0:
            continue
        
        track = np.array(track)   # shape: (T, 4)
        xs = track[:, 0]          # x
        ys = track[:, 2]          # y

        plt.plot(xs, ys, '-', linewidth=2, label=f"目标 {target_id}")
        plt.scatter(xs[0], ys[0], marker='o', color='green')   # 出生点
        plt.scatter(xs[-1], ys[-1], marker='x', color='red')   # 终止点

    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_Z_tracks(targets_tracks):
    plt.figure(figsize=(8, 8))
    plt.title("目标真实轨迹", fontsize=16)
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")

    # 为每个目标绘制轨迹
    for target_id, track in targets_tracks.items():
        if len(track) == 0:
            continue
        
        track = np.array(track)   # shape: (T, 4)
        xs = track[:, 0]          # x
        ys = track[:, 2]          # y

        plt.plot(xs, ys, '-', linewidth=2, label=f"目标 {target_id}")
        plt.scatter(xs[0], ys[0], marker='o', color='green')   # 出生点
        plt.scatter(xs[-1], ys[-1], marker='x', color='red')   # 终止点

    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def plot_observations(Z_dicaer1, Z_dicaer2, targets_tracks=None): #绘制两台传感器的观测点
    """
    绘制两台传感器的观测点
    Z_dicaer1, Z_dicaer2: 每个时刻的观测点列表 [[array([x,y]), ...], ...]
    targets_tracks: 可选，真实目标轨迹
    """
    plt.figure(figsize=(8, 8))
    plt.title("传感器观测点分布", fontsize=16)
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")

    # 绘制观测点
    for obs_list in Z_dicaer1:
        if len(obs_list) > 0:
            obs_array = np.array(obs_list)
            plt.scatter(obs_array[:,0], obs_array[:,1], c='blue', s=10, alpha=0.6, label='传感器1')
    for obs_list in Z_dicaer2:
        if len(obs_list) > 0:
            obs_array = np.array(obs_list)
            plt.scatter(obs_array[:,0], obs_array[:,1], c='orange', s=10, alpha=0.6, label='传感器2')

    # 避免重复图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # 可选：绘制真实轨迹
    if targets_tracks is not None:
        for target_id, track in targets_tracks.items():
            if len(track) == 0:
                continue
            track = np.array(track)
            plt.plot(track[:,0], track[:,2], '-', color='green', linewidth=1.5, label=f'目标{target_id}')

    plt.grid(True)
    plt.axis('equal')
    plt.show()

def animate_observations(Z_dicaer1, Z_dicaer2, targets_tracks):
    num_steps = len(Z_dicaer1)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("每个时刻目标与观测点")

    scat1 = ax.scatter([], [], c='blue', s=30, alpha=0.6, label='传感器1')
    scat2 = ax.scatter([], [], c='orange', s=30, alpha=0.6, label='传感器2')
    traj_lines = [ax.plot([], [], '-', color='green', lw=1.5)[0] for _ in targets_tracks]

    def init():
        scat1.set_offsets(np.empty((0,2)))
        scat2.set_offsets(np.empty((0,2)))
        for line in traj_lines:
            line.set_data([], [])
        return [scat1, scat2, *traj_lines]

    def update(frame):
        # 当前时间步观测
        obs1 = np.array(Z_dicaer1[frame]) if len(Z_dicaer1[frame])>0 else np.empty((0,2))
        obs2 = np.array(Z_dicaer2[frame]) if len(Z_dicaer2[frame])>0 else np.empty((0,2))

        scat1.set_offsets(obs1)
        scat2.set_offsets(obs2)

        # 当前轨迹（从出生到当前时间）
        for i, (target_id, track) in enumerate(targets_tracks.items()):
            if len(track)>frame:
                track_array = np.array(track[:frame+1])
                traj_lines[i].set_data(track_array[:,0], track_array[:,2])

        return [scat1, scat2, *traj_lines]

    ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=100)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    