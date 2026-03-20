"""
验证 run104 训练模型的脚本
加载训练好的 MAPPO 模型，执行一个 episode，并将目标轨迹与传感器轨迹画图显示
"""
import os
import sys

# 避免 OpenMP 重复初始化报错 (numpy/torch 等)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from MY_ENV.envs.target_search import TargetSearchEnv
from MY_ENV.envs.my_wrappers import FlattenObservation
from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy


def parse_args(parser):
    """解析与训练一致的参数"""
    parser.add_argument("--scenario_name", type=str, default="MyEnv")
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to run dir (e.g. .../run104/models)")
    all_args = parser.parse_known_args()[0]
    return all_args


def load_policy(all_args, env, device):
    """加载策略网络"""
    from gym import spaces
    
    share_obs_space = env.share_observation_space[0]
    
    policy = RMAPPOPolicy(
        all_args,
        env.observation_space[0],
        share_obs_space,
        env.action_space[0],
        device=device,
    )
    
    model_dir = all_args.model_dir
    if model_dir and os.path.exists(os.path.join(model_dir, "actor.pt")):
        actor_state = torch.load(os.path.join(model_dir, "actor.pt"), map_location=device)
        policy.actor.load_state_dict(actor_state)
        print(f"[OK] Loaded actor: {model_dir}/actor.pt")
    else:
        print("[WARN] Model not found, using random policy")
    
    return policy


def _build_share_obs(obs, num_agents, use_centralized_V=True):
    """构建集中式 critic 观测 (与 env_runner 逻辑一致)"""
    # obs: list of arrays, each shape (obs_dim,)
    obs_arr = np.array(obs)
    if use_centralized_V:
        max_agents = num_agents
        active_agents = min(max_agents, len(obs_arr))
        active_mask = np.zeros((max_agents, 1), dtype=np.float32)
        active_mask[:active_agents, 0] = 1.0
        critic_tokens = np.concatenate([obs_arr, active_mask], axis=-1)
        share_obs = critic_tokens.reshape(-1)
        share_obs = np.tile(share_obs, (num_agents, 1))
    else:
        share_obs = obs_arr
    return share_obs


def main():
    default_model = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", "MyEnv", "MyEnv", "mappo", "check", "run104", "models"
    )
    
    parser = get_config()
    all_args = parse_args(parser)
    all_args.model_dir = all_args.model_path or getattr(all_args, "model_dir", None) or default_model
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)
    
    # 创建环境 (与 train.py 保持一致)
    curriculum_stage = all_args.curriculum_stage if all_args.use_curriculum else 0
    env = TargetSearchEnv(
        n_agent=all_args.num_agents,
        max_steps=all_args.episode_length,
        curriculum_stage=curriculum_stage,
    )
    env = FlattenObservation(env)
    env.seed(all_args.seed)
    
    # 获取底层环境用于读取 agent_pos 和 trajectories
    base_env = env.env
    
    # 加载策略
    policy = load_policy(all_args, env, device)
    
    # 轨迹记录
    sensor_paths_x = [[] for _ in range(all_args.num_agents)]
    sensor_paths_y = [[] for _ in range(all_args.num_agents)]
    target_paths_x = [[] for _ in range(base_env.n_targets)]
    target_paths_y = [[] for _ in range(base_env.n_targets)]
    
    # 重置环境
    obs_list = env.reset()
    obs = np.array(obs_list)
    
    rnn_states = np.zeros(
        (1, all_args.num_agents, all_args.recurrent_N, all_args.hidden_size),
        dtype=np.float32,
    )
    masks = np.ones((1, all_args.num_agents, 1), dtype=np.float32)
    
    print(f"Running: use_curriculum={all_args.use_curriculum}, curriculum_stage={curriculum_stage}, "
          f"n_agents={all_args.num_agents}, max_steps={all_args.episode_length}")
    
    for step in range(all_args.episode_length):
        # 记录当前时刻的传感器位置
        for i in range(all_args.num_agents):
            if i < len(base_env.agent_pos):
                sx, sy = base_env.agent_pos[i]
                sensor_paths_x[i].append(float(sx))
                sensor_paths_y[i].append(float(sy))
        
        # 记录当前时刻的目标位置 (trajectories 在 step 之前是上一帧状态，reset 后 step 0 时是初始)
        step_idx = base_env.step_count
        curr_targets = base_env.trajectories[step_idx] if step_idx < len(base_env.trajectories) else []
        for t_idx in range(base_env.n_targets):
            if t_idx < len(curr_targets):
                tgt = curr_targets[t_idx]
                target_paths_x[t_idx].append(float(tgt[0]))
                target_paths_y[t_idx].append(float(tgt[2]))
            elif target_paths_x[t_idx]:
                target_paths_x[t_idx].append(target_paths_x[t_idx][-1])
                target_paths_y[t_idx].append(target_paths_y[t_idx][-1])
            else:
                target_paths_x[t_idx].append(np.nan)
                target_paths_y[t_idx].append(np.nan)
        
        # 策略推理 (obs: [n_agents, obs_dim]，每行一个智能体的局部观测)
        with torch.no_grad():
            policy.actor.eval()
            obs_flat = np.array(obs)  # [n_agents, obs_dim]
            rnn_flat = np.concatenate(rnn_states)
            mask_flat = np.concatenate(masks)
            actions, rnn_states_new = policy.act(
                obs_flat, rnn_flat, mask_flat, deterministic=True
            )
            actions = np.squeeze(actions.cpu().numpy())
            if actions.ndim == 0:
                actions = np.array([actions])
        
        # 非循环策略下保持 rnn_states 为零即可
        if not getattr(all_args, "use_recurrent_policy", False):
            rnn_states = np.zeros((1, all_args.num_agents, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
        else:
            rnn_arr = rnn_states_new.cpu().numpy()
            rnn_states = rnn_arr.reshape(1, all_args.num_agents, all_args.recurrent_N, all_args.hidden_size)
        
        # 转换为环境动作格式 (Discrete -> one-hot)
        n_actions = base_env.n_actions
        actions_env = []
        for i in range(all_args.num_agents):
            a = actions[i] if i < len(actions) else actions[-1]
            idx = int(np.asarray(a).flat[0])
            idx = np.clip(idx, 0, n_actions - 1)
            actions_env.append(np.eye(n_actions)[idx].astype(np.float32))
        
        # 环境步进
        obs_list, rewards, dones, infos = env.step(actions_env)
        obs = np.array(obs_list)
        
        masks = np.ones((1, all_args.num_agents, 1), dtype=np.float32)
        if np.any(dones):
            rnn_states = np.zeros_like(rnn_states)
        if np.all(dones):
            break
    
    # 补充最后一步的传感器位置 (step 结束后 agent_pos 已更新)
    for i in range(all_args.num_agents):
        if i < len(base_env.agent_pos):
            sx, sy = base_env.agent_pos[i]
            if len(sensor_paths_x[i]) < base_env.step_count + 1:
                sensor_paths_x[i].append(float(sx))
                sensor_paths_y[i].append(float(sy))
    
    # ========== 绘图 ==========
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(base_env.x_min, base_env.x_max)
    ax.set_ylim(base_env.y_min, base_env.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Target & Sensor Trajectories (Run104)")
    ax.grid(alpha=0.3)
    
    # 传感器轨迹
    sensor_colors = ["#2563eb", "#7c3aed"]
    for i in range(all_args.num_agents):
        if sensor_paths_x[i]:
            ax.plot(
                sensor_paths_x[i], sensor_paths_y[i],
                color=sensor_colors[i % len(sensor_colors)],
                linewidth=2.0, label=f"Sensor {i}"
            )
            ax.scatter(
                [sensor_paths_x[i][0]], [sensor_paths_y[i][0]],
                color=sensor_colors[i % len(sensor_colors)], s=80, marker="o", zorder=5
            )
            ax.scatter(
                [sensor_paths_x[i][-1]], [sensor_paths_y[i][-1]],
                color=sensor_colors[i % len(sensor_colors)], s=80, marker="s", zorder=5
            )
            # 绘制传感器视域圆 (最后一帧)
            circle = patches.Circle(
                (sensor_paths_x[i][-1], sensor_paths_y[i][-1]),
                base_env.sensor_r, fill=False, linestyle="--",
                linewidth=1.2, color=sensor_colors[i % len(sensor_colors)], alpha=0.5
            )
            ax.add_patch(circle)
    
    # 目标轨迹
    target_colors = ["#ea580c", "#16a34a", "#ca8a04", "#0891b2", "#db2777"]
    for t_idx in range(base_env.n_targets):
        if target_paths_x[t_idx] and not np.all(np.isnan(target_paths_x[t_idx])):
            ax.plot(
                target_paths_x[t_idx], target_paths_y[t_idx],
                color=target_colors[t_idx % len(target_colors)],
                linewidth=2.0, linestyle="-", label=f"Target {t_idx}"
            )
            ax.scatter(
                [target_paths_x[t_idx][0]], [target_paths_y[t_idx][0]],
                color=target_colors[t_idx % len(target_colors)], s=80, marker="^", zorder=5
            )
            ax.scatter(
                [target_paths_x[t_idx][-1]], [target_paths_y[t_idx][-1]],
                color=target_colors[t_idx % len(target_colors)], s=80, marker="v", zorder=5
            )
    
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    
    # 保存与显示
    save_dir = os.path.join(all_args.model_dir, "..", "plots")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "validate_trajectories.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n[OK] Plot saved: {save_path}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
