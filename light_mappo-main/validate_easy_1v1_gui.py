import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from MY_ENV.envs.target_search import TargetSearchEnv


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _choose_chase_actions(env):
    """Simple policy for 1v1/2v2: each sensor chases nearest visible target."""
    step_idx = env.step_count
    actions = []
    if step_idx >= len(env.trajectories):
        return [int(env.n_actions // 2) for _ in range(env.n_agents)]

    curr_targets = env.trajectories[step_idx]
    if len(curr_targets) == 0:
        return [int(env.n_actions // 2) for _ in range(env.n_agents)]

    for i in range(env.n_agents):
        sx, sy = env.agent_pos[i]
        heading = float(env.agent_headings[i])

        nearest_t = min(
            curr_targets,
            key=lambda t: (float(t[0]) - sx) ** 2 + (float(t[2]) - sy) ** 2,
        )
        tx, ty = float(nearest_t[0]), float(nearest_t[2])
        desired_heading = np.arctan2(ty - sy, tx - sx)
        delta = _wrap_to_pi(desired_heading - heading)
        action_idx = int(np.argmin(np.abs(env.angle_adjustments - delta)))
        actions.append(action_idx)

    return actions


def main():
    # 修改这里可切换课程阶段: "1v1" 或 "2v2"
    mode = "2v2"
    is_2v2 = mode == "2v2"

    env = TargetSearchEnv(
        x_min=-600,
        x_max=600,
        y_min=-600,
        y_max=600,
        max_steps=120,
        easy_1v1=not is_2v2,
        easy_2v2=is_2v2,
    )
    env.enable_position_debug = False
    env.seed(7)
    env.reset()

    sensor_paths_x = [[] for _ in range(env.n_agents)]
    sensor_paths_y = [[] for _ in range(env.n_agents)]
    target_paths_x = [[] for _ in range(env.n_targets)]
    target_paths_y = [[] for _ in range(env.n_targets)]
    track_rewards_by_agent = [[] for _ in range(env.n_agents)]
    total_rewards_by_agent = [[] for _ in range(env.n_agents)]

    plt.ion()
    fig, (ax_map, ax_reward) = plt.subplots(1, 2, figsize=(13, 6))

    for _ in range(env.max_steps):
        actions = _choose_chase_actions(env)
        _, rewards, dones, infos = env.step(actions)

        for i in range(env.n_agents):
            sx, sy = env.agent_pos[i]
            sensor_paths_x[i].append(float(sx))
            sensor_paths_y[i].append(float(sy))

            reward_terms = infos[i].get("reward_terms", {})
            track_rewards_by_agent[i].append(float(reward_terms.get("track", 0.0)))
            total_rewards_by_agent[i].append(float(rewards[i][0]))

        step_idx = env.step_count - 1
        curr_targets = env.trajectories[step_idx] if 0 <= step_idx < len(env.trajectories) else []
        for t_idx in range(env.n_targets):
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

        ax_map.clear()
        ax_map.set_title(f"Easy {mode} Track Validation")
        ax_map.set_xlim(env.x_min, env.x_max)
        ax_map.set_ylim(env.y_min, env.y_max)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.grid(alpha=0.3)

        sensor_colors = ["tab:blue", "tab:purple"]
        target_colors = ["tab:orange", "tab:green"]
        for i in range(env.n_agents):
            sx, sy = env.agent_pos[i]
            color = sensor_colors[i % len(sensor_colors)]
            sensor_circle = patches.Circle(
                (sx, sy), env.sensor_r, fill=False, linestyle="--", linewidth=1.2, color=color, alpha=0.4
            )
            ax_map.add_patch(sensor_circle)
            ax_map.plot(sensor_paths_x[i], sensor_paths_y[i], color=color, linewidth=1.8, label=f"sensor {i} path")
            ax_map.scatter([sx], [sy], color=color, s=45)

        for t_idx in range(env.n_targets):
            color = target_colors[t_idx % len(target_colors)]
            ax_map.plot(target_paths_x[t_idx], target_paths_y[t_idx], color=color, linewidth=1.8, label=f"target {t_idx} path")
            if len(target_paths_x[t_idx]) > 0 and not np.isnan(target_paths_x[t_idx][-1]):
                ax_map.scatter([target_paths_x[t_idx][-1]], [target_paths_y[t_idx][-1]], color=color, s=45)
        ax_map.legend(loc="upper right")

        ax_reward.clear()
        ax_reward.set_title("Reward Curves")
        ax_reward.set_xlabel("Step")
        for i in range(env.n_agents):
            ax_reward.plot(track_rewards_by_agent[i], linewidth=1.6, label=f"agent {i} track")
            ax_reward.plot(total_rewards_by_agent[i], linestyle="--", alpha=0.8, label=f"agent {i} total")
        ax_reward.axhline(0.0, color="black", linestyle=":", linewidth=1)
        ax_reward.grid(alpha=0.3)
        ax_reward.legend(loc="lower right")

        curr_track_str = ", ".join(
            [f"a{i}:{track_rewards_by_agent[i][-1]:.3f}" for i in range(env.n_agents)]
        )
        fig.suptitle(
            f"step={env.step_count} | track [{curr_track_str}]",
            fontsize=11,
        )
        plt.pause(0.05)

        if all(dones):
            break

    plt.ioff()
    for i in range(env.n_agents):
        arr = np.array(track_rewards_by_agent[i], dtype=np.float32)
        mean_track = float(np.mean(arr)) if arr.size > 0 else 0.0
        positive_ratio = float(np.mean(arr > 0.0)) if arr.size > 0 else 0.0
        print(f"[Easy {mode}] agent {i} mean track reward: {mean_track:.4f}")
        print(f"[Easy {mode}] agent {i} positive track ratio: {positive_ratio:.2%}")
    plt.show()


if __name__ == "__main__":
    main()
