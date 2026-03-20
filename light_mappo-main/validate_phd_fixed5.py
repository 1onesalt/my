import copy
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.optimize import linear_sum_assignment

from MY_ENV.envs.PHD import PHD, State_extraction, generate
from MY_ENV.envs.phd_fusion import AGMFusionCenter
from MY_ENV.envs.test_model import model, observe_Fov, polar2dicaer


class FixedSensorAgent:
    """Single fixed sensor + local PHD filter wrapper."""

    def __init__(self, sensor_id, x, y, base_params):
        self.id = sensor_id
        self.params = copy.deepcopy(base_params)
        self.params["x_agent"] = float(x)
        self.params["y_agent"] = float(y)

        self.phd = PHD(self.params)
        self.state = [[], [], [], 0]
        self.z_cart_history = []

    def step_filter(self, gt_targets_this_step):
        z_polar = observe_Fov(self.params, gt_targets_this_step)
        z_cart = polar2dicaer(z_polar, self.params) if len(z_polar) > 0 else []
        self.z_cart_history.append(z_cart)

        # Need >= 3 frames to form two-frame difference robustly.
        if len(self.z_cart_history) < 3:
            return self.state

        z_last = self.z_cart_history[-2]
        z_now = self.z_cart_history[-1]
        self.phd.init_params(z_last, z_now, z_polar, self.state)
        self.phd.w_new, self.phd.m_new, self.phd.P_new, self.phd.J_new = generate(
            self.phd.z_lastD,
            len(self.phd.z_lastD),
            self.phd.z_nowD,
            len(self.phd.z_nowD),
            self.params["Vx_thre"],
            self.params["Vy_thre"],
        )
        return self.phd.predict_update()


def calculate_ospa(est_states, true_states, p=2, c=200.0):
    """OSPA distance on target positions."""
    est_pos = [np.array([s[0], s[2]], dtype=np.float32) for s in est_states]
    true_pos = [np.array([s[0], s[2]], dtype=np.float32) for s in true_states]

    m = len(est_pos)
    n = len(true_pos)
    if m == 0 and n == 0:
        return 0.0
    if n == 0:
        return float(c)
    if m > n:
        return calculate_ospa(true_states, est_states, p=p, c=c)

    cost = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            d = float(np.linalg.norm(est_pos[i] - true_pos[j]))
            cost[i, j] = min(d, c) ** p

    matched_cost = 0.0
    if m > 0:
        ridx, cidx = linear_sum_assignment(cost)
        matched_cost = float(cost[ridx, cidx].sum())
    term = matched_cost + (c ** p) * (n - m)
    return float((term / n) ** (1.0 / p))


def build_targets(num_steps=100, dt=1.0, n_targets=10, seed=7):
    """Create 10 targets with constant-velocity straight motion."""
    rng = np.random.default_rng(seed)
    states = np.zeros((num_steps, n_targets, 4), dtype=np.float32)

    # Keep initial states near center and limit speed so all tracks stay
    # inside the 500 m sensing radius over 100 s.
    pos = rng.uniform(-180.0, 180.0, size=(n_targets, 2)).astype(np.float32)
    vel = rng.uniform(-1.2, 1.2, size=(n_targets, 2)).astype(np.float32)

    for t in range(num_steps):
        states[t, :, 0] = pos[:, 0]
        states[t, :, 1] = vel[:, 0]
        states[t, :, 2] = pos[:, 1]
        states[t, :, 3] = vel[:, 1]

        pos = pos + vel * dt
    return states


def main():
    # ===== Experiment setup =====
    num_steps = 100
    dt = 1.0
    n_targets = 10
    sensor_range = 500.0
    seed = 7

    # Fixed sensors (all static): center + four cardinal points
    sensor_positions = [
        (0.0, 0.0),
        (300.0, 0.0),
        (-300.0, 0.0),
        (0.0, 300.0),
        (0.0, -300.0),
    ]

    # ===== Parameters =====
    base_params = model()
    base_params["obverser_d"] = sensor_range
    base_params["Pd"] = 0.98
    base_params["Zr"] = 2
    base_params["Vx_thre"] = 50.0
    base_params["Vy_thre"] = 50.0
    base_params["num_scans"] = num_steps
    base_params["Ps"] = 0.99
    base_params["phd_prune_threshold"] = 0.02
    base_params["fusion_unmatched_scale"] = 0.4
    base_params["fusion_t_merge"] = 50.0

    # ===== Build targets (always inside center FoV) =====
    target_states = build_targets(num_steps=num_steps, dt=dt, n_targets=n_targets, seed=seed)

    # ===== Init sensors + fusion =====
    agents = [
        FixedSensorAgent(i, pos[0], pos[1], base_params)
        for i, pos in enumerate(sensor_positions)
    ]
    fusion_center = AGMFusionCenter(base_params)
    sensor_configs = [
        {"x": float(x), "y": float(y), "range": float(sensor_range)}
        for x, y in sensor_positions
    ]

    # ===== Logging =====
    ospa_values = []
    est_cardinality = []
    true_cardinality = []

    target_tracks_x = [[target_states[t, j, 0] for t in range(num_steps)] for j in range(n_targets)]
    target_tracks_y = [[target_states[t, j, 2] for t in range(num_steps)] for j in range(n_targets)]

    # ===== Main simulation loop =====
    for t in range(num_steps):
        gt_list = [target_states[t, j, :] for j in range(n_targets)]

        local_estimates = []
        for agent in agents:
            local_estimates.append(agent.step_filter(gt_list))

        global_state = fusion_center.run(local_estimates, sensor_configs)
        feedback_list = fusion_center.distribute_to_sensors(global_state, sensor_configs)
        for i, agent in enumerate(agents):
            fb = feedback_list[i]
            agent.state = [fb[0], fb[1], fb[2], len(fb[0])]

        global_estimates, n_est = State_extraction(global_state)
        ospa_values.append(calculate_ospa(global_estimates, gt_list, c=120.0, p=1))
        est_cardinality.append(int(n_est))
        true_cardinality.append(len(gt_list))

    # ===== Plot trajectories =====
    fig1, ax1 = plt.subplots(figsize=(9, 9))
    ax1.set_title("Fixed 5-Sensor + 10-Target Trajectories")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(-650, 650)
    ax1.set_ylim(-650, 650)
    ax1.grid(alpha=0.3)

    for i, (sx, sy) in enumerate(sensor_positions):
        ax1.scatter([sx], [sy], c="tab:blue", s=50, zorder=5)
        ax1.text(sx + 10, sy + 10, f"S{i}", color="tab:blue", fontsize=10)
        circle = patches.Circle((sx, sy), sensor_range, fill=False, linestyle="--", linewidth=1.2, alpha=0.35)
        ax1.add_patch(circle)

    for j in range(n_targets):
        ax1.plot(target_tracks_x[j], target_tracks_y[j], linewidth=1.6, alpha=0.9)
        ax1.scatter([target_tracks_x[j][0]], [target_tracks_y[j][0]], marker="^", s=20)
        ax1.scatter([target_tracks_x[j][-1]], [target_tracks_y[j][-1]], marker="v", s=20)

    # ===== Plot OSPA =====
    fig2, ax2 = plt.subplots(figsize=(10, 4.6))
    time_axis = np.arange(1, num_steps + 1)
    ax2.plot(time_axis, ospa_values, color="tab:red", linewidth=2.0, label="OSPA (c=120, p=1)")
    ax2.set_title("OSPA Indicator over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("OSPA (m)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")

    # Optional: show cardinality consistency
    ax2_t = ax2.twinx()
    ax2_t.plot(time_axis, est_cardinality, color="tab:green", linewidth=1.2, alpha=0.65, label="Estimated #")
    ax2_t.plot(time_axis, true_cardinality, color="tab:gray", linewidth=1.2, alpha=0.65, linestyle="--", label="True #")
    ax2_t.set_ylabel("Cardinality")

    # ===== Save =====
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "phd_fixed5")
    os.makedirs(out_dir, exist_ok=True)
    traj_path = os.path.join(out_dir, "fixed5_target_trajectories.png")
    ospa_path = os.path.join(out_dir, "fixed5_ospa_curve.png")
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(traj_path, dpi=160)
    fig2.savefig(ospa_path, dpi=160)
    print(f"[OK] Trajectory plot saved: {traj_path}")
    print(f"[OK] OSPA plot saved: {ospa_path}")
    print(f"[INFO] Mean OSPA over {num_steps}s: {float(np.mean(ospa_values)):.3f} m")

    plt.show()


if __name__ == "__main__":
    main()
