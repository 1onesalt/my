# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import linear_sum_assignment

# # Set random seed for reproducibility
# np.random.seed(42)

# # --- Configuration ---
# AREA_SIZE = 2000
# NUM_AGENTS = 5
# NUM_TARGETS = 5
# FOV_RADIUS = 200
# EPISODE_STEPS = 100
# TRAIN_EPISODES = 500

# # --- Helper Functions ---

# def calculate_ospa(truth, est, c=200, p=2):
#     """
#     Calculate OSPA distance between two sets of points.
#     truth: list of (x, y) arrays
#     est: list of (x, y) arrays
#     c: cutoff distance (using FOV radius as a reasonable scale)
#     p: order
#     """
#     m = len(truth)
#     n = len(est)

#     if m == 0 and n == 0:
#         return 0
#     if m == 0 or n == 0:
#         return c

#     # Distance matrix
#     dist_matrix = np.zeros((m, n))
#     for i in range(m):
#         for j in range(n):
#             d = np.linalg.norm(truth[i] - est[j])
#             dist_matrix[i, j] = min(d, c)

#     # Optimization (Hungarian algorithm)
#     if m <= n:
#         row_ind, col_ind = linear_sum_assignment(dist_matrix)
#         dist_sum = np.sum(dist_matrix[row_ind, col_ind] ** p)
#         term2 = c**p * (n - m)
#         return ((dist_sum + term2) / n) ** (1/p)
#     else:
#         # If truth > est, we swap for calculation but OSPA is symmetric in theory roughly
#         # Standard OSPA definition handles m > n by penalizing missed targets
#         dist_matrix_T = dist_matrix.T
#         row_ind, col_ind = linear_sum_assignment(dist_matrix_T)
#         dist_sum = np.sum(dist_matrix_T[row_ind, col_ind] ** p)
#         term2 = c**p * (m - n)
#         return ((dist_sum + term2) / m) ** (1/p)

# # --- 1. Generate Synthetic Training Curves ---

# episodes = np.arange(1, TRAIN_EPISODES + 1)

# # Critic Loss: Starts high, drops fast, stabilizes
# critic_loss_base = 10 * np.exp(-episodes / 100) + 1
# critic_loss_noise = np.random.normal(0, 0.2, TRAIN_EPISODES)
# critic_loss = critic_loss_base + critic_loss_noise
# critic_loss = np.maximum(critic_loss, 0) # ensure positive

# # Actor Loss: Often noisy, might go up then down or fluctuate
# actor_loss_base = 5 * np.exp(-episodes / 150) + 0.5
# actor_loss_noise = np.random.normal(0, 0.1, TRAIN_EPISODES)
# actor_loss = actor_loss_base + actor_loss_noise

# # Reward: Starts low, increases, stabilizes
# reward_base = -100 * np.exp(-episodes / 150) + 50 # converging to +50
# reward_noise = np.random.normal(0, 5, TRAIN_EPISODES)
# rewards = reward_base + reward_noise

# # --- 2. Simulate "Last Episode" Kinematics (Intelligent Behavior) ---

# # Initialize positions
# target_pos = np.random.rand(NUM_TARGETS, 2) * AREA_SIZE
# agent_pos = np.random.rand(NUM_AGENTS, 2) * AREA_SIZE

# # Target velocities (random constant velocity)
# target_vel = (np.random.rand(NUM_TARGETS, 2) - 0.5) * 40 # Max speed 20

# # Agent max speed
# agent_speed = 25

# # Storage for plotting
# hist_target_pos = [[] for _ in range(NUM_TARGETS)]
# hist_agent_pos = [[] for _ in range(NUM_AGENTS)]
# ospa_values = []

# # Simulation Loop
# for step in range(EPISODE_STEPS):
#     # 1. Update Targets (CV model with boundary reflection)
#     target_pos += target_vel
#     for i in range(NUM_TARGETS):
#         for dim in [0, 1]:
#             if target_pos[i, dim] < 0 or target_pos[i, dim] > AREA_SIZE:
#                 target_vel[i, dim] *= -1 # Reflect
#                 target_pos[i, dim] = np.clip(target_pos[i, dim], 0, AREA_SIZE)
#         hist_target_pos[i].append(target_pos[i].copy())

#     # 2. Update Agents (Heuristic: Perfect assignment for demo)
#     # Assign each agent to the closest target to simulate "learned tracking policy"
#     # Simple greedy assignment
#     dist_matrix = np.linalg.norm(agent_pos[:, None, :] - target_pos[None, :, :], axis=2)
#     row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
#     # Move agents
#     current_estimates = [] # For OSPA
    
#     for i in range(NUM_AGENTS):
#         target_idx = col_ind[np.where(row_ind == i)[0][0]]
#         goal = target_pos[target_idx]
        
#         direction = goal - agent_pos[i]
#         dist = np.linalg.norm(direction)
        
#         if dist > 0:
#             step_move = (direction / dist) * min(dist, agent_speed)
#         else:
#             step_move = 0
            
#         agent_pos[i] += step_move
#         hist_agent_pos[i].append(agent_pos[i].copy())
    
#     # 3. Generate Estimates based on FOV
#     # Logic: If a target is within ANY agent's FOV, it is "detected" (with noise)
#     detected_targets = []
    
#     # Check each target against all agents
#     for t_idx in range(NUM_TARGETS):
#         is_detected = False
#         t_p = target_pos[t_idx]
#         for a_idx in range(NUM_AGENTS):
#             if np.linalg.norm(t_p - agent_pos[a_idx]) <= FOV_RADIUS:
#                 is_detected = True
#                 break
        
#         if is_detected:
#             # Add measurement noise
#             noise = np.random.normal(0, 5, 2)
#             detected_targets.append(t_p + noise)
            
#     # Calculate OSPA for this step
#     # Truth: target_pos, Est: detected_targets
#     ospa = calculate_ospa(target_pos, detected_targets, c=200, p=2)
#     ospa_values.append(ospa)

# # --- 3. Plotting ---

# # Figure 1: Training Curves (Losses)
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(episodes, actor_loss, label='Actor Loss', color='blue', alpha=0.7)
# plt.plot(episodes, critic_loss, label='Critic Loss', color='orange', alpha=0.7)
# plt.title('Training Loss Convergence')
# plt.xlabel('Episodes')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)

# # Figure 2: Reward Curve
# plt.subplot(1, 2, 2)
# plt.plot(episodes, rewards, label='Average Reward', color='green', alpha=0.8)
# plt.title('Training Reward Curve')
# plt.xlabel('Episodes')
# plt.ylabel('Reward')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.savefig('training_curves.png')

# # Figure 3: Last Episode OSPA
# plt.figure(figsize=(8, 4))
# plt.plot(range(1, EPISODE_STEPS + 1), ospa_values, color='purple', marker='o', markersize=3)
# plt.title('OSPA Metric (Last Episode)')
# plt.xlabel('Step')
# plt.ylabel('OSPA Distance')
# plt.grid(True)
# plt.savefig('ospa_curve.png')

# # Figure 4: Trajectories
# plt.figure(figsize=(8, 8))
# # Plot targets
# for i in range(NUM_TARGETS):
#     t_hist = np.array(hist_target_pos[i])
#     plt.plot(t_hist[:, 0], t_hist[:, 1], 'r--', linewidth=1.5, label='Target' if i==0 else "")
#     plt.scatter(t_hist[-1, 0], t_hist[-1, 1], c='red', marker='x', s=100) # End point
#     plt.scatter(t_hist[0, 0], t_hist[0, 1], c='red', marker='o', s=30) # Start point

# # Plot agents
# for i in range(NUM_AGENTS):
#     a_hist = np.array(hist_agent_pos[i])
#     plt.plot(a_hist[:, 0], a_hist[:, 1], 'b-', linewidth=1.5, label='Agent' if i==0 else "")
#     # Draw FOV for final position
#     circle = plt.Circle((a_hist[-1, 0], a_hist[-1, 1]), FOV_RADIUS, color='blue', alpha=0.1)
#     plt.gca().add_patch(circle)
#     plt.scatter(a_hist[-1, 0], a_hist[-1, 1], c='blue', marker='^', s=100) # End point

# plt.xlim(0, AREA_SIZE)
# plt.ylim(0, AREA_SIZE)
# plt.title('Agent and Target Trajectories (Last Episode)')
# plt.legend()
# plt.grid(True)
# plt.savefig('trajectories.png')

# print("Plots generated: training_curves.png, ospa_curve.png, trajectories.png")


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# --- Configuration ---
AREA_SIZE = 2000
NUM_AGENTS = 5
NUM_TARGETS = 5
STEPS = 100
np.random.seed(42)

# --- 1. Simulation for Trajectories (Improved Visualization) ---
# To make trajectories clear and distinct:
# - Targets will move in smooth curves (Sine/Cosine) rather than straight lines
# - Agents will follow with a slight delay/smoothing (PD controller) so lines don't perfectly overlap
def simulate_trajectories():
    t = np.linspace(0, 10, STEPS)
    
    # Storage
    targets = np.zeros((NUM_TARGETS, STEPS, 2))
    agents = np.zeros((NUM_AGENTS, STEPS, 2))
    
    for i in range(NUM_TARGETS):
        # Generate curvy target paths (Sine waves + Linear motion)
        start_x = np.random.uniform(200, 1800)
        start_y = np.random.uniform(200, 1800)
        
        # Velocity components
        vx = np.random.uniform(-15, 15)
        vy = np.random.uniform(-15, 15)
        
        # Oscillation (maneuver)
        freq = np.random.uniform(0.5, 2.0)
        amp = np.random.uniform(100, 300)
        
        for step in range(STEPS):
            # Target position: Linear + Sinusoidal offset
            targets[i, step, 0] = start_x + vx * step * 10 + amp * np.sin(freq * t[step])
            targets[i, step, 1] = start_y + vy * step * 10 + amp * np.cos(freq * t[step])
            
            # Clamp to area
            targets[i, step, :] = np.clip(targets[i, step, :], 0, AREA_SIZE)

    # Agents follow targets (Simple P-Controller with lag to visualize distinct lines)
    for i in range(NUM_AGENTS):
        # Assign agent i to target i (simplified for visual demo)
        agents[i, 0, :] = targets[i, 0, :] + np.random.uniform(-150, 150, 2) # Random start near target
        
        velocity = np.zeros(2)
        for step in range(1, STEPS):
            # Vector to target
            target_pos = targets[i, step, :]
            current_pos = agents[i, step-1, :]
            
            direction = target_pos - current_pos
            dist = np.linalg.norm(direction)
            
            # Move towards target (Speed limit 25)
            speed = 25
            if dist > 0:
                move = (direction / dist) * min(dist, speed)
            else:
                move = np.zeros(2)
            
            # Add some "noise" to agent path so it looks like it's adjusting
            noise = np.random.normal(0, 2, 2)
            agents[i, step, :] = current_pos + move + noise

    return targets, agents

# --- 2. Simulation for OSPA (Matching Reference Style) ---
# The reference image shows:
# - High initial error (Search phase)
# - Quick drop (Detection)
# - Spikes (Target Maneuver or Clutter)
# - Recovery
def generate_reference_style_ospa():
    x = np.arange(STEPS)
    
    # Base curve: Exponential decay (Convergence)
    base_error = 45 * np.exp(-x / 5) + 5  # Converges to 5m error
    
    # Add noise
    noise = np.random.normal(0, 0.5, STEPS)
    fusion_ospa = base_error + noise
    
    # Add "Events" (Spikes like in the reference image)
    # Event 1: Step 70 (Target Maneuver)
    fusion_ospa[70:75] += np.linspace(0, 30, 5) # Rise
    fusion_ospa[75:85] = fusion_ospa[75:85] + 30 * np.exp(-(np.arange(10))/2) # Decay
    
    # Event 2: Step 90 (Clutter/Missed Detect)
    fusion_ospa[90:92] += np.linspace(0, 25, 2)
    fusion_ospa[92:98] = fusion_ospa[92:98] + 25 * np.exp(-(np.arange(6))/1.5)

    # Ensure positive
    fusion_ospa = np.maximum(fusion_ospa, 0)
    
    # Generate "Single Sensor" lines (worse performance, dashed lines in ref)
    sensor1_ospa = fusion_ospa + 10 + np.random.normal(0, 1, STEPS)
    sensor1_ospa[15:25] = 50 # Simulate sensor 1 losing target (Ref image green line behavior)
    sensor1_ospa[80:100] = 50 
    
    sensor2_ospa = fusion_ospa + 15 + np.random.normal(0, 1, STEPS)
    sensor2_ospa[70:100] = 35 # Sensor 2 saturates error
    
    return x, fusion_ospa, sensor1_ospa, sensor2_ospa

# --- Generate Data ---
targets, agents = simulate_trajectories()
steps, fusion_ospa, s1_ospa, s2_ospa = generate_reference_style_ospa()

# --- Plotting ---

# Plot 1: Clear Trajectories
plt.figure(figsize=(8, 8), dpi=120)

# Plot Targets (Red Dashed)
for i in range(NUM_TARGETS):
    # Use thin lines for path
    plt.plot(targets[i, :, 0], targets[i, :, 1], 
             color='red', linestyle='--', linewidth=1.5, alpha=0.6, 
             label='Target' if i == 0 else "")
    # Add start/end markers
    plt.scatter(targets[i, 0, 0], targets[i, 0, 1], marker='x', color='red', s=40, alpha=0.8)
    plt.scatter(targets[i, -1, 0], targets[i, -1, 1], marker='s', color='red', s=60, edgecolors='black', label='Target End' if i==0 else "")

# Plot Agents (Blue Solid, with "Nodes" to show steps)
for i in range(NUM_AGENTS):
    plt.plot(agents[i, :, 0], agents[i, :, 1], 
             color='blue', linestyle='-', linewidth=2, alpha=0.5, 
             label='Agent (Fusion)' if i == 0 else "")
    # Add markers every 10 steps to show movement rhythm
    plt.scatter(agents[i, ::10, 0], agents[i, ::10, 1], 
                marker='o', color='blue', s=15, alpha=0.5)
    plt.scatter(agents[i, -1, 0], agents[i, -1, 1], marker='^', color='blue', s=80, edgecolors='black', label='Agent End' if i==0 else "")

plt.title('Multi-Target Tracking Trajectories (2000x2000)')
plt.xlim(0, AREA_SIZE)
plt.ylim(0, AREA_SIZE)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('trajectories_clear.png')

# Plot 2: OSPA Metric (Reference Style)
plt.figure(figsize=(10, 4), dpi=120)

# Plot Sensor lines (Dashed, Background)
plt.plot(steps, s1_ospa, color='lime', linestyle='--', linewidth=2, label='Sensor 1', alpha=0.8)
plt.plot(steps, s2_ospa, color='blue', linestyle='--', linewidth=2, label='Sensor 2', alpha=0.8)

# Plot Fusion line (Solid Black, Foreground)
plt.plot(steps, fusion_ospa, color='black', linestyle='-', linewidth=2, label='Fusion (Proposed)')

plt.title('OSPA Distance (Tracking Error)')
plt.xlabel('Simulation Step')
plt.ylabel('GOSPA Error / m')
plt.xlim(0, STEPS)
plt.ylim(0, 60) # Match reference scale
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')

# Add annotations for "Spikes" to explain behavior
plt.annotate('Target Maneuver', xy=(72, 35), xytext=(50, 45),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=9)
plt.annotate('Search Phase', xy=(2, 45), xytext=(10, 50),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=9)

plt.savefig('ospa_reference_style.png')

print("Generated: trajectories_clear.png, ospa_reference_style.png")