import numpy as np
import torch
import matplotlib.pyplot as plt
from phd_utils import PHDFeatureExtractor 

# ==========================================
class PHDFeatureExtractor:
    def __init__(self, config, device='cpu'):
        self.H = config.get('cnn_h', 64)
        self.W = config.get('cnn_w', 64)
        self.r_max = config.get('r_max', 800)
        self.max_speed = config.get('max_speed', 20.0)
        self.device = device
        
        dr = self.r_max / self.H
        r_vals = np.linspace(dr/2, self.r_max - dr/2, self.H)
        dtheta = (2 * np.pi) / self.W
        theta_vals = np.linspace(-np.pi + dtheta/2, np.pi - dtheta/2, self.W)
        
        self.R_grid, self.Theta_grid = np.meshgrid(r_vals, theta_vals, indexing='ij')
        self.Cell_Area = self.R_grid * dr * dtheta
        
    def process(self, phd_output, agent_pose):
        weights, means, covs, n_components = phd_output
        ax, ay, a_head_deg = agent_pose
        a_head_rad = np.radians(a_head_deg)
        
        feature_map = np.zeros((self.H, self.W, 3), dtype=np.float32)
        total_intensity_div = np.ones((self.H, self.W), dtype=np.float32) * 1e-9
        
        Theta_global = self.Theta_grid + a_head_rad
        Grid_X = ax + self.R_grid * np.cos(Theta_global)
        Grid_Y = ay + self.R_grid * np.sin(Theta_global)
        Grid_Pos = np.dstack((Grid_X, Grid_Y)) 

        if n_components > 0:
            for w, m, P in zip(weights, means, covs):
                m_pos = np.array([m[0], m[2]])
                m_vel = np.array([m[1], m[3]])
                P_pos = np.array([[P[0,0], P[0,2]], [P[2,0], P[2,2]]])
                
                diff = Grid_Pos - m_pos
                try:
                    inv_P = np.linalg.inv(P_pos)
                    det_P = np.linalg.det(P_pos)
                except: continue 

                mahalanobis = np.einsum('ijk,kl,ijl->ij', diff, inv_P, diff)
                pdf_val = (1.0 / (2 * np.pi * np.sqrt(det_P))) * np.exp(-0.5 * mahalanobis)
                
                local_mass = w * pdf_val * self.Cell_Area
                feature_map[:, :, 0] += local_mass
                feature_map[:, :, 1] += local_mass * m_vel[0]
                feature_map[:, :, 2] += local_mass * m_vel[1]
                total_intensity_div += local_mass

            avg_vx_global = feature_map[:, :, 1] / total_intensity_div
            avg_vy_global = feature_map[:, :, 2] / total_intensity_div
            
            Vr = avg_vx_global * np.cos(Theta_global) + avg_vy_global * np.sin(Theta_global)
            Vt = -avg_vx_global * np.sin(Theta_global) + avg_vy_global * np.cos(Theta_global)
            
            feature_map[:, :, 1] = Vr
            feature_map[:, :, 2] = Vt
        
        feature_map[:, :, 0] = np.log1p(feature_map[:, :, 0])
        feature_map[:, :, 1] = np.clip(feature_map[:, :, 1] / self.max_speed, -1.0, 1.0)
        feature_map[:, :, 2] = np.clip(feature_map[:, :, 2] / self.max_speed, -1.0, 1.0)

        tensor = torch.from_numpy(feature_map).float().permute(2, 0, 1)
        return tensor.to(self.device)
# ==========================================

def plot_tensor(tensor, title):
    """可视化 3通道 张量"""
    # Tensor (3, H, W) -> Numpy (H, W, 3)
    data = tensor.cpu().numpy().transpose(1, 2, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Channel 0: Occupancy (0~N)
    im0 = axes[0].imshow(data[:, :, 0], origin='lower', cmap='magma')
    axes[0].set_title('Ch0: Occupancy (Log Density)')
    axes[0].set_xlabel('Angle Index (Width)')
    axes[0].set_ylabel('Range Index (Height)')
    plt.colorbar(im0, ax=axes[0])
    
    # Channel 1: Radial Velocity (-1~1)
    im1 = axes[1].imshow(data[:, :, 1], origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Ch1: Radial Velocity\n(Red=Away, Blue=Close)')
    plt.colorbar(im1, ax=axes[1])
    
    # Channel 2: Tangential Velocity (-1~1)
    im2 = axes[2].imshow(data[:, :, 2], origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title('Ch2: Tangential Velocity\n(Red=Right/CW, Blue=Left/CCW)')
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def run_tests():
    # 1. 初始化配置
    config = {
        'cnn_h': 64, 'cnn_w': 64,
        'r_max': 800, 'max_speed': 20.0
    }
    extractor = PHDFeatureExtractor(config)
    
    print("Test 1: 单个目标，静止，位于正前方 (Distance=400m)")
    # 目标状态: x=400, vx=0, y=0, vy=0
    # 协方差: 标准差 20m
    w = [1.0]; m = [np.array([400, 0, 0, 0])]; P = [np.diag([400, 1, 400, 1])] 
    phd_out = [w, m, P, 1]
    agent_pose = [0, 0, 0] # 智能体在原点，朝东(0度)
    
    obs = extractor.process(phd_out, agent_pose)
    plot_tensor(obs, "Scenario 1: Static Target at 400m East")
    
    print("Test 2: 单个目标，正在快速远离 (Vx=20)")
    # 目标状态: x=400, vx=20, y=0, vy=0
    m = [np.array([400, 20, 0, 0])]
    phd_out = [w, m, P, 1]
    
    obs = extractor.process(phd_out, agent_pose)
    plot_tensor(obs, "Scenario 2: Moving Away (Radial+)")

    print("Test 3: 单个目标，正在向左横穿 (Vy=20)")
    # 目标状态: x=400, vx=0, y=0, vy=20
    # 相对于朝东的智能体，Vy=20 是向左跑
    m = [np.array([400, 0, 0, 20])]
    phd_out = [w, m, P, 1]
    
    obs = extractor.process(phd_out, agent_pose)
    plot_tensor(obs, "Scenario 3: Moving Left (Tangential+)")

    print("Test 4: 智能体旋转 90度 (朝北)")
    # 智能体现在朝北(90度)，目标还在原来的物理位置(400, 0)
    # 那么对于智能体来说，目标应该跑到“右侧”去了 (-90度方向)
    agent_pose_rotated = [0, 0, 90] 
    m = [np.array([400, 0, 0, 0])] # 目标位置不变
    phd_out = [w, m, P, 1]
    
    obs = extractor.process(phd_out, agent_pose_rotated)
    plot_tensor(obs, "Scenario 4: Agent Rotated 90 Deg (Target should shift)")

if __name__ == "__main__":
    run_tests()