import numpy as np
import random
from scipy.linalg import sqrtm


def generate(z_lastD, nums_z_lastD, z_nowD, nums_z_nowD, Vx_thre, Vy_thre):
    w_new = []
    m_new = []
    P_new = []
    Vx_max = 6
    Vy_max = 6

    # 先将 list[array([x, y])] 转换为 2×N 数组
    z_lastD = np.array(z_lastD).T  # shape: (2, nums_z_lastD)
    z_nowD = np.array(z_nowD).T    # shape: (2, nums_z_nowD)

    for i in range(nums_z_nowD):
        for j in range(nums_z_lastD):
            z_delta = z_nowD[:,i] - z_lastD[:,j]  #得到的是一个目标的位置变化量 [delta_x, delta_y]
            if abs(z_delta[0]) < Vx_thre and abs(z_delta[1]) < Vy_thre:
                # x轴速度赋值
                vx = Vx_max if abs(z_delta[0]) > Vx_max and z_delta[0] >= 0 else \
                     -Vx_max if abs(z_delta[0]) > Vx_max and z_delta[0] < 0 else z_delta[0]
                # y轴速度赋值
                vy = Vy_max if abs(z_delta[1]) > Vy_max and z_delta[1] >= 0 else \
                     -Vy_max if abs(z_delta[1]) > Vy_max and z_delta[1] < 0 else z_delta[1]
                m = [z_nowD[0, i], vx, z_nowD[1, i], vy]
                m_new.append(m)
                w_new.append(0.01)
                P_new.append(np.diag([100, 400, 100, 400]))

    # 转为numpy数组
    w_new = np.array(w_new)
    m_new = np.array(m_new)
    P_new = np.array(P_new)
    J_new = len(w_new)
    return w_new, m_new, P_new, J_new

def UKFpart(X_fusion_pre, P_fusion_pre, R, x_radar, y_radar):
    n_num = X_fusion_pre.shape[0]      #粒子的维度
    Z_ob_diffusion = np.zeros((2 * n_num + 1 , 2))
    w_m =  np.zeros((2 * n_num + 1 , 1))
    w_p = np.zeros((2 * n_num + 1 , 1))

    alpha = 0.1
    beta = 2
    kap = 0
    lam = (alpha ** 2) * (n_num + kap) - n_num

    X_pre_diffusion = np.zeros((2 * n_num + 1 , 4))
    X_pre_diffusion[0] = X_fusion_pre

    try:
        # 尝试进行 Cholesky 分解
        L = np.linalg.cholesky(P_fusion_pre)
    except np.linalg.LinAlgError:
        # 如果 Cholesky 分解失败，进行奇异值分解 (SVD)
        U, S, Vt = np.linalg.svd(P_fusion_pre)
        H = Vt.T @ np.diag(S) @ Vt
        P_fusion_pre = (P_fusion_pre + P_fusion_pre.T + H + H.T) / 4
    
    degree_diffusion = np.real(sqrtm((n_num + lam) * P_fusion_pre)).T
    
    for i in range(n_num):
        X_pre_diffusion[i + 1] = X_fusion_pre + degree_diffusion[i]
        X_pre_diffusion[i + 1 + n_num] = X_fusion_pre - degree_diffusion[i]

    w_m[0] = lam / (n_num + lam)
    w_p[0] = lam / (n_num + lam) + (1 - alpha ** 2 + beta)

    for i in range(2 * n_num):
        w_m[i + 1] = 1 / (2 * (n_num + lam))
        w_p[i + 1] = 1 / (2 * (n_num + lam))
              
    #计算撒点的观测值
    for i in range(2 * n_num + 1):    
        Z_ob_diffusion[i, 0] = np.sqrt((X_pre_diffusion[i, 0] - x_radar) ** 2 + (X_pre_diffusion[i, 2] - y_radar) ** 2)  #距离
        theta_sus_head = np.rad2deg(np.arctan((X_pre_diffusion[i, 2] - y_radar) / (X_pre_diffusion[i, 0] - x_radar)))

        if np.logical_and(X_pre_diffusion[i, 0] - x_radar >= 0, X_pre_diffusion[i, 2] - y_radar >= 0):
            Z_ob_diffusion[i, 1] = theta_sus_head
        elif np.logical_and(X_pre_diffusion[i, 0] - x_radar < 0, X_pre_diffusion[i, 2] - y_radar >= 0):
            Z_ob_diffusion[i, 1] = theta_sus_head + 180
        elif np.logical_and(X_pre_diffusion[i, 0] - x_radar < 0, X_pre_diffusion[i, 2] - y_radar < 0):
            Z_ob_diffusion[i, 1] = theta_sus_head + 180
        elif np.logical_and(X_pre_diffusion[i, 0] - x_radar >= 0, X_pre_diffusion[i, 2] - y_radar < 0):
            Z_ob_diffusion[i, 1] = theta_sus_head + 360

    flag_jump = 0
    max_theta = 0
    for i in range(1 , 2 * n_num + 1):
        d_theta = abs(Z_ob_diffusion[0, 1] - Z_ob_diffusion[i, 1])
        if d_theta > max_theta:
            max_theta = d_theta
    
    if max_theta > 180:
        for i in range(2 * n_num + 1):
            if Z_ob_diffusion[i, 1] > 270:
                Z_ob_diffusion[i, 1] = Z_ob_diffusion[i, 1] - 360
        flag_jump = 1

    Z_fusion_ob = np.zeros((1, 2))

    for i in range(2 * n_num + 1):
        Z_fusion_ob = Z_fusion_ob + w_m[i] * Z_ob_diffusion[i]
    
    P_fusion_ob = R
    for i in range(2 * n_num + 1):       
        P_fusion_ob = P_fusion_ob + w_p[i] * (Z_ob_diffusion[i] - Z_fusion_ob).T @ (Z_ob_diffusion[i] - Z_fusion_ob) #2×2
    
    Pzx = np.zeros((4, 2))
    for i in range(2 * n_num + 1):
        #X_fusion_pre状态向量的预测，X_pre_diffusion粒子散布
        Pzx += w_p[i] * (X_fusion_pre - X_pre_diffusion[i]).reshape(4, 1) @ (Z_fusion_ob - Z_ob_diffusion[i])#

    k_ukf = Pzx @ (np.linalg.inv(P_fusion_ob))

    Pnew = P_fusion_pre - k_ukf @ P_fusion_ob @ k_ukf.T

    return Z_fusion_ob, P_fusion_ob, k_ukf, Pnew, flag_jump



def M_UKF(X_fusion_pre, Z_polar, Z_fusion_ob, k_ukf, flag_jump):
    """
    参数说明：
    - X_fusion_pre: 状态预测值 (e.g., 4维状态向量)
    - Z_polar: 实际量测 (2维: [r, θ])
    - Z_fusion_ob: 预测量测 (2维: [r, θ])
    - k_ukf: 卡尔曼增益矩阵 (形状 4x2)
    - flag_jump: 跳变标记 (0或1)
    返回：
    - X_ukf: 更新后的状态
    """
   
    complement = np.array([0.0, 360.0]).reshape(2, 1)
    Z_polar = Z_polar.reshape(2, 1)
    Z_fusion_ob = Z_fusion_ob.reshape(2, 1)
    angle_obs = Z_polar[1, 0]    # 第2行第1列
    angle_pred = Z_fusion_ob[1, 0]
    delta_angle = Z_polar[1, 0] - Z_fusion_ob[1, 0]

    if abs(delta_angle) > 180 and flag_jump == 0:
        if delta_angle < 0:
            X_ukf = X_fusion_pre.reshape(4, 1) + k_ukf @ (Z_polar - Z_fusion_ob + complement)
        else:
            X_ukf = X_fusion_pre.reshape(4, 1) + k_ukf @ (Z_polar - (Z_fusion_ob + complement))
    elif abs(delta_angle) > 180 and flag_jump == 1:
        X_ukf = X_fusion_pre.reshape(4, 1) + k_ukf @ (Z_polar - (Z_fusion_ob + complement))
    else:
        X_ukf = X_fusion_pre.reshape(4, 1) + k_ukf @ (Z_polar - Z_fusion_ob).reshape(2, 1)

    X_ukf = X_ukf.flatten()
    #print(X_ukf)
    return X_ukf

def State_extraction(X_now):
    state_draw_list = []
    
    weights = X_now[0]  # 权重列表
    means = X_now[1]  # 均值列表


    for w, m in zip(weights, means):
        j = min(round(w), 2)  # 限制最多复制2次
        for _ in range(j):
            state_draw_list.append(m)

    state_draw = np.array(state_draw_list)
    num_draw = len(state_draw)

    return state_draw, num_draw


class PHD():
    def __init__(self, params):
        """
        初始化滤波器参数
        - model_params: z_lastD笛卡尔坐标系下上时刻的观测
                        z_nowD笛卡尔坐标系下当前时刻的观测   
                        z_nowP极坐标系下当前时刻的观测 
                        X_laxt上一时刻的全局后验分布
        """
        self.T = 1
        self.Ps = 1
        self.Pd = params["Pd"]           #检测概率
        self.R = params["obverser_R"]    #观测误差矩阵
        self.Vx_thre = 8
        self.Vy_thre = 8
        self.x_agent = params['x_agent']
        self.y_agent = params['y_agent']
        self.Zr = params["Zr"]           #杂波强度
        self.A = np.array([[1, self.T, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, self.T],
                            [0, 0, 0, 1]])
        self.Q = np.diag([1, 0.01, 1, 0.01])   
    
    def init_params(self, z_lastD, z_nowD, z_nowP, X_last):
        self.z_lastD = z_lastD
        self.nums_z_lastD = len(self.z_lastD)
        self.z_nowD = z_nowD
        self.nums_z_nowD = len(self.z_nowD)
        self.z_nowP = z_nowP
        self.X_last = X_last    

        self.w_priori = self.X_last[0]  # 上一时刻权重
        self.m_priori = self.X_last[1]  # 均值
        self.P_priori = self.X_last[2]  # 协方差
        self.J_priori = self.X_last[3]  # 先验粒子数量


    def predict_update(self):
        total = self.J_new + self.J_priori

        # 初始化
        w_pre = []
        m_pre = []
        P_pre = []

        w_pos = []
        m_pos = []
        P_pos = []

        P_z = []            #
        K_ukf = []          #
        P_ukf = []          #

        # 新生分量批量赋值
        for i in range(self.J_new):  #0 ~ J_new - 1
            w_pre.append(self.w_new[i])
            m_pre.append(np.array(self.m_new[i]) @ self.A.T)
            P_pre.append(self.A @ np.array(self.P_new[i]) @ self.A.T + self.Q)

        # 先验分量批量赋值
        for i in range(self.J_priori):   #J_new ~ total - 1
            w_pre.append(self.Ps * self.w_priori[i])
            m_pre.append(np.array(self.m_priori[i]) @ self.A.T)
            P_pre.append(self.Q + self.A @ np.array(self.P_priori[i]) @ self.A.T)

        J_pre = total       #预测的粒子数量
        if J_pre == 0:
            return [[], [], [], 0]
        
        else:
            Z_obpre = []
            P_z = []
            K_ukf = []
            P_ukf = []

            for i in range(J_pre):#未被检测到的部分
                w_pos.append((1 - self.Pd) * w_pre[i])
                m_pos.append(m_pre[i])
                P_pos.append(P_pre[i])

            flag_jump = np.zeros((J_pre, 1))
            for i in range(J_pre):                                  
                Z_obpre_tmp, P_z_tmp, K_ukf_tmp, P_ukf_tmp, flag_jump_tmp = UKFpart(m_pre[i], P_pre[i], self.R, self.x_agent, self.y_agent)
                Z_obpre.append(Z_obpre_tmp)
                P_z.append(P_z_tmp)
                K_ukf.append(K_ukf_tmp)
                P_ukf.append(P_ukf_tmp)
                flag_jump[i] = flag_jump_tmp


            # #量测更新
            e = 0
            #self.z_nowP = np.array(self.z_nowP)
            n_znow = len(self.z_nowP)


            for i in range(n_znow):    #观测的数量
                e = e + 1
                for j in range(J_pre):  #预测的粒子数量
                    Z_pred = Z_obpre[j]            #观测预测状态  
                    Pz_j = P_z[j]                  #观测之后的协方差
                    
                    # 实际极坐标观测
                    z_now = self.z_nowP[i]              

                    # 概率密度计算
                    diff = (z_now - Z_pred).reshape(-1)             

                    exponent = -0.5 * diff @ np.linalg.inv(Pz_j) @ diff
                    denom = 2 * np.pi * np.sqrt(np.linalg.det(Pz_j))
                    likelihood = np.exp(exponent) / denom

                    # 权重更新
                    w_val = w_pre[j] * likelihood
                    w_pos.append(w_val)

                    # 协方差和均值更新
                    P_pos.append(P_ukf[j])  
                    m_pos.append(M_UKF(m_pre[j], z_now, Z_pred, K_ukf[j], flag_jump[j])) 

                # 当前观测关联的目标更新权重归一化
                updated_weights = w_pos[e*J_pre - J_pre:e*J_pre]
                if max(updated_weights) <= 1e-8:
                    w_pos[e*J_pre - J_pre:e*J_pre] = [0] * J_pre
                else:
                    w_sum = sum(updated_weights)
        
                    updated_weights = [self.Pd * w / ((self.Zr / (900**2)) + w_sum) for w in updated_weights]
                    w_pos[e*J_pre - J_pre:e*J_pre] = updated_weights

            J_pos = e * J_pre + J_pre  #存疑


            # ------- 剪枝 -------
            T = 0.2 * (1 - self.Pd) * 0.99 + 1e-5
            W_select, M_select, P_select = [], [], []
            #k = 0
            for i in range(J_pos):
                if w_pos[i] >= T:
                    #k += 1
                    W_select.append(w_pos[i])
                    M_select.append(m_pos[i])
                    P_select.append(P_pos[i])

            # ------- 融合 -------
            U = 50
            used = [False] * len(W_select)
            X_now = []

            W_phd, M_phd, P_phd = [], [], []

            while not all(used):

                unused_indices = []
                for i in range(len(used)):
                    if not used[i]:
                        unused_indices.append(i)

                # 找出未被使用粒子中权重最大的粒子的索引
                max_idx = unused_indices[0]
                for i in unused_indices:
                    if W_select[i] > W_select[max_idx]:
                        max_idx = i

                center = np.array(M_select[max_idx])
                close_idx = [max_idx]

                # 遍历所有未被使用的粒子索引
                for j in unused_indices:
                    # 如果这个粒子就是当前的最大权重粒子，则跳过
                    if j == max_idx:
                        continue

                    # 提取当前粒子和中心粒子的 [x, y] 坐标（索引 0 和 2）
                    particle_pos = np.array(M_select[j])[ [0, 2] ]
                    center_pos = center[[0, 2]]

                    # 计算这两个粒子在位置上的欧式距离
                    dist = np.linalg.norm(particle_pos - center_pos)

                    # 如果距离小于聚合阈值 U，则将其视为一个聚类成员
                    if dist < U:
                        close_idx.append(j)

                weights = np.array([W_select[k] for k in close_idx])
                w_total = weights.sum()
                W_phd.append(w_total)

                # 计算加权均值
                mean = np.zeros_like(M_select[0])  #与M_select[0]形状相同全为0的数组
                for k in range(len(close_idx)):
                    mean += weights[k] * np.array(M_select[close_idx[k]])
                mean /= w_total
                M_phd.append(mean)

                # 计算协方差
                cov = np.zeros_like(P_select[0])
                for k in range(len(close_idx)):
                    m = np.array(M_select[close_idx[k]])
                    p = P_select[close_idx[k]]
                    cov += weights[k] * (p + (m - mean)[:, None] @ (m - mean)[None, :])
                cov /= w_total
                P_phd.append(cov)

                for k in close_idx:
                    used[k] = True

            # ------- 裁剪 -------
            Jmax = 150
            if len(W_phd) > Jmax:
                sorted_indices = sorted(range(len(W_phd)), key=lambda i: W_phd[i], reverse=True)
                top_indices = sorted_indices[:Jmax]
                W_phd = [W_phd[i] for i in top_indices]
                M_phd = [M_phd[i] for i in top_indices]
                P_phd = [P_phd[i] for i in top_indices]
            # X_now = [
            #     [W_phd, M_phd, P_phd, len(W_phd)]
            # ]   
            # return X_now      
                 
            return [W_phd, M_phd, P_phd, len(W_phd)] 



                    


