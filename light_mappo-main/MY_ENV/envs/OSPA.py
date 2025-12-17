import numpy as np
from scipy.optimize import linear_sum_assignment

def ospa(X, Y, c=120, p=2):
    n, m = len(X), len(Y)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return c
    
    X = np.array(X)
    Y = np.array(Y)
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            D[i, j] = min(c, np.linalg.norm(X[i] - Y[j]))  #计算范数
    row_ind, col_ind = linear_sum_assignment(D)   #匈牙利算法，匹配误差最小的预测目标 ↔ 真实目标
    matched_distances = D[row_ind, col_ind]

    term1 = np.sum(matched_distances**p)
    term2 = c**p * abs(n - m)
    ospa_dist = ((term1 + term2) / max(n, m)) ** (1/p)
    return ospa_dist