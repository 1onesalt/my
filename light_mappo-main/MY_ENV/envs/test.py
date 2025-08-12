import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import gym
import numpy as np


from MY_ENV.envs.target_model2 import model, targets, target_CV, observe_Fov, polar2dicaer
from MY_ENV.envs.PHD import PHD, State_extraction

from MY_ENV.envs.target_search import target_search

# myenv = target_search()
# myenv.reset()


# 假设你的环境文件名是 env_target_search.py
# from env_target_search import target_search  

# 初始化环境
env = target_search(n_agent=3, n_target=5, max_steps=100)

# 重置环境
obs = env.reset()

print("初始观测：")
for i, o in enumerate(obs):
    print(f"Agent {i} 初始位置:", o['self_pos'])

# 随机策略运行 5 步
for t in range(5):
    # 随机为每个智能体采样动作（0~8）
    actions = [env.action_space[i].sample() for i in range(env.n_agents)]
    
    # 与环境交互
    obs, rewards, dones, infos = env.step(actions)
    
    print(f"\n=== Step {t+1} ===")
    print("动作:", actions)
    for i, o in enumerate(obs):
        print(f"Agent {i} 位置:", o['self_pos'], "奖励:", rewards[i])

    # 如果全部 done 则提前结束
    if all(dones):
        print("环境结束")
        break

env.close()