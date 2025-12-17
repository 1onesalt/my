import numpy as np
import gym
from MY_ENV.envs.target_search import target_search

class EnvCore(object):
    """
    # 环境中的智能体
    """
    def __init__(self):
        self.agent_num = 3
        self.action_dim = 9
        self.x_min = -1000
        self.x_max = 1000
        self.y_min = -1000
        self.y_max = 1000
        # 初始化目标搜索环境
        self.env = target_search(
            x_min=-1000, 
            x_max=1000, 
            y_min=-1000, 
            y_max=1000,
            n_agent=self.agent_num ,
            n_target=5,
            max_steps=100
        )
        # 从 target_search 中拿空间参数
        agent0_space = self.env.observation_space[0] 
        self.obs_dim  = gym.spaces.flatdim(agent0_space) 

        #agent0_space = self.env.observation_space[0] .sample()

        # 观测空间相关参数
        self.utility_channel = 2
        self.utilityMap_M = 20     
        self.utilityMap_N = 20 
        self.utility_obs_shape = (self.utility_channel, self.utilityMap_M, self.utilityMap_N)

        self.channel = 3                  
        self.ring_num = 4     
        self.sector_num = 8
        self.target_obs_shape = (self.channel, self.ring_num, self.sector_num) 

        #self.obs_dim = self.env.observation_space[0]
    

        # # 更新观测和动作空间维度
        # self.obs_shape = {
        #     'target': self.env.target_obs_shape,      # (3, 4, 8)
        #     'utility': self.env.utility_obs_shape,     # (2, 20, 20)
        #     'self_pos': (2,)                          # (2,)
        # }

    def reset(self):
        """重置环境
        Returns:
            sub_agent_obs: 所有智能体的观测列表
        """
        observations = self.env.reset()  #直接继承了self.env.reset()的观测结构
        return observations

    def step(self, actions):
        """环境步进
        Args:
            actions: 所有智能体的动作列表
        Returns:
            sub_agent_obs: 所有智能体的观测
            sub_agent_reward: 所有智能体的奖励
            sub_agent_done: 所有智能体的完成状态
            sub_agent_info: 所有智能体的额外信息
        """
        return self.env.step(actions)
    




    # def __init__(self):
    #     self.agent_num = 3  
    #     self.obs_dim = 5  
    #     self.action_dim = 9  
    #     self.map_size = 1000
    #     self.env = target_search()

    # def reset(self):
    #     """
    #     # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
    #     # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
    #     """

        
    #     sub_agent_obs = []
    #     for i in range(self.agent_num):
    #         sub_obs = np.random.random(size=(14,))
    #         sub_agent_obs.append(sub_obs)
    #     return sub_agent_obs

    # def step(self, actions):
    #     """
    #     # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
    #     # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
    #     # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
    #     # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
    #     """
    #     sub_agent_obs = []
    #     sub_agent_reward = []
    #     sub_agent_done = []
    #     sub_agent_info = []
    #     for i in range(self.agent_num):
    #         sub_agent_obs.append(np.random.random(size=(14,)))  #智能体观测，np
    #         sub_agent_reward.append([np.random.rand()])
    #         sub_agent_done.append(False)
    #         sub_agent_info.append({})

    #     return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
