"""
# @Time    : 2021/7/2 5:22 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : env_discrete.py
"""

import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore
from gym.spaces import Dict, Box

class DiscreteActionEnv(object):
    """
    对于离散动作环境的封装，为什么要封装？
    1、之后我们需要符合opengym的格式,告诉环境我的动作空间应该长什么样子,方便之后的网络拿到动作空间的参数,
    然后去设置网络输出需要多少个神经元
    2、拿到obd观测的参数,去设计网络输入的参数
    Wrapper for discrete action environment.
    """

    def __init__(self):
        self.env = EnvCore()
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # 观测空间相关参数
        self.utility_channel = self.env.utility_channel
        self.utilityMap_M = self.env.utilityMap_M     
        self.utilityMap_N = self.env.utilityMap_N
        self.utility_obs_shape = (self.utility_channel, self.utilityMap_M, self.utilityMap_N)

        self.channel = self.env.channel                  
        self.ring_num = self.env.ring_num
        self.sector_num = self.env.sector_num
        self.target_obs_shape = (self.channel, self.ring_num, self.sector_num) 

        self.x_min = self.env.x_min
        self.x_max = self.env.x_max
        self.y_min = self.env.y_min
        self.y_max = self.env.y_max

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        #注意一下这两个参数
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent_idx in range(self.num_agent):
            # physical action space
            u_action_space = spaces.Discrete(self.signal_action_dim)  #Discrete函数获取 5个离散的动作

            # if self.movable:  #movable这个参数是把每个动作空间加到一起
            total_action_space.append(u_action_space)

            # total action space  可能需要写一下
            # if len(total_action_space) > 1:  #这里为了方便拿到动作空间和观测参数
            #     # all action spaces are discrete, so simplify to MultiDiscrete action space
            #     if all(
            #         [
            #             isinstance(act_space, spaces.Discrete)
            #             for act_space in total_action_space
            #         ]
            #     ):
            #         act_space = MultiDiscrete(
            #             [[0, act_space.n - 1] for act_space in total_action_space]
            #         )
            #     else:
            #         act_space = spaces.Tuple(total_action_space)
            # self.action_space.append(act_space)
            # else:
            self.action_space.append(total_action_space[agent_idx])  #为每个智能体定义了动作空间

            # observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(  #也可以写一个字典，主要要明白代码在干一件什么样的事情
                Dict({
                    'target': Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=self.target_obs_shape,
                        dtype=np.float32
                    ),
                    'utility': Box(
                        low=0.0,
                        high=1.0,
                        shape=self.utility_obs_shape,
                        dtype=np.float32
                    ),
                    'self_pos': Box(
                        low=np.array([self.x_min, self.y_min]),
                        high=np.array([self.x_max, self.y_max]),
                        shape=(2,),
                        dtype=np.float32
                    )
                })
            )  

        self.share_observation_space = [
            Dict({
            'target': Box(
                low=-np.inf,
                high=np.inf,
                shape=self.target_obs_shape,
                dtype=np.float32
            ),
            'utility': Box(
                low=0.0,
                high=1.0,
                shape=self.utility_obs_shape,
                dtype=np.float32
            ),
            'self_pos': Box(
                low=np.array([self.x_min, self.y_min]),
                high=np.array([self.x_max, self.y_max]),
                shape=(2,),
                dtype=np.float32
            )
        })
            for _ in range(self.num_agent)
        ]
    

        # 共享观测空间
        if getattr(self.env, "share_observation_space", None) is not None:
            self.share_observation_space = self.env.share_observation_space
        else:
            self.share_observation_space = self.observation_space


    def step(self, actions):
        """
        输入actions维度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码
        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of the environment, with 2 intelligent agents inside, and each intelligent agent's action is a 5-dimensional one_hot encoding
        """

        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass


class MultiDiscrete:
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """Returns a array with one sample from each discrete action space"""
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.0), random_array) + self.low)]

    def contains(self, x):
        return (
            len(x) == self.num_discrete_space
            and (np.array(x) >= self.low).all()
            and (np.array(x) <= self.high).all()
        )

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


if __name__ == "__main__":
    DiscreteActionEnv().step(actions=None)
