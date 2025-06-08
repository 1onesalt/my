import copy
import logging
import random

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from target_model import model, targets, target_CV, observe_Fov
from PHD import PHD


class target_search(gym.Env):
    def __init__(self, x_min = -1000, x_max = 1000, y_min = -1000, y_max = 1000, n_agent = 3, n_target = 5, step = 100):


    
    def get_agent_mea(self):
        

    def reset():
