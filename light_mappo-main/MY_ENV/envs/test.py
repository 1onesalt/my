import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import gym
import numpy as np


from MY_ENV.envs.target_model2 import model, targets, target_CV, observe_Fov, polar2dicaer
from MY_ENV.envs.PHD import PHD, State_extraction

from MY_ENV.envs.target_search import target_search

myenv = target_search()
myenv.reset()