"""
Written by Matteo Dunnhofer - 2018

Class that defines a wrapper for OpenAI continuous actions environment
"""
import sys
sys.path.append('../..')

import copy
import torch
import numpy as np
import gym
import utils as ut
from AbstractEnvironment import AbstractEnvironment
from DataTransformer import DataTransformer

class CartPoleEnv(AbstractEnvironment):

	def __init__(self, cfg):
		self.cfg = cfg

		self.env = gym.make('CartPole-v1')
		self.env.seed(self.cfg.SEED)

		self.done = False
		self.total_reward = 0
		self.steps = 0
		
		self.last_obs = self.env.reset()
		self.current_obs = copy.deepcopy(self.last_obs)
		self.state = list(self.last_obs) + list(self.current_obs)
		
		#self.state = self.env.reset()



	def reset(self):
		self.done = False
		self.total_reward = 0
		self.steps = 0

		self.last_obs = self.env.reset()
		self.current_obs = copy.deepcopy(self.last_obs)
		self.state = list(self.last_obs) + list(self.current_obs)
		
		#self.state = self.env.reset()

	def step(self, action):
		observe, reward, done, _ = self.env.step(action)

		self.last_obs = copy.deepcopy(self.current_obs)
		self.current_obs = copy.deepcopy(observe)
		self.state = list(self.last_obs) + list(self.current_obs)

		#self.state = observe
		self.done = done
		self.total_reward += reward
		self.steps += 1
		
		return reward

	def get_state(self):
		return torch.from_numpy(np.array(self.state)).float().unsqueeze(0)

	def render(self):
		self.env.render()



