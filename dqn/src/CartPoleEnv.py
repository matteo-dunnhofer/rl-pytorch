"""
Written by Matteo Dunnhofer - 2018

Class that defines a wrapper for OpenAI continuous actions environment
"""
import sys
sys.path.append('../..')

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

		self.done = False
		self.total_reward = 0
		self.steps = 0
		self.state = self.env.reset()



	def reset(self):
		self.done = False
		self.total_reward = 0
		self.steps = 0
		self.state = self.env.reset()

	def step(self, action):
		observe, reward, done, _ = self.env.step(action)
		self.state = observe
		self.done = done
		self.total_reward += reward
		self.steps += 1
		
		return reward

	def get_state(self):
		return torch.from_numpy(np.array(self.state)).float()

	def render(self):
		self.env.render()



