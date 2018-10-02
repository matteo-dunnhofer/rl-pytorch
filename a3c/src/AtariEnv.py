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

class AtariEnv(AbstractEnvironment):

	# crop sizes
	default = (34, 34, 80)
	pong = (34, 34, 80)
	breakout = (34, 34, 80)
	spaceinvaders = (30, 30, 94)
	mspacman = (30, 30, 94)

	def __init__(self, cfg):
		self.cfg = cfg

		self.env_name = self.cfg.ENV
		self.data_transformer = DataTransformer(self.cfg)

		if 'Breakout' in self.env_name:
			self.crop = self.breakout
		elif 'Pong' in self.env_name:
			self.crop = self.pong
		elif 'SpaceInvaders' in self.env_name:
			self.crop = self.spaceinvaders	
		elif 'MsPacman' in self.env_name:
			self.crop = self.mspacman
		else:
			self.crop = self.default		

		self.env = gym.make(self.env_name)

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
		state = self.data_transformer.preprocess(self.state, self.crop)
		
		return torch.from_numpy(state).float()

	def render(self):
		self.env.render()

