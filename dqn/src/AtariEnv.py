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


class AtariState(object):
	"""
	Class to represent Deep Q-Learning states as Stacks of RGB frames
	"""

	def __init__(self, cfg, init_frame, crop):
		self.cfg = cfg
		self.crop = crop
		
		self.data_transformer = DataTransformer(self.cfg)

		frame = self.data_transformer.preprocess(init_frame, self.crop)

		state = []
		for i in range(self.cfg.ATARI_STATE_STACK_N):
			state.append(frame)
		
		self.state = np.reshape(np.array(state), (self.cfg.OBSERVATION_SIZE[0], self.cfg.OBSERVATION_SIZE[1], self.cfg.ATARI_STATE_STACK_N))

	
	def add(self, next_frame):
		"""
		Add a new frame to the stack
		"""
		next_frame = self.data_transformer.preprocess(next_frame, self.crop)
		
		self.state = np.append(self.state[:,:,1:], next_frame, axis=2)

	
	def get(self):
		"""
		Returns the current state
		"""
		return self.state.transpose((2, 0, 1))

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
		self.env.seed(self.cfg.SEED)

		self.done = False
		self.total_reward = 0
		self.steps = 0
		
		frame = self.env.reset()
		self.atari_state = AtariState(self.cfg, frame, self.crop)
		self.state = self.atari_state.get()


	def reset(self):
		self.done = False
		self.total_reward = 0
		self.steps = 0
		
		frame = self.env.reset()
		self.atari_state = AtariState(self.cfg, frame, self.crop)
		self.state = self.atari_state.get()

	def step(self, action):
		frame, reward, done, _ = self.env.step(action)
		self.atari_state.add(frame)
		
		self.state = self.atari_state.get()
		self.done = done
		self.total_reward += reward
		self.steps += 1
		
		return reward

	def get_state(self):
		return torch.from_numpy(self.state).float().unsqueeze(0)

	def render(self):
		self.env.render()

