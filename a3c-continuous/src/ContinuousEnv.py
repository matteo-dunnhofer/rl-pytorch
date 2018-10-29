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

class LunarLanderEnv(AbstractEnvironment):

	def __init__(self, cfg):
		self.cfg = cfg

		self.env = gym.make('LunarLanderContinuous-v2')
		self.env.seed(self.cfg.SEED)

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
		self.state = np.array(self.state, dtype=np.float32)
		return torch.from_numpy(self.state).float()

	def render(self):
		self.env.render()


class BipedalWalkerEnv(AbstractEnvironment):

	def __init__(self, cfg):
		self.cfg = cfg

		#self.env = gym.make('BipedalWalkerHardcore-v2')
		self.env = gym.make('BipedalWalker-v2')
		self.env.seed(self.cfg.SEED)

		self.done = False
		self.total_reward = 0
		self.steps = 0
		state = self.env.reset()
		if self.cfg.STATE_STACK_N > 0:
			self.stack_state = StackState(self.cfg, state)
			self.state = self.stack_state.get()
		else:
			self.state = state

	def reset(self):
		self.done = False
		self.total_reward = 0
		self.steps = 0
		state = self.env.reset()
		if self.cfg.STATE_STACK_N > 0:
			self.stack_state = StackState(self.cfg, state)
			self.state = self.stack_state.get()
		else:
			self.state = state

	def step(self, action):
		observe, reward, done, _ = self.env.step(action)

		if self.cfg.STATE_STACK_N > 0:
			self.stack_state.add(observe)
			self.state = self.stack_state.get()
		else:
			self.state = observe

		self.done = done
		self.total_reward += reward
		self.steps += 1
		
		return reward

	def get_state(self):
		self.state = np.array(self.state, dtype=np.float32)
		return torch.from_numpy(self.state).float()

	def render(self):
		self.env.render()


class MountainCarEnv(AbstractEnvironment):

	def __init__(self, cfg):
		self.cfg = cfg

		self.env = gym.make('MountainCarContinuous-v0')
		self.env.seed(self.cfg.SEED)

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
		self.state = np.array(self.state, dtype=np.float32)
		return torch.from_numpy(self.state).float()

	def render(self):
		self.env.render()


class CartPoleEnv(AbstractEnvironment):

	def __init__(self, cfg):
		self.cfg = cfg

		self.env = gym.make('CartPole-v1')
		self.env.seed(self.cfg.SEED)

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
		self.state = np.array(self.state, dtype=np.float32)
		return torch.from_numpy(self.state).float()

	def render(self):
		self.env.render()

	
class StackState(object):
	"""
	Class to represent states as Stacks of single states
	"""

	def __init__(self, cfg, init_state):
		self.cfg = cfg
		
		state = []
		for i in range(self.cfg.STATE_STACK_N):
			state.append(init_state)
		
		self.state = np.array(state)

	
	def add(self, next_state):
		"""
		Add a new frame to the stack
		"""
		self.state = np.append(self.state[1:,:], np.array([next_state]), axis=0)

	
	def get(self):
		"""
		Returns the current state
		"""
		return self.state
