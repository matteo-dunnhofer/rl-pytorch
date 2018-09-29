"""
Written by Matteo Dunnhofer - 2018

Abstract class use to represent an environment
"""
from abc import ABCMeta, abstractmethod

class AbstractEnvironment:
	__metaclass__ = ABCMeta

	@abstractmethod
	def start(self):
		"""
		Starts the environment
		"""
		raise NotImplementedError()

	@abstractmethod
	def reset(self):
		"""
		Reinitializes the environment
		"""
		raise NotImplementedError()

	@abstractmethod
	def step(self, action):
		"""
		Executes the action and returns the reward
		"""
		raise NotImplementedError()

	@abstractmethod
	def get_state(self):
		"""
		Return the current state of the environment
		"""
		raise NotImplementedError()

	@abstractmethod
	def render(self):
		"""
		Shows visually the state of the environment
		"""
		raise NotImplementedError()

	