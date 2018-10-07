"""
Written by Matteo Dunnhofer - 2018

Class that defines the DQN Experience Replay
"""
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ExperienceReplay(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.cfg.EXPERIENCE_REPLAY_SIZE:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.cfg.EXPERIENCE_REPLAY_SIZE

    def sample(self):
        return random.sample(self.memory, self.cfg.BATCH_SIZE)

    def __len__(self):
        return len(self.memory)