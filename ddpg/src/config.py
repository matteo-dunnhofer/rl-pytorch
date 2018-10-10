"""
Written by Matteo Dunnhofer - 2018

Configuration class
"""
import math
import torch
from torch.autograd import Variable


class Configuration(object):
    """
    Class defining all the hyper parameters of:
        - net architecture
        - training
        - test
        - dataset path
        ...
    """
    PROJECT_NAME = 'DDPG'

    ENV = 'LunarLanderContinuous'
    EXPERIMENTS_PATH = '../experiments'

    # training hyperparameters
    OBSERVATION_SIZE = [80, 80]

    SEED = 4 #123
    ACTOR_LEARNING_RATE = 1e-4
    CRITIC_LEARNING_RATE = 1e-3
    CRITIC_WEIGHT_DECAY = 1e-2
    BATCH_SIZE = 128
    DECAY_LR = False
    DECAY_LR_STEPS = [500]
    OPTIM = 'adam'
    MOMENTUM = 0.95
    MAX_EPISODES = 1000000
    UPDATE_STEPS = 5
    GAMMA = 0.99
    
    EXPL_NOISE_SCALE_INIT = 0.3
    EXPL_NOISE_SCALE_END = 0.3
    EXPL_EP_END = 100 # number of episodes to end exploration
    OU_EXPL_MU = 0.0
    OU_EXPL_THETA = 0.15
    OU_EXPL_SIGMA = 0.2
    TAU = 1e-3
    EXPERIENCE_REPLAY_SIZE = 1000000
    STD_REWARDS = False

    NUM_ACTIONS = 6
    
    USE_GPU = True
    GPU_IDS = [0] #[0, 1, 2]

    RENDER = False
    DISPLAY_STEP = 100
    SAVE_STEP = 1

    def __init__(self):
        super(Configuration, self).__init__()

    def __str__(self):
        cfg2str = "Configuration parameters\n"
        cfg2str += "PROJECT_NAME = " + str(self.PROJECT_NAME) + '\n'
        cfg2str += "ENV = " + str(self.ENV) + '\n'
        cfg2str += "SEED = " + str(self.SEED) + '\n'
        cfg2str += "ACTOR_LEARNING_RATE = " + str(self.ACTOR_LEARNING_RATE) + '\n'
        cfg2str += "CRITIC_LEARNING_RATE = " + str(self.CRITIC_LEARNING_RATE) + '\n'
        cfg2str += "BATCH_SIZE = " + str(self.BATCH_SIZE) + '\n'
        cfg2str += "DECAY_LR = " + str(self.DECAY_LR) + '\n'
        cfg2str += "DECAY_LR_STEPS = " + str(self.DECAY_LR_STEPS) + '\n'
        cfg2str += "OPTIM = " + str(self.OPTIM) + '\n'
        cfg2str += "MOMENTUM = " + str(self.MOMENTUM) + '\n'
        cfg2str += "MAX_EPISODES = " + str(self.MAX_EPISODES) + '\n'
        cfg2str += "NUM_ACTIONS = " + str(self.NUM_ACTIONS) + '\n'
        cfg2str += "GAMMA = " + str(self.GAMMA) + '\n'
        cfg2str += "STD_REWARDS = " + str(self.STD_REWARDS) + '\n'
        cfg2str += "EXPL_NOISE_SCALE_INIT = " + str(self.EXPL_NOISE_SCALE_INIT) + '\n'
        cfg2str += "EXPL_NOISE_SCALE_END = " + str(self.EXPL_NOISE_SCALE_END) + '\n'
        cfg2str += "EXPL_EP_END = " + str(self.EXPL_EP_END) + '\n'
        cfg2str += "OU_EXPL_MU = " + str(self.OU_EXPL_MU) + '\n'
        cfg2str += "OU_EXPL_THETA = " + str(self.OU_EXPL_THETA) + '\n'
        cfg2str += "OU_EXPL_SIGMA = " + str(self.OU_EXPL_SIGMA) + '\n'
        cfg2str += "TAU = " + str(self.TAU) + '\n'
        cfg2str += "OBSERVATION_SIZE = " + str(self.OBSERVATION_SIZE) + '\n'
        cfg2str += "RENDER = " + str(self.RENDER) + '\n'
        cfg2str += "DISPLAY_STEP = " + str(self.DISPLAY_STEP) + '\n'
        cfg2str += "SAVE_STEP = " + str(self.SAVE_STEP) + '\n'
        cfg2str += "EXPERIMENTS_PATH = " + str(self.EXPERIMENTS_PATH) + '\n'
        cfg2str += "USE_GPU = " + str(self.USE_GPU) + '\n'
        cfg2str += "GPU_IDS = " + str(self.GPU_IDS) + '\n'

        return cfg2str
