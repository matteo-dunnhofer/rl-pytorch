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
    PROJECT_NAME = 'LunarLander'

    ENV = ''
    DATA_PATH = ''
    EXPERIMENTS_PATH = '../experiments'

    # training hyperparameters
    OBSERVATION_SIZE = [128, 128]

    SEED = 123
    LEARNING_RATE = 1e-5
    DECAY_LR = False
    DECAY_LR_STEPS = []
    OPTIM = 'adam'
    MOMENTUM = 0.95
    MAX_EPISODES = 10000 
    GAMMA = 0.99
    STD_REWARDS = False
    
    USE_GAE = True # use Generalized Advantage Estimation
    TAU = 1.0

    ENTROPY_BETA = 1e-4
    
    ROLLOUT_STEPS = 20
    NUM_WORKERS = 8
    NUM_ACTIONS = 2
    
    USE_GPU = False
    GPU_IDS = [0] #[0, 1, 2]


    DISPLAY_STEP = 100 #1100
    SAVE_STEP = 50
    VALIDATION_STEP = 1000
    CKPT_PATH = '../ckpt'
    SUMMARY_PATH = 'summary'

    def __init__(self):
        super(Configuration, self).__init__()

    def __str__(self):
        cfg2str = "Configuration parameters\n"
        cfg2str += "SEED = " + str(self.SEED) + '\n'
        cfg2str += "LEARNING_RATE = " + str(self.LEARNING_RATE) + '\n'
        cfg2str += "OPTIM = " + str(self.OPTIM) + '\n'
        cfg2str += "MOMENTUM = " + str(self.MOMENTUM) + '\n'
        cfg2str += "MAX_EPISODES = " + str(self.MAX_EPISODES) + '\n'
        cfg2str += "NUM_ACTIONS = " + str(self.NUM_ACTIONS) + '\n'
        cfg2str += "GAMMA = " + str(self.GAMMA) + '\n'
        cfg2str += "TAU = " + str(self.TAU) + '\n'
        cfg2str += "ROLLOUT_STEPS = " + str(self.ROLLOUT_STEPS) + '\n'
        cfg2str += "DISPLAY_STEP = " + str(self.DISPLAY_STEP) + '\n'
        cfg2str += "SAVE_STEP = " + str(self.SAVE_STEP) + '\n'
        cfg2str += "DATA_PATH = " + str(self.DATA_PATH) + '\n'

        return cfg2str
