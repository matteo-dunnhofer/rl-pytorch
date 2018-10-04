"""
Written by Matteo Dunnhofer - 2018

Classes that defines the Deep Q-Network model
"""
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils as ut


class DQN(nn.Module):

    def __init__(self, cfg):
        super(DQN, self).__init__()
        self.cfg = cfg
        self.model_name = 'DQN'

        self.conv1 = nn.Conv2d(self.cfg.STATE_STACK_N, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.fc = nn.Linear(1568, self.cfg.NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        return self.fc(x.view(x.size(0), -1))


class DQNMLP(torch.nn.Module):

    def __init__(self, cfg, training=False, gpu_id=0):
        super(DQNMLP, self).__init__()

        self.model_name = 'DQNMLP'

        self.cfg = cfg
        self.training = training

        # network layers
        self.hidden1 = nn.Linear(8, 128)
        self.hidden2 = nn.Linear(128, 256)
        #self.hidden3 = nn.Linear(256, 256)

        # actor
        self.output = nn.Linear(256, self.cfg.NUM_ACTIONS)

    def forward(self, x, gpu_id=0):
        """ 
        Function that executes the model 
        """
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        #x = F.relu(self.hidden3(x))

        return self.output(x)
