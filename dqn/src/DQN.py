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

