"""
Written by Matteo Dunnhofer - 2018

Classes that defines the MLP model
"""
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils as ut


class MLPModel(torch.nn.Module):

    def __init__(self, cfg, training=False):
        super(MLPModel, self).__init__()

        self.model_name = 'MLPModel'

        self.cfg = cfg
        self.training = training

        # network layers
        self.hidden = nn.Linear(4, 128)
        #self.ln = nn.LayerNorm(128)
        self.output = nn.Linear(128, self.cfg.NUM_ACTIONS)

        # weight initialisation
        #self.apply(ut.weight_init)


    def forward(self, x, state, device):
        """ 
        Function that executes the model 
        """
        x = F.relu((self.hidden(x)))
        x = self.output(x)

        return x, None

    def init_state(self, device):
        """
        Returns the initial state of the model
        """
        return None
