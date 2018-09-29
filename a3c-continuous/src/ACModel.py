"""
Written by Matteo Dunnhofer - 2017

Class that defines the A3C Network
"""
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils as ut

class ActorCriticMLP(torch.nn.Module):

    def __init__(self, cfg, training=False, gpu_id=0):
        super(ActorCriticMLP, self).__init__()

        self.model_name = 'ActorCriticMLP'

        self.cfg = cfg
        self.training = training

        # network layers

        self.hidden1 = nn.Linear(8, 128)
        self.hidden2 = nn.Linear(128, 256)
        #self.hidden3 = nn.Linear(256, 256)

        # actor
        self.actor_mu = nn.Linear(256, self.cfg.NUM_ACTIONS)
        self.actor_sigma = nn.Linear(256, self.cfg.NUM_ACTIONS)

        # critic
        self.critic = nn.Linear(256, 1)

        # weight initialisation
        """
        self.apply(ut.weight_init)
        
        self.policy_logits.weight.data = ut.normalized_columns_initializer(self.policy_logits.weight.data, 0.01)
        self.policy_logits.bias.data.fill_(0)

        self.value.weight.data = ut.normalized_columns_initializer(self.value.weight.data, 1.0)
        self.value.bias.data.fill_(0)

        """
        if training:
            self.train()
        else:
            self.eval()

    def forward(self, x, state, gpu_id=0):
        """ Function that executes the model 

        """
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        #x = F.relu(self.hidden3(x))

        #x = x.view(x.size(0), -1)

        return self.actor_mu(x), self.actor_sigma(x), self.critic(x), None

    def init_state(self):
        """
        Returns the initial state of the model
        """
        return None


class ActorCriticModel2(torch.nn.Module):

    def __init__(self, cfg, training=False, gpu_id=0):
        super(ActorCriticModel2, self).__init__()

        self.cfg = cfg
        self.training = training
        self.cfg = cfg
        self.scope = scope
        self.training = training

        # network layers

        self.hidden1 = nn.Linear(8, 128)
        self.hidden2 = nn.Linear(128, 256)
        #self.hidden3 = nn.Linear(256, 256)

        #self.lstm = nn.LSTMCell(256, 256)

        self.init_state = (Variable(torch.zeros(1, 256)),
                            Variable(torch.zeros(1, 256)))
        
        # actor
        self.policy_mu = nn.Linear(256, self.cfg.NUM_ACTIONS)
        self.policy_sigma = nn.Linear(256, self.cfg.NUM_ACTIONS)

        # critic
        self.value = nn.Linear(256, 1)

        # weight initialisation
        """
        self.apply(ut.weight_init)
        
        self.policy_logits.weight.data = ut.normalized_columns_initializer(self.policy_logits.weight.data, 0.01)
        self.policy_logits.bias.data.fill_(0)

        self.value.weight.data = ut.normalized_columns_initializer(self.value.weight.data, 1.0)
        self.value.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        """
        if training:
            self.train()
        else:
            self.eval()

    def forward(self, x, state):
        """ Function that executes the model 

        """
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        #x = F.relu(self.hidden3(x))

        #x = x.view(x.size(0), -1)

        state = (Variable(state[0].data), Variable(state[1].data))
        
        h_state, c_state = state[0], state[1]
        #h_state, c_state = self.lstm(x, state)

        #x = h_state

        if self.training:
            return self.policy_sigma(x), self.policy_mu(x), self.value(x), (h_state, c_state)
        else:
            return self.policy_sigma(x), self.policy_mu(x), self.value(x), (h_state, c_state)

    def init_state(self):
        """
        Returns the initial state of the model
        """
        if self.cfg.USE_GPU:
            with torch.cuda.device(self.gpu_id):
                return (Variable(torch.zeros(self.lstm_layers, 1, self.lstm_size).cuda()),
                                   Variable(torch.zeros(self.lstm_layers, 1, self.lstm_size).cuda()))
        else:
            return (Variable(torch.zeros(self.lstm_layers, 1, self.lstm_size)),
                               Variable(torch.zeros(self.lstm_layers, 1, self.lstm_size)))


