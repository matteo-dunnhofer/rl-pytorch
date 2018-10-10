"""
Written by Matteo Dunnhofer - 2018

Classes that defines the Actor Critc model
"""
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils as ut

class ActorMLP(torch.nn.Module):

    def __init__(self, cfg, training=False):
        super(ActorMLP, self).__init__()

        self.model_name = 'ActorMLP'

        self.cfg = cfg
        self.training = training

        # network layers
        self.hidden1 = nn.Linear(17, 128)
        self.ln1 = nn.LayerNorm(128)
        self.hidden2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)

        # actor
        self.actor = nn.Linear(128, self.cfg.NUM_ACTIONS)

        # weight initialisation
        #self.apply(ut.weight_init)
        
        #self.actor.weight.data = ut.normalized_columns_initializer(self.actor.weight.data, 0.01)
        #self.actor.bias.data.fill_(0)
        self.actor.weight.data.mul_(0.1)
        self.actor.bias.data.mul_(0.1)


    def forward(self, x):
        """ 
        Function that executes the model 
        """
        x = F.relu(self.ln1(self.hidden1(x)))
        x = F.relu(self.ln2(self.hidden2(x)))

        return torch.clamp(self.actor(x), -1.0, 1.0)

class CriticMLP(torch.nn.Module):

    def __init__(self, cfg, training=False):
        super(CriticMLP, self).__init__()

        self.model_name = 'CriticMLP'

        self.cfg = cfg
        self.training = training

        # network layers
        self.hidden1 = nn.Linear(17, 128)
        self.ln1 = nn.LayerNorm(128)
        self.hidden2 = nn.Linear(128 + self.cfg.NUM_ACTIONS, 128)
        self.ln2 = nn.LayerNorm(128)
        # critic
        self.critic = nn.Linear(128, 1)

        # weight initialisation
        #self.apply(ut.weight_init)

        #self.critic.weight.data = ut.normalized_columns_initializer(self.critic.weight.data, 1.0)
        #self.critic.bias.data.fill_(0)
        self.critic.weight.data.mul_(0.1)
        self.critic.bias.data.mul_(0.1)

    def forward(self, states, actions):
        """ 
        Function that executes the model 
        """
        x = F.relu(self.ln1(self.hidden1(states)))

        x = torch.cat((x, actions), 1)

        x = F.relu(self.ln2(self.hidden2(x)))

        return self.critic(x)


class ActorCriticLSTM(torch.nn.Module):

    def __init__(self, cfg, training=False):
        super(ActorCriticLSTM, self).__init__()

        self.model_name = 'ActorCriticLSTM'

        self.cfg = cfg
        self.training = training

        self.lstm_layers = 1
        self.lstm_size = 128

        # network layers
        self.hidden1 = nn.Linear(8, 128)
        #self.hidden3 = nn.Linear(256, 256)

        #self.lstm = nn.LSTMCell(256, 256)
        self.lstm = nn.LSTM(128, hidden_size=self.lstm_size, num_layers=self.lstm_layers)

        # actor
        self.actor_mu = nn.Linear(128, self.cfg.NUM_ACTIONS)
        self.actor_sigma = nn.Linear(128, self.cfg.NUM_ACTIONS)

        # critic
        self.critic = nn.Linear(128, 1)

        # weight initialisation
        self.apply(ut.weight_init)
        
        self.actor_mu.weight.data = ut.normalized_columns_initializer(self.actor_mu.weight.data, 0.01)
        self.actor_mu.bias.data.fill_(0)

        self.actor_sigma.weight.data = ut.normalized_columns_initializer(self.actor_sigma.weight.data, 0.001)
        self.actor_sigma.bias.data.fill_(0)

        self.critic.weight.data = ut.normalized_columns_initializer(self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)
        """
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        """

    def forward(self, x, state, device):
        """ Function that executes the model 

        """
        x = F.relu(self.hidden1(x))
        #x = F.relu(self.hidden2(x))
        #x = F.relu(self.hidden3(x))

        #x = x.view(x.size(0), -1)

        #state = (Variable(state[0].data), Variable(state[1].data))
        #h_state, c_state = state[0], state[1]
        #h_state, c_state = self.lstm(x, state)

        #x = h_state

        state = (Variable(state[0].data.to(device)), Variable(state[1].data.to(device)))

        x, n_state = self.lstm(x.unsqueeze(0), state)
        x = x.squeeze(0)

        return self.actor_mu(x), self.actor_sigma(x), self.critic(x), n_state #(h_state, c_state)

    def init_state(self, device):
        """
        Returns the initial state of the model
        """
        return (torch.zeros(self.lstm_layers, 1, self.lstm_size).to(device),
                torch.zeros(self.lstm_layers, 1, self.lstm_size).to(device))


