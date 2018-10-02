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


class ActorCriticLSTM(torch.nn.Module):

    def __init__(self, cfg, training=False, gpu_id=0):
        super(ActorCriticLSTM, self).__init__()

        self.model_name = 'ActorCriticLSTM'

        self.cfg = cfg
        self.training = training
        self.gpu_id = gpu_id

        self.lstm_layers = 1
        self.lstm_size = 512

        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        #self.lstm = nn.LSTMCell(256, 256)
        self.lstm = nn.LSTM(1024, hidden_size=self.lstm_size, num_layers=self.lstm_layers)

        # actor
        self.actor = nn.Linear(self.lstm_size, self.cfg.NUM_ACTIONS)

        # critic
        self.critic = nn.Linear(self.lstm_size, 1)

        # weight initialisation
        self.apply(ut.weight_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        
        self.actor.weight.data = ut.normalized_columns_initializer(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)

        self.critic.weight.data = ut.normalized_columns_initializer(self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)
        """
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        """

    def forward(self, x, state, gpu_id=0):
        """ 
        Function that executes the model 
        """
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        #state = (Variable(state[0].data), Variable(state[1].data))
        #h_state, c_state = state[0], state[1]
        #h_state, c_state = self.lstm(x, state)

        #x = h_state

        if self.cfg.USE_GPU:
            with torch.cuda.device(gpu_id):
                state = (Variable(state[0].data.cuda()), Variable(state[1].data.cuda()))
        else:
            state = (Variable(state[0].data), Variable(state[1].data))


        x, n_state = self.lstm(x.unsqueeze(0), state)
        x = x.squeeze(0)

        return self.actor(x), self.critic(x), n_state #(h_state, c_state)

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

class ActorCriticLSTM2(torch.nn.Module):

    def __init__(self, cfg, training=False, gpu_id=0):
        super(ActorCriticLSTM2, self).__init__()

        self.model_name = 'ActorCriticLSTM2'

        self.cfg = cfg
        self.training = training
        self.gpu_id = gpu_id

        self.lstm_layers = 1
        self.lstm_size = 128

        # network layers
        self.hidden1 = nn.Linear(4, 128)
        #self.hidden3 = nn.Linear(256, 256)

        #self.lstm = nn.LSTMCell(256, 256)
        self.lstm = nn.LSTM(128, hidden_size=self.lstm_size, num_layers=self.lstm_layers)

        # actor
        self.actor = nn.Linear(128, self.cfg.NUM_ACTIONS)

        # critic
        self.critic = nn.Linear(128, 1)

        # weight initialisation
        self.apply(ut.weight_init)
        
        self.actor.weight.data = ut.normalized_columns_initializer(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
       
        self.critic.weight.data = ut.normalized_columns_initializer(self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)

    def forward(self, x, state, gpu_id=0):
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

        if self.cfg.USE_GPU:
            with torch.cuda.device(gpu_id):
                state = (Variable(state[0].data.cuda()), Variable(state[1].data.cuda()))
        else:
            state = (Variable(state[0].data), Variable(state[1].data))


        x, n_state = self.lstm(x.unsqueeze(0), state)
        x = x.squeeze(0)

        return self.actor(x), self.critic(x), n_state #(h_state, c_state)

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

