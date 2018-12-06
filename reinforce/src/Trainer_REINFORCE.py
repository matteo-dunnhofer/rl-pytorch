"""
Written by Matteo Dunnhofer - 2018

Class that launch the training procedure
"""
import sys
sys.path.append('../..')

import argparse
import random
import os
import copy
import datetime as dt
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import utils as ut
from config import Configuration
from Logger import Logger
from MLPModel import MLPModel
from CartPoleEnv import CartPoleEnv



class Trainer(object):

    def __init__(self, cfg, ckpt_path=None):
        self.cfg = cfg

        random.seed(self.cfg.SEED)
        np.random.seed(self.cfg.SEED)
        torch.manual_seed(self.cfg.SEED)
        torch.cuda.manual_seed(self.cfg.SEED)

        dt_now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.experiment_name = self.cfg.PROJECT_NAME + '_' + dt_now
        self.experiment_path = os.path.join(self.cfg.EXPERIMENTS_PATH, self.experiment_name)
        self.make_folders()

        self.env = CartPoleEnv(self.cfg)

        self.logger = Logger(self.experiment_path, to_file=True, to_tensorboard=True)

        if self.cfg.USE_GPU:
            self.gpu_id = 0
            self.device = torch.device('cuda', self.gpu_id)
        else:
            self.device = torch.device('cpu')

        self.model = MLPModel(self.cfg, training=True).to(self.device)

        self.logger.log_config(self.cfg)
        self.logger.log_pytorch_model(self.model)

        self.ckpt_path = os.path.join(self.experiment_path, 'ckpt', self.model.model_name + '.weights')

        if self.cfg.OPTIM == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.cfg.LEARNING_RATE)
            #self.optimizer = optim.Adam(parameters, lr=self.cfg.LEARNING_RATE)
        elif self.cfg.OPTIM == 'rms-prop':
            self.optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.cfg.LEARNING_RATE)
            #self.optimizer = optim.RMSprop(parameters,
            #                            lr=self.cfg.LEARNING_RATE)
        elif self.cfg.OPTIM == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.cfg.LEARNING_RATE)

        if self.cfg.DECAY_LR:
            lr_milestones = self.cfg.DECAY_LR_STEPS
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=0.1)
        
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))

        self.total_reward = 0
        self.episode_count = 0
        self.step = 0


    def train(self):
        """
        Procedure to start the training and testing agents
        """
        self.step = 0
        
        self.model_state = copy.deepcopy(self.model.init_state(self.device))

        while True:
            
            self.step += 1
            

            self.model = self.model.eval()
            # accumulate some experience
            # and build the loss
            loss = self.process_rollout()

            self.model = self.model.train()
            # backward pass and
            # update the global model weights
            self.model.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 40.0)
            self.optimizer.step()
            
            self.logger.log_value('loss', self.step, loss.item(), print_value=False, to_file=False)


            if (self.step % self.cfg.SAVE_STEP) == 0:
                torch.save(self.model.state_dict(), self.ckpt_path)
                self.logger.log_variables()

            if self.episode_count > self.cfg.MAX_EPISODES:
                # terminate the training
                torch.save(self.model.state_dict(), self.ckpt_path)
                self.logger.log_variables()
                break


    def process_rollout(self):
        """
        Interact with the envirnomant for a few time steps
        and build the loss
        """
        if self.env.done:
            self.env.reset()

            self.model_state = copy.deepcopy(self.model.init_state(self.device))

        
        log_probs, rewards, entropies = [], [], []

        while not self.env.done:

            state = self.env.get_state()

            state = state.to(self.device)

            policy, n_model_state = self.model(state.unsqueeze(0), self.model_state, self.device)

            action_prob = F.softmax(policy, dim=1)
            #action_log_prob = F.log_softmax(policy, dim=1)
            #entropy = -(action_log_prob * action_prob).sum(1)

            #action = action_prob.multinomial(1).data
            #action_log_prob = action_log_prob.gather(1, Variable(action))

            action_dist = torch.distributions.Categorical(action_prob)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            
            reward = self.env.step(action.cpu().item())

            if self.cfg.CLIP_REWARDS:
                # reward clipping
                r = max(min(float(reward), 1.0), -1.0)
            else:
                r = reward

            log_probs.append(action_log_prob)
            rewards.append(r)
            entropies.append(entropy)

            self.model_state = n_model_state

            if self.env.done:

                if self.cfg.DECAY_LR:
                    self.lr_scheduler.step(self.episode_count)

                self.total_reward += self.env.total_reward
                self.episode_count += 1
                self.logger.log_episode('REINFORCE', self.episode_count, self.env.total_reward)
                
                break

        R = Variable(torch.zeros(1, 1).to(self.device))

        # computing loss
        policy_loss = 0.0

        rewards_ = []
        for i in reversed(range(len(rewards))):
            R = self.cfg.GAMMA * R + rewards[i]
            rewards_.insert(0, R)

        rewards = torch.Tensor(rewards_).to(self.device)

        # reward standardization
        if self.cfg.STD_REWARDS and len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps.item())

        for i in range(len(rewards)):
            policy_loss = policy_loss + (-log_probs[i] * Variable(rewards[i]))

        self.logger.log_value('policy_loss', self.step, policy_loss.item(), print_value=False, to_file=False)

        return policy_loss


    def make_folders(self):
        """
        Creates folders for the experiment logs and ckpt
        """
        os.makedirs(os.path.join(self.experiment_path, 'ckpt'))
        os.makedirs(os.path.join(self.experiment_path, 'logs'))

        
if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help='Resume the training from the given on the command line', type=str)
    args = parser.parse_args()

    cfg = Configuration()

    trainer = Trainer(cfg, ckpt_path=args.ckpt)
    trainer.train()





