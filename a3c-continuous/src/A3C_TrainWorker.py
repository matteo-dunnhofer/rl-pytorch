"""
Written by Matteo Dunnhofer - 2018

Class that defines the A3C training worker
"""
import sys
sys.path.append('../..')

import math
import os
import time
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import utils as ut
from Logger import Logger
from ContinuousEnv import LunarLanderEnv, BipedalWalkerEnv, MountainCarEnv
from ACModel import ActorCriticMLP, ActorCriticLSTM


class A3C_TrainWorker(object):

    def __init__(self, cfg, ident, global_model, experiment_path):

        self.worker_name = 'a3c_train_worker_' + str(ident)
        self.ident = ident

        self.cfg = cfg
        self.experiment_path = os.path.join(experiment_path, self.worker_name)
        self.make_folders()

        self.env = LunarLanderEnv(self.cfg)
        #self.env = CartPoleEnv(self.cfg)

        torch.manual_seed(self.cfg.SEED + ident)

        if self.cfg.USE_GPU:
            self.gpu_id = self.cfg.GPU_IDS[ident % len(self.cfg.GPU_IDS)]
            torch.cuda.manual_seed(self.cfg.SEED + ident)
            self.device = torch.device('cuda', self.gpu_id)
        else:
            self.gpu_id = 0
            self.device = torch.device('cpu')

        self.logger = Logger(self.experiment_path, to_file=True, to_tensorboard=True)

        self.total_reward = 0
        self.episode_count = 0
        self.step = 0

        self.global_model = global_model
        
        self.local_model = ActorCriticMLP(self.cfg, training=True).to(self.device)
        #self.local_model = ActorCriticLSTM(self.cfg, training=True, gpu_id=self.gpu_id)
        self.local_model.train()

        self.ckpt_path = os.path.join(experiment_path, 'ckpt', self.global_model.model_name + '.weights')

        self.logger.log_config(self.cfg, print_log=False)
        self.logger.log_pytorch_model(self.global_model, print_log=False)

        params = list(list(self.global_model.hidden1.parameters()) + \
                            list(self.global_model.hidden2.parameters()) + \
                            list(self.global_model.actor_mu.parameters()) + \
                            list(self.global_model.actor_mu.parameters()) )

        critic_params = list(list(self.global_model.critic.parameters()))

        parameters = [ {'params': params},
                       {'params': critic_params, 'lr': self.cfg.CRITIC_LR}]
        
        
        #if optimizer is None:
        if self.cfg.OPTIM == 'adam':
            #self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.global_model.parameters()),
            #                            lr=self.cfg.LEARNING_RATE)
            self.optimizer = optim.Adam(parameters, lr=self.cfg.LEARNING_RATE)
        elif self.cfg.OPTIM == 'rms-prop':
            self.optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.global_model.parameters()),
                                        lr=self.cfg.LEARNING_RATE)
            #self.optimizer = optim.RMSprop(parameters,
            #                            lr=self.cfg.LEARNING_RATE)
        elif self.cfg.OPTIM == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.global_model.parameters()),
                                        lr=self.cfg.LEARNING_RATE)

        if self.cfg.DECAY_LR:
            lr_milestones = self.cfg.DECAY_LR_STEPS
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=0.1)



    def work(self):
        """
        Worker training procedure
        """
        self.step = 0
        
        self.model_state = copy.deepcopy(self.local_model.init_state(self.device))

        while True:
            
            self.step += 1
            
            # update local variables with the weights
            # of the global net
            if self.cfg.USE_GPU:
                with torch.cuda.device(self.gpu_id):
                    self.local_model.load_state_dict(self.global_model.state_dict())
            else:
                self.local_model.load_state_dict(self.global_model.state_dict())

            # accumulate some experience
            # and build the loss
            loss = self.process_rollout()

            # backward pass and
            # update the global model weights
            self.local_model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.local_model.parameters()), 40.0)
            ut.ensure_shared_grads(self.local_model, self.global_model, use_gpu=self.cfg.USE_GPU)
            self.optimizer.step()
            
            self.logger.log_value('loss', self.step, loss.item(), print_value=False, to_file=False)


            if (self.step % self.cfg.SAVE_STEP) == 0 and (self.ident % 4 == 0): #self.name == 'a3c_train_worker_0':
                torch.save(self.global_model.state_dict(), self.ckpt_path)
                print('Variables saved')

            if self.episode_count > self.cfg.MAX_EPISODES:
                # terminate the training
                if self.worker_name == 'a3c_train_worker_0':
                    torch.save(self.global_model.state_dict(), self.ckpt_path)
                    print('Variables saved')
                break


    def process_rollout(self):
        """
        Interact with the envirnomant for a few time steps
        and build the loss
        """
        if self.env.done:
            self.env.reset()

            self.model_state = copy.deepcopy(self.local_model.init_state(self.device))

        
        log_probs, rewards, values, entropies = [], [], [], []

        for _ in range(self.cfg.ROLLOUT_STEPS):
        #while not self.env.done:

            state = self.env.get_state()

            state = state.to(self.device)

            policy_mu, policy_sigma, value, n_model_state = self.local_model(state.unsqueeze(0), self.model_state, self.device)

            #mu = F.softsign(policy_mu)
            mu = torch.clamp(policy_mu, -1.0, 1.0)
            #mu = F.tanh(policy_mu)
            sigma = F.softplus(policy_sigma + np.finfo(np.float32).eps.item(), beta=1.0) 
            
            """
            # Does not work good # https://discuss.pytorch.org/t/backpropagation-through-sampling-a-normal-distribution/3164
            action_dist = torch.distributions.Normal(mu, sigma.sqrt())
            action = action_dist.rsample().data
            action_log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()

            action = torch.clamp(action, -1.0, 1.0)
            """
            noise = torch.randn(mu.size()).to(self.device)
            pi = torch.FloatTensor([math.pi]).to(self.device)

            action = (mu + sigma.sqrt() * noise).data
            action_prob = ut.normal(action, mu, sigma, self.device)
            action_log_prob = (action_prob + 1e-6).log()
            entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)

            action = torch.clamp(action, -1.0, 1.0)
            
            reward = self.env.step(action.cpu().numpy()[0])

            # reward clipping
            r = max(min(float(reward), 1.0), -1.0)
            #r = reward

            log_probs.append(action_log_prob)
            rewards.append(r)
            values.append(value)
            entropies.append(entropy)

            self.model_state = n_model_state

            if self.env.done:

                if self.cfg.DECAY_LR:
                    self.lr_scheduler.step(self.episode_count)

                self.total_reward += self.env.total_reward
                self.episode_count += 1
                self.logger.log_episode(self.worker_name, self.episode_count, self.env.total_reward)
                
                break

        if self.env.done:
            R = torch.zeros(1, 1).to(self.device)
        else:
            state = self.env.get_state()

            state = state.to(self.device)

            _, _, value, _ = self.local_model(state.unsqueeze(0), self.model_state, self.device)

            R = value.data

        values.append(R)

        # computing loss
        policy_loss = 0.0
        value_loss = 0.0

        # reward standardization
        if self.cfg.STD_REWARDS and len(rewards) > 1:
            rewards = torch.Tensor(rewards).to(self.device)

            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps.item())

        if self.cfg.USE_GAE:
            gae = torch.zeros(1, 1).to(self.device)

        for i in reversed(range(len(rewards))):
            R = self.cfg.GAMMA * R + rewards[i]
            advantage = R - values[i]

            value_loss = value_loss + 0.5 * advantage.pow(2)

            if self.cfg.USE_GAE:
                delta = rewards[i] + self.cfg.GAMMA * \
                        values[i+1].data - values[i].data

                gae = gae * self.cfg.GAMMA * self.cfg.TAU + delta

            else:
                gae = advantage

            policy_loss = policy_loss - \
                            (log_probs[i].sum() * gae) - \
                          (self.cfg.ENTROPY_BETA * entropies[i].sum())

        self.logger.log_value('policy_loss', self.step, policy_loss.item(), print_value=False, to_file=False)
        self.logger.log_value('value_loss', self.step, value_loss.item(), print_value=False, to_file=False)

        return policy_loss + self.cfg.VALUE_LOSS_MULT * value_loss

    def make_folders(self):
        """
        Creates folders for the experiment logs and ckpt
        """
        os.makedirs(os.path.join(self.experiment_path, 'logs'))






