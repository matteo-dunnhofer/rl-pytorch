"""
Written by Matteo Dunnhofer - 2018

Class that launch the training procedure
"""
import sys
sys.path.append('../..')

import argparse
import os
import datetime as dt
import time
import random
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import utils as ut
from config import Configuration
from Logger import Logger
from ContinuousEnv import LunarLanderEnv
from MujocoEnv import HalfCheetahEnv
from ACModel import ActorMLP, CriticMLP
from ExperienceReplay import ExperienceReplay, Transition
from OUExploration import OUNoise, OrnsteinUhlenbeckActionNoise


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

        self.logger = Logger(self.experiment_path, to_file=True, to_tensorboard=True)

        #self.env = LunarLanderEnv(self.cfg)
        self.env = HalfCheetahEnv(self.cfg)

        if self.cfg.USE_GPU:
            self.gpu_id = 0
            self.device = torch.device('cuda', self.gpu_id)
        else:
            self.device = torch.device('cpu')

        self.target_actor = ActorMLP(self.cfg).to(self.device)
        self.target_actor.eval()
        self.target_critic = CriticMLP(self.cfg).to(self.device)
        self.target_critic.eval()
        
        self.actor = ActorMLP(self.cfg).to(self.device)
        self.critic = CriticMLP(self.cfg).to(self.device)

        self.ou_noise = OUNoise(self.cfg.NUM_ACTIONS)
        #self.ou_noise = OrnsteinUhlenbeckActionNoise(np.zeros(self.cfg.NUM_ACTIONS), np.ones(self.cfg.NUM_ACTIONS) * 0.2)

        self.logger.log_config(self.cfg)
        self.logger.log_pytorch_model(self.actor)
        self.logger.log_pytorch_model(self.critic)

        self.ckpt_path = os.path.join(self.experiment_path, 'ckpt', self.target_actor.model_name + '.weights')

        if self.cfg.OPTIM == 'adam':
            self.actor_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.actor.parameters()),
                                        lr=self.cfg.ACTOR_LEARNING_RATE)
            self.critic_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.critic.parameters()),
                                        lr=self.cfg.CRITIC_LEARNING_RATE,
                                        weight_decay=self.cfg.CRITIC_WEIGHT_DECAY)
        elif self.cfg.OPTIM == 'rms-prop':
            self.actor_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.actor.parameters()),
                                        lr=self.cfg.ACTOR_LEARNING_RATE)
            self.critic_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.critic.parameters()),
                                        lr=self.cfg.CRITIC_LEARNING_RATE)
        elif self.cfg.OPTIM == 'sgd':
            self.actor_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.actor.parameters()),
                                        lr=self.cfg.ACTOR_LEARNING_RATE)
            self.critic_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.critic.parameters()),
                                        lr=self.cfg.CRITIC_LEARNING_RATE)
        
        if ckpt_path is not None:
            self.actor.load_state_dict(torch.load(ckpt_path))
            self.critic.load_state_dict(torch.load(ckpt_path))


    def train(self):
        """
        Procedure to run the training procedure
        """
        ut.hard_update(self.target_actor, self.actor)
        ut.hard_update(self.target_critic, self.critic)

        experience_replay = ExperienceReplay(self.cfg)

        global_step = 0
        epsilon_step = 0

        for e in range(self.cfg.MAX_EPISODES):

            self.ou_noise.scale = (self.cfg.EXPL_NOISE_SCALE_INIT - self.cfg.EXPL_NOISE_SCALE_END) * max(0, self.cfg.EXPL_EP_END - 
                                                                                                                        e) / self.cfg.EXPL_EP_END + self.cfg.EXPL_NOISE_SCALE_END
            self.ou_noise.reset()
    
            self.env.reset()

            state = self.env.get_state()

            while not self.env.done:

                if self.cfg.RENDER:
                    self.env.render()
                
                self.actor.eval()
                mu = self.actor(state.to(self.device))
                        
                self.actor.train()

                mu = mu.data

                exploration_noise = torch.Tensor(self.ou_noise.noise()).to(self.device)
                #exploration_noise = torch.Tensor(self.ou_noise()).to(self.device) #epsilon * ut.ornstein_uhlenbeck_exploration(mu, self.cfg.OU_EXPL_MU, self.cfg.OU_EXPL_THETA, self.cfg.OU_EXPL_SIGMA, self.device)
                mu += exploration_noise

                action = mu.clamp(-1.0, 1.0)
                action = action.cpu().numpy()
            
                reward = self.env.step(action[0])

                #r = np.clip(reward, -1.0, 1.0)
                
                action = torch.Tensor(action)
                r = torch.Tensor([reward])

                next_state = self.env.get_state()

                not_done = torch.Tensor([not self.env.done])

                experience_replay.push(state, action, next_state, r, not_done)

                state = next_state

                if len(experience_replay) > self.cfg.BATCH_SIZE:

                    for _ in range(self.cfg.UPDATE_STEPS):

                        global_step += 1
                        
                        transitions = experience_replay.sample()

                        batch = Transition(*zip(*transitions))

                        state_batch = Variable(torch.cat(batch.state)).to(self.device)
                        action_batch = Variable(torch.cat(batch.action)).to(self.device)
                        reward_batch = Variable(torch.cat(batch.reward)).to(self.device)
                        next_state_batch = Variable(torch.cat(batch.next_state)).to(self.device)
                        non_final_mask = Variable(torch.cat(batch.done)).to(self.device)

                        # reward standardization
                        if self.cfg.STD_REWARDS:
                            reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + np.finfo(np.float32).eps.item())
                        
                        next_actions = self.target_actor(next_state_batch)
                        next_state_action_values = self.target_critic(next_state_batch, next_actions)

                        reward_batch = reward_batch.unsqueeze(1)
                        non_final_mask = non_final_mask.unsqueeze(1)
                        expected_state_action_values = reward_batch + (self.cfg.GAMMA * non_final_mask * next_state_action_values)
                        
        
                        # training op
                        self.critic_optimizer.zero_grad()
                        state_action_values = self.critic(state_batch, action_batch)
                        value_loss = F.mse_loss(state_action_values, expected_state_action_values)
                        value_loss.backward()
                        self.critic_optimizer.step()

                        self.actor_optimizer.zero_grad()
                        policy_loss = -self.critic(state_batch, self.actor(state_batch))
                        policy_loss = policy_loss.mean()
                        policy_loss.backward()
                        self.actor_optimizer.step()

                        self.logger.log_value('policy_loss', global_step, policy_loss.item(), print_value=False, to_file=False)
                        self.logger.log_value('value_loss', global_step, value_loss.item(), print_value=False, to_file=False)

                        # update target weights
                        ut.soft_update(self.target_actor, self.actor, self.cfg.TAU)
                        ut.soft_update(self.target_critic, self.critic, self.cfg.TAU)

            #self.ou_noise.reset()

            self.logger.log_episode('DDPG agent', e, self.env.total_reward)
   
            if (e % self.cfg.SAVE_STEP) == 0:
                torch.save(self.target_actor.state_dict(), self.ckpt_path)
                self.logger.log_variables()

        torch.save(self.target_actor.state_dict(), self.ckpt_path)
        self.logger.log_variables()
        

    def make_folders(self):
        """
        Creates folders for the experiment logs and ckpt
        """
        os.makedirs(os.path.join(self.experiment_path, 'ckpt'))
        os.makedirs(os.path.join(self.experiment_path, 'logs'))


        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help='Resume the training from the given on the command line', type=str)
    args = parser.parse_args()

    cfg = Configuration()

    trainer = Trainer(cfg, ckpt_path=args.ckpt)
    trainer.train()





