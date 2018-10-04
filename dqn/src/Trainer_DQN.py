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
import utils as ut
from config import Configuration
from Logger import Logger
from AtariEnv import AtariEnv
from CartPoleEnv import CartPoleEnv
from DQN import DQN, DQNMLP
from ExperienceReplay import ExperienceReplay, Transition



class Trainer(object):

    def __init__(self, cfg, ckpt_path=None):
        self.cfg = cfg

        torch.manual_seed(self.cfg.SEED)
        torch.cuda.manual_seed(self.cfg.SEED)

        dt_now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.experiment_name = self.cfg.PROJECT_NAME + '_' + dt_now
        self.experiment_path = os.path.join(self.cfg.EXPERIMENTS_PATH, self.experiment_name)
        self.make_folders()

        self.logger = Logger(self.experiment_path, to_file=True, to_tensorboard=True)

        #self.env = AtariEnv(self.cfg)
        self.env = CartPoleEnv(self.cfg)

        if self.cfg.USE_GPU:
            self.gpu_id = 0
            self.device = torch.device('cuda', self.gpu_id)
        else:
            self.device = torch.device('cpu')

        #self.target_model = DQN(self.cfg).to(self.device)
        self.target_model = DQNMLP(self.cfg).to(self.device)
        self.target_model.eval()
        #self.model = DQN(self.cfg).to(self.device)
        self.model = DQNMLP(self.cfg).to(self.device)
        self.model.train()

        self.logger.log_config(self.cfg)
        self.logger.log_pytorch_model(self.model)

        self.ckpt_path = os.path.join(self.experiment_path, 'ckpt', self.target_model.model_name + '.weights')

        
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
        
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path))


    def train(self):
        """
        Procedure to run the training procedure
        """
        self.target_model.load_state_dict(self.model.state_dict())

        experience_replay = ExperienceReplay(self.cfg)

        epsilon = self.cfg.EPS_START
        drop_epsilon = (self.cfg.EPS_START - self.cfg.EPS_END) / self.cfg.EPS_DECAY

        global_step = 0

        for e in range(self.cfg.MAX_EPISODES):
    
            self.env.reset()

            state = self.env.get_state()
            #state = state.to(self.device)

            while not self.env.done:

                global_step += 1

                if self.cfg.RENDER:
                    self.env.render()
                
                if global_step > self.cfg.TRAIN_START:
                    epsilon = self.cfg.EPS_END + (self.cfg.EPS_START - self.cfg.EPS_END) * \
                                math.exp(-1. * global_step / self.cfg.EPS_DECAY)

                    self.logger.log_value('epsilon', global_step, epsilon, print_value=False, to_file=False)
            
                if random.random() < epsilon:
                    # epsilon-greedy exploration
                    action = torch.tensor([[random.randrange(self.cfg.NUM_ACTIONS)]], dtype=torch.long)
                else:
                    with torch.no_grad():
                        q_pred = self.model(state.to(self.device))
                        
                        _, action = q_pred.max(1)
                        action = action.view(1, 1).cpu()
            
                reward = self.env.step(action.item())

                r = np.clip(reward, -1.0, 1.0)
                r = torch.tensor([reward])

                if not self.env.done:
                    next_state = self.env.get_state()
                    #next_state = next_state.to(self.device)
                else:
                    next_state = None

                experience_replay.push(state, action, next_state, r)
                #total_experience.add(total_experience.get_max(), (w_state, action, r, next_w_state, done))

                if global_step > self.cfg.TRAIN_START and len(experience_replay) > self.cfg.BATCH_SIZE:

                    transitions = experience_replay.sample()

                    batch = Transition(*zip(*transitions))

                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.uint8)
                    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

                    state_batch = torch.cat(batch.state).to(self.device)
                    action_batch = torch.cat(batch.action).to(self.device)
                    reward_batch = torch.cat(batch.reward).to(self.device)
                    
                    state_action_values = self.model(state_batch).gather(1, action_batch)
                    next_state_values = torch.zeros(self.cfg.BATCH_SIZE, device=self.device)
                    next_state_values[non_final_mask] = self.target_model(non_final_next_states.to(self.device)).max(1)[0].detach()

                    expected_state_action_values = (next_state_values * self.cfg.GAMMA) + reward_batch
                    # double dqn
                    """
                    if True:
                        q1_pred_n = self.train_dqn(next_state_batch)
                        _, q1_action = torch.max(q1_pred_n, 1)

                        q2_pred[not_done_mask] = self.target_dqn(next_state_batch).gather(1, q1_action.unsqueeze(1)).squeeze(1)
                    else:
                        q2_pred[not_done_mask] = self.target_dqn(next_state_batch).max(1)[0]
                    """

                    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

                    # training op
                    self.optimizer.zero_grad()
                    loss.backward()
                    for param in self.model.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.optimizer.step()

                    self.logger.log_value('loss', global_step, loss.item(), print_value=False, to_file=False)

                    #for i in range(self.cfg.BATCH_SIZE):
                    #   total_experience.update(b_i[i], errors[i])

            # update target weights
            if (e % self.cfg.TARGET_UPDATE) == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            #self.logger.log_value('reward', e, self.env.total_reward, print_value=True, to_file=True)
            self.logger.log_episode('DQN agent', e, self.env.total_reward)

            
            if (e % self.cfg.SAVE_STEP) == 0:
                torch.save(self.target_model.state_dict(), self.ckpt_path)
                self.logger.log_variables()
                """
                errors = np.abs(t - target_q)
                total_experience.add(errors[0], (w_state, action, r, next_w_state, done))
                """
        torch.save(self.target_model.state_dict(), self.ckpt_path)
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





