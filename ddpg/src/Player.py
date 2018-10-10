"""
Written by Matteo Dunnhofer - 2018

Class that defines the procedure to make an agent play and evaluate its performances
"""
import argparse
import os
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import gym
import utils as ut
from config import Configuration
from ACModel import ActorMLP, CriticMLP
from MujocoEnv import HalfCheetahEnv

class Player(object):

    def __init__(self, cfg, ckpt_path):

        self.cfg = cfg
        
        #self.env = AtariEnv(self.cfg)
        self.env = HalfCheetahEnv(self.cfg)

        if self.cfg.USE_GPU:
            self.gpu_id = 0
            self.device = torch.device('cuda', self.gpu_id)
        else:
            self.device = torch.device('cpu')

        self.model = ActorMLP(self.cfg).to(self.device)
        self.model.eval()

        self.model.load_state_dict(torch.load(ckpt_path))
        

    def play(self):

        step = 0

        while not self.env.done:
            step += 1

            state = self.env.get_state()

            state = state.to(self.device)

            self.env.render()
            
            time.sleep(0.05)
        
            mu = self.model(Variable(state))

            action = mu.data.cpu().numpy()

            _ = self.env.step(action)
        
        print ('Final score: {:.01f}'.format(self.env.total_reward))
        print ('Steps: {:.01f}'.format(self.env.steps))

    

    def play_n(self, num_games):
        """
        Play num_games games to evaluate average perfomances
        """
        score, max_score, min_score = 0., 1e-10, 1e10

        for game in range(num_games):
            step = 0

            self.env.reset()

            while not self.env.done:
                step += 1

                state = self.env.get_state()

                state = state.to(self.device)

                self.env.render()
                
                time.sleep(0.05)
            
                q_values = self.model(state)
                _, action = q_values.max(1)
                action = action.view(1, 1).cpu()

                _ = self.env.step(action.item())
            
                
            print ('Game {:04d} - Final score: {:.01f}'.format(game, self.env.total_reward))
            score += self.env.total_reward

            if self.env.total_reward > max_score:
                max_score = self.env.total_reward

            if self.env.total_reward < min_score:
                min_score = self.env.total_reward

        print ('Avg score: {:.03f}'.format(score / num_games))
        print ('Max score: {:.01f}'.format(max_score))
        print ('Min score: {:.01f}'.format(min_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help='Path to checkpoint file', type=str)
    parser.add_argument('--num-games', help='Number of games to test', type=int)
    args = parser.parse_args()

    cfg = Configuration()

    player = Player(cfg, args.ckpt)

    if args.num_games:
        player.play_n(args.num_games)
    else:
        player.play()
