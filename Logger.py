"""
Written by Matteo Dunnhofer - 2018

Utility class to log training info to sstdout, to file, to tensorboard
"""
import os
import datetime as dt
from tensorboardX import SummaryWriter

class Logger(object):
    
    def __init__(self, experiment_path, to_file=False, to_tensorboard=False):
        super(Logger, self).__init__()
        
        self.experiment_path = os.path.join(experiment_path, 'logs')
        self.to_file = to_file
        self.to_tensorboard = to_tensorboard

        if self.to_file:
            self.log_file = open(os.path.join(self.experiment_path, 'log.txt'), 'w+')
            self.log_file.write(self.experiment_path + '\n\n')

        if self.to_tensorboard:
            self.tb_writer = SummaryWriter(os.path.join(experiment_path, 'tb_logs'))

    def log_config(self, config, print_log=True):
        """
        Log to file all the congfiguration data of the experiment
        """
        if print_log:
            print(str(config) + '\n')

        if self.to_file:
            self.log_file.write(str(config) + '\n')

    def log_pytorch_model(self, model, print_log=True):
        """
        Log to file the model architecture
        """
        if print_log:
            print(str(model) + '\n')

        if self.to_file:
            self.log_file.write(str(model) + '\n\n')

    def log_loss(self, loss_type, step, value):
        """
        Logging loss data
        """
        log_str = '[{}] Step: {:08d} - Loss {:.05f}'.format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step, value)
        print(log_str)

        if self.to_file:
            self.log_file.write(log_str + '\n')

        if self.to_tensorboard:
            self.tb_writer.add_scalar(loss_type + '_loss', value, step)

    def log_value(self, value_type, step, value, print_value=True, to_file=True, to_tensorboard=True):
        """
        Logging loss data
        """
        log_str = '[{}] Step: {:08d} - {} {:.05f}'.format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step, value_type, value)
        
        if print_value:
            print(log_str)

        if to_file:
            self.log_file.write(log_str + '\n')

        if to_tensorboard:
            self.tb_writer.add_scalar(value_type, value, step)


    def log_episode(self, worker_name, episode, reward):
        """
        Logging loss data
        """
        log_str = '[{}] Worker: {} --- Episode: {:07d} - Reward {:05f}'.format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), worker_name, episode, reward)
        print(log_str)

        if self.to_file:
            self.log_file.write(log_str + '\n')

        if self.to_tensorboard:
            self.tb_writer.add_scalar('reward', reward, episode)

    def log_test_episode(self, episode, reward):
        """
        Logging loss data
        """
        log_str = '[{}] Test episode: {:05d}  - Reward {:05f}'.format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), episode, reward)
        print(log_str)

        if self.to_file:
            self.log_file.write(log_str + '\n')

        #if self.to_tensorboard:
            # TODO

    def log_variables(self):
        log_str = '[{}] Variables saved'.format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(log_str)

        if self.to_file:
            self.log_file.write(log_str + '\n')

    def close(self):
        self.log_file.close()
