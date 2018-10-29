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
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import utils as ut
from config import Configuration
from Logger import Logger
from ACModel import ActorCriticMLP, ActorCriticLSTM
from A3C_TrainWorker import A3C_TrainWorker
#from A3C_TestWorker import A3C_TestWorker
#from SharedAdam import SharedAdam



class Trainer(object):

	def __init__(self, cfg, ckpt_path=None):
		self.cfg = cfg

		torch.manual_seed(self.cfg.SEED)
		torch.cuda.manual_seed(self.cfg.SEED)

		dt_now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		self.experiment_name = self.cfg.PROJECT_NAME + '_' + dt_now
		self.experiment_path = os.path.join(self.cfg.EXPERIMENTS_PATH, self.experiment_name)
		self.make_folders()

		self.main_logger = Logger(self.experiment_path, to_file=False)

		#self.global_model = ActorCriticMLP(self.cfg, training=True)
		self.global_model = ActorCriticLSTM(self.cfg, training=True)
		self.global_model.share_memory()

		self.optimizer = None #optim.Adam(self.global_model.parameters(), lr=self.cfg.LEARNING_RATE)
		#self.optimizer.share_memory()
		
		if ckpt_path is not None:
			self.global_model.load_state_dict(torch.load(ckpt_path))


	def train(self):
		"""
		Procedure to start the training and testing agents
		"""
		worker_threads = []

		# starting training threads
		for i in range(self.cfg.NUM_WORKERS):
			p = mp.Process(target=(self.train_worker), args=(i,))
			p.start()
			time.sleep(0.1)
			worker_threads.append(p)

		# starting test thread
		"""
		p = mp.Process(target=(self.test_worker))
		p.start()
		time.sleep(0.1)
		worker_threads.append(p)
		"""

		for worker in worker_threads:
			time.sleep(0.1)
			worker.join()

	def train_worker(self, i):
		"""
		Start a train agent
		"""
		worker = A3C_TrainWorker(self.cfg, i, self.global_model, self.experiment_path)
		worker.work()

	def test_worker(self):
		"""
		Start a test agent
		"""
		worker = A3C_TestWorker(self.cfg, self.global_model, self.experiment_path)
		worker.work()

	def make_folders(self):
		"""
		Creates folders for the experiment logs and ckpt
		"""
		os.makedirs(os.path.join(self.experiment_path, 'ckpt'))

		
if __name__ == '__main__':
	#os.environ["OMP_NUM_THREADS"] = "4"
	#os.environ["KMP_AFFINITY"] = "scatter"
	#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

	mp.set_start_method('spawn', force=True)

	parser = argparse.ArgumentParser()
	parser.add_argument('--ckpt', help='Resume the training from the given on the command line', type=str)
	args = parser.parse_args()

	cfg = Configuration()

	trainer = Trainer(cfg, ckpt_path=args.ckpt)
	trainer.train()





