# Asynchronous Advantage Actor-Critc (A3C) implementation for discrete action RL tasks

Based on the paper Asynchronous Methods for Deep Reinforcement Learning [(link)](https://arxiv.org/abs/1602.01783) by Minh et al. It is possible use Generalized Advantage Estimation [(link)](https://arxiv.org/abs/1506.02438) to get better results.

Uses the OpenAI gym's environment `CartPole-v1` and the Atari envoironments. 

Model implementation is a combination of convolutional and recurrent layers.

Use python 3. Can run also on a GPU.

Inspired by https://github.com/dgriff777/rl_a3c_pytorch .