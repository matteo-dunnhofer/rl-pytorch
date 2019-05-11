"""
Written by Matteo Dunnhofer - 2018

Utility functions
"""
import math
import numpy as np
import torch
from torch.autograd import Variable

def ensure_shared_grads(local_model, global_model, use_gpu=False):
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        if global_param.grad is not None and not use_gpu:
            return
        elif not use_gpu:
            global_param._grad = local_param.grad
        else:
            global_param._grad = local_param.grad.cpu()

def normalized_columns_initializer(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

def weight_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def normal(x, mu, sigma, device):
    pi = Variable(torch.FloatTensor([math.pi]).to(device))

    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()

    return a * b
