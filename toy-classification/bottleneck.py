import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import math

"""
Code inspiration from https://github.com/1Konny/VIB-pytorch/blob/master/model.py
"""
def xavier_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        print("Initiating bottleneck")
        nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
        m.bias.data.zero_()

class Bottleneck(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size

        self.encode = nn.Linear(input_size, 2 * output_size)
        self.weight_init()

    def forward(self, x):
        device = x.device
        stats = self.encode(x)

        mu = stats[:,:self.output_size]
        std = F.softplus(stats[:,self.output_size:])

        # if self.noisy_prior:
        #     prior_0 = Normal(torch.zeros(self.output_size).to(device), torch.ones(self.output_size).to(device) / math.sqrt(2.))
        #     prior_mu = prior_0.sample()
        #     prior = Normal(prior_mu, torch.ones(self.output_size).to(device) / math.sqrt(2.))
        prior = Normal(torch.zeros(self.output_size).to(device), torch.ones(self.output_size).to(device))

        dist = Normal(mu, std)
        kl = kl_divergence(dist, prior)

        return dist.rsample(), kl

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])
