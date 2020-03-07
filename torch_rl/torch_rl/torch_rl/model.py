from abc import abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F

class ACModel:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass

    @abstractmethod
    def compute_run(self, obs):
        pass

    @abstractmethod
    def compute_train(self, obs):
        pass

class RecurrentACModel(ACModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass