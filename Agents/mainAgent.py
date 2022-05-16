from abc import abstractmethod

import numpy
import torch
import random
from collections import deque


class Agent:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque()
        self.batch_size = 256

        self.gamma = 0.99

        self.curr_step = 0
        self.burnin = self.batch_size  # min. experiences before training
        self.save_dir = save_dir

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def load(self, load_path):
        pass

    @abstractmethod
    def save(self):
        pass

    def cache(self, state, next_state, action, reward, done):

        done = 1 - done
        state = torch.FloatTensor(numpy.array(state)).cuda()
        next_state = torch.FloatTensor(numpy.array(next_state)).cuda()
        action = torch.LongTensor([action]).cuda()
        reward = torch.DoubleTensor([reward]).cuda()
        done = torch.BoolTensor([done]).cuda()

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
