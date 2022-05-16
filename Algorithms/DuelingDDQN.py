from Algorithms.DDQN import DDQN
from Model import CPNet
import torch
import numpy as np


class DuelingDDQN(DDQN):
    def __init__(self, state_dim, action_dim, save_dir, checkpoint):
        super().__init__(state_dim, action_dim, save_dir, None)
        self.net = CPNet(self.state_dim, self.action_dim + 1)
        self.target = CPNet(self.state_dim, self.action_dim + 1)
        if checkpoint:
            self.load(checkpoint)

    def td_estimate(self, state, action):
        current_Q = self.net(state).view(self.batch_size, self.action_dim + 1)  # Q_online(s,a)
        current_Q = self._dueling_Q_values(current_Q)[np.arange(0, self.batch_size), action]
        return current_Q

    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state).view(self.batch_size, self.action_dim + 1)[:, :-1]
        best_action = torch.argmax(next_state_Q, dim=1)
        next_Q = self.net(next_state).view(self.batch_size, self.action_dim + 1)
        next_Q = self._dueling_Q_values(next_Q)[np.arange(0, self.batch_size), best_action]
        return (reward + done.float() * self.gamma * next_Q).float()

    @staticmethod
    def _dueling_Q_values(dueling_Q):
        state_value = dueling_Q[:, -1]
        avg_advantage = torch.mean(dueling_Q[:, :-1], dim=1)
        Q_values = state_value.unsqueeze(1) + (dueling_Q[:, :-1] - avg_advantage.unsqueeze(1))
        return Q_values

    def act(self, state):
        self.curr_step += 1

        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_dim)

        state = torch.FloatTensor(np.array(state)).cuda()
        state = state.unsqueeze(0)
        action_values = self.net(state).view(1, self.action_dim + 1)
        action_values = action_values[:, :-1]
        action_idx = torch.argmax(action_values).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        return action_idx

