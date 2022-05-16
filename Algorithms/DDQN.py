from Algorithms.DQN import DQN
import torch
import numpy as np
from Model import CPNet


class DDQN(DQN):
    def __init__(self, state_dim, action_dim, save_dir, checkpoint):
        super().__init__(state_dim, action_dim, save_dir, checkpoint)
        self.sync_every = 10000  # no. of experiences between Q_target & Q_online sync
        self.target = CPNet(state_dim, action_dim)

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state).view(self.batch_size, self.action_dim)
        best_action = torch.argmax(next_state_Q, dim=1)
        next_Q = self.target(next_state).view(self.batch_size, self.action_dim)
        next_Q = next_Q[np.arange(0, self.batch_size), best_action]
        return (reward + done.float() * self.gamma * next_Q).float()

    def sync_Q_target(self):
        self.target.load_state_dict(self.net.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        return super().learn()
