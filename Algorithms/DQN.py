from Agents.DQNAgent import DQNAgent
import torch
import numpy as np


class DQN(DQNAgent):

    def td_estimate(self, state, action):
        current_Q = self.net(state).view(self.batch_size, self.action_dim)
        current_Q = current_Q[np.arange(0, self.batch_size), action]
        return current_Q

    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state).view(self.batch_size, self.action_dim)
        best_action = torch.argmax(next_state_Q, dim=1)
        next_Q = next_state_Q[np.arange(0, self.batch_size), best_action]
        return (reward + done.float() * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.net.loss_fn(td_estimate, td_target)

        self.net.zero_grad()
        loss.backward()
        self.net.optimizer.step()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        return loss.item()

    def learn(self):

        if self.curr_step < self.burnin:
            return None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Back-propagate loss through Q_online
        loss = self.update_Q_online(td_tgt, td_est)

        return loss
