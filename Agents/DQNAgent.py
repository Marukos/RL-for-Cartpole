from abc import ABC

import numpy as np
from Agents.mainAgent import Agent
from Model import CPNet
import torch


class DQNAgent(Agent, ABC):
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        super().__init__(state_dim, action_dim, save_dir)

        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99
        self.exploration_rate_min = 7.40e-05

        self.net = CPNet(self.state_dim, self.action_dim).float()

        if checkpoint:
            self.load(checkpoint)

    def act(self, state):
        self.curr_step += 1

        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_dim)

        state = torch.FloatTensor(np.array(state)).cuda()
        state = state.unsqueeze(0)
        action_values = self.net(state)
        action_idx = torch.argmax(action_values).item()

        return action_idx

    def save(self):
        save_path = self.save_dir / f"CartPole_net.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"CartPole_net saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location='cuda')
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.exploration_rate = exploration_rate
        self.net.load_state_dict(state_dict)
