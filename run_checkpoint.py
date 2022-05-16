from pathlib import Path
import gym
from Algorithms.DQN import DQN
from Algorithms.DDQN import DDQN
from Algorithms.DuelingDDQN import DuelingDDQN

env = gym.make("CartPole-v1")

env.reset()

checkpoint = Path('checkpoints/2022-05-17 01-57-13 Dueling CP/CartPole_net.chkpt')
agent = DuelingDDQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, save_dir=None,
                    checkpoint=checkpoint)
agent.exploration_rate = agent.exploration_rate_min
rewards = 0

state = env.reset()
while True:
    env.render()
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    rewards += reward
    state = next_state
    if done:
        break

print("Reward:", rewards)
