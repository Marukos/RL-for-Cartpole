import os

from Algorithms.DQN import DQN
from Algorithms.DDQN import DDQN
from Algorithms.DuelingDDQN import DuelingDDQN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import datetime
from pathlib import Path
import gym
from collections import deque

from metrics import MetricLogger

env = gym.make("CartPole-v1")

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S Dueling CP')
save_dir.mkdir(parents=True)

checkpoint = None
agent = DuelingDDQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n, save_dir=save_dir,
                    checkpoint=checkpoint)

logger = MetricLogger(save_dir)
rewards = deque(maxlen=100)
episode = 0
finish = False

while True:
    state = env.reset().reshape(1, env.observation_space.shape[0])
    while True:
        # env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.reshape(1, env.observation_space.shape[0])
        agent.cache(state, next_state, action, reward, done)
        loss = agent.learn()
        logger.log_step(reward, loss)
        state = next_state
        if done:
            break

    rewards.append(logger.curr_ep_reward)
    if len(rewards) == 100 and sum(rewards) / 100 >= 475:
        finish = True
        print("Solved the CartPole-v1 problem with", sum(rewards) / 100,
              "average score in the last 100 consecutive episodes.")

    logger.log_episode(
        episode=episode,
        step=agent.curr_step,
        finish=finish
    )

    if finish:
        agent.save()
        break

    episode += 1
