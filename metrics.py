import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from collections import deque


class MetricLogger:
    def __init__(self, save_dir):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_loss_length = 0
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'MeanReward':>15}"
                f"{'MeanLoss':>15}"
                f"{'TimeDelta':>15}{'Time':>25}\n"
            )
        self.rewards_plot = save_dir / "reward_plot.jpg"
        self.avg_losses_plot = save_dir / "loss_plot.jpg"

        # History metrics
        self.avg_losses = []
        self.rewards = []
        self.avg_rewards = deque(maxlen=100)
        self.avg_loss = deque(maxlen=100)

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_loss_length += 1

    def log_episode(self, episode, step, finish):

        """Mark end of episode"""

        ep_avg_loss = self.curr_ep_loss / (self.curr_ep_loss_length + 1)

        self.avg_loss.append(ep_avg_loss)
        self.avg_rewards.append(self.curr_ep_reward)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}"
                f"{self.curr_ep_reward:15.3f}{ep_avg_loss:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%d ~ %H:%M:%S'):>25}\n"
            )

            self.rewards.append(sum(self.avg_rewards)/len(self.avg_rewards))
            self.avg_losses.append(sum(self.avg_loss)/len(self.avg_loss))

        if episode % 100 == 0 or finish:
            for metric in ["rewards", "avg_losses"]:
                plt.plot(getattr(self, f"{metric}"))
                plt.savefig(getattr(self, f"{metric}_plot"))
                plt.clf()

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_loss_length = 0
