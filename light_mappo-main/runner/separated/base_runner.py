import time
import os
import csv
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from utils.separated_buffer import SeparatedReplayBuffer
from utils.util import update_linear_schedule


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio

            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / "gifs")
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            # if self.use_wandb:
            #     self.save_dir = str(wandb.run.dir)
            # else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / "logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.plot_dir = str(self.run_dir / "plots")
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)
            self.save_dir = str(self.run_dir / "models")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.enable_training_plots = not getattr(self.all_args, "disable_training_plots", False)
            self.plot_smooth_window = max(1, int(getattr(self.all_args, "plot_smooth_window", 5)))
            self.training_history = {
                "step": [],
                "average_episode_rewards": [],
                "value_loss": [],
                "policy_loss": [],
                "dist_entropy": [],
            }

        from algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            # policy network
            po = Policy(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            # buffer
            share_observation_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            bu = SeparatedReplayBuffer(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
            )
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(
                self.buffer[agent_id].share_obs[-1],
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1],
            )
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
            )
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(
                policy_critic.state_dict(),
                str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt",
            )

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor_agent" + str(agent_id) + ".pt")
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + "/critic_agent" + str(agent_id) + ".pt"
            )
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                # if self.use_wandb:
                #     pass
                # wandb.log({agent_k: v}, step=total_num_steps)
                # else:
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                # if self.use_wandb:
                #     wandb.log({k: np.mean(v)}, step=total_num_steps)
                # else:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def record_training_metrics(self, train_infos, total_num_steps, avg_episode_reward):
        """Cache averaged multi-agent reward/loss metrics for plotting."""
        self.training_history["step"].append(int(total_num_steps))
        self.training_history["average_episode_rewards"].append(float(avg_episode_reward))

        for key in ("value_loss", "policy_loss", "dist_entropy"):
            values = []
            for agent_info in train_infos:
                if key in agent_info:
                    try:
                        values.append(float(agent_info[key]))
                    except Exception:
                        pass
            self.training_history[key].append(float(np.mean(values)) if len(values) > 0 else np.nan)

    @staticmethod
    def _moving_average(values, window):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        window = max(1, int(window))
        valid = ~np.isnan(arr)
        filled = np.where(valid, arr, 0.0)
        kernel = np.ones(window, dtype=np.float32)
        num = np.convolve(filled, kernel, mode="same")
        den = np.convolve(valid.astype(np.float32), kernel, mode="same")
        if num.size != arr.size:
            start = (num.size - arr.size) // 2
            num = num[start:start + arr.size]
            den = den[start:start + arr.size]
        den = np.maximum(den, 1.0)
        smoothed = num / den
        smoothed[~valid] = np.nan
        return smoothed

    def plot_training_curves(self):
        """Export reward/loss curves and cached metric csv after training."""
        if self.use_render:
            return
        if not self.enable_training_plots:
            print("Skip training plots: disabled by --disable_training_plots.")
            return

        steps = np.asarray(self.training_history["step"], dtype=np.int64)
        if steps.size == 0:
            print("Skip training plots: no recorded metrics.")
            return

        history_csv_path = os.path.join(self.plot_dir, "training_metrics.csv")
        fieldnames = ["step", "average_episode_rewards", "value_loss", "policy_loss", "dist_entropy"]
        with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(steps)):
                writer.writerow(
                    {
                        "step": int(steps[i]),
                        "average_episode_rewards": self.training_history["average_episode_rewards"][i],
                        "value_loss": self.training_history["value_loss"][i],
                        "policy_loss": self.training_history["policy_loss"][i],
                        "dist_entropy": self.training_history["dist_entropy"][i],
                    }
                )

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Skip training curve plotting: matplotlib unavailable ({e}).")
            print(f"Metrics csv still saved to: {history_csv_path}")
            return

        reward = np.asarray(self.training_history["average_episode_rewards"], dtype=np.float32)
        value_loss = np.asarray(self.training_history["value_loss"], dtype=np.float32)
        policy_loss = np.asarray(self.training_history["policy_loss"], dtype=np.float32)
        entropy = np.asarray(self.training_history["dist_entropy"], dtype=np.float32)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=120, sharex=True)
        axes[0].plot(steps, reward, color="#1f77b4", alpha=0.35, label="Reward raw")
        axes[0].plot(steps, self._moving_average(reward, self.plot_smooth_window), color="#1f77b4", linewidth=2.0, label=f"Reward MA({self.plot_smooth_window})")
        axes[0].set_ylabel("Average Episode Reward")
        axes[0].set_title("Training Reward Curve")
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(steps, value_loss, color="#d62728", alpha=0.4, label="Value loss")
        axes[1].plot(steps, policy_loss, color="#2ca02c", alpha=0.4, label="Policy loss")
        axes[1].plot(steps, entropy, color="#9467bd", alpha=0.4, label="Entropy")
        axes[1].plot(steps, self._moving_average(value_loss, self.plot_smooth_window), color="#d62728", linewidth=1.8, label=f"Value MA({self.plot_smooth_window})")
        axes[1].plot(steps, self._moving_average(policy_loss, self.plot_smooth_window), color="#2ca02c", linewidth=1.8, label=f"Policy MA({self.plot_smooth_window})")
        axes[1].set_xlabel("Environment Steps")
        axes[1].set_ylabel("Loss / Entropy")
        axes[1].set_title("Network Loss Indicators")
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="best")

        fig.tight_layout()
        fig_path = os.path.join(self.plot_dir, "training_curves.png")
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Saved training curves to: {fig_path}")
        print(f"Saved training metrics csv to: {history_csv_path}")
