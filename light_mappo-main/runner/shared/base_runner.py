import os
import csv
import numpy as np
import torch
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V  #是否使用集中式价值函数
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
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

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.plot_dir = str(self.run_dir / "plots")
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.save_dir = str(self.run_dir / 'models')
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

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy = Policy(self.all_args,                 #实例化policy
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm没有模型路径就实例化一个trainer
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)  #里面应该是计算路径、反向传播等内容
        
        # buffer存数据
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def record_training_metrics(self, train_infos, total_num_steps):
        """Cache reward/loss metrics for post-training visualization."""
        self.training_history["step"].append(int(total_num_steps))
        for key in ("average_episode_rewards", "value_loss", "policy_loss", "dist_entropy"):
            value = train_infos.get(key, np.nan)
            try:
                value = float(np.mean(value))
            except Exception:
                value = np.nan
            self.training_history[key].append(value)

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
        if not self.enable_training_plots:
            print("Skip training plots: disabled by --disable_training_plots.")
            return

        steps = np.asarray(self.training_history["step"], dtype=np.int64)
        if steps.size == 0:
            print("Skip training plots: no recorded metrics.")
            print("  可能原因: 1) num_env_steps 过小导致 episode=0  2) log_interval 过大 3) 训练在首次 log 前中断")
            print(f"  绘图将保存到: {self.plot_dir}")
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
        reward_ma = self._moving_average(reward, self.plot_smooth_window)
        axes[0].plot(steps, reward_ma, color="#1f77b4", linewidth=2.0, label=f"Reward MA({self.plot_smooth_window})")
        axes[0].set_ylabel("Average Episode Reward")
        axes[0].set_title("Training Reward Curve")
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(steps, value_loss, color="#d62728", alpha=0.4, label="Value loss")
        axes[1].plot(steps, policy_loss, color="#2ca02c", alpha=0.4, label="Policy loss")
        axes[1].plot(steps, entropy, color="#9467bd", alpha=0.4, label="Entropy")
        axes[1].plot(
            steps,
            self._moving_average(value_loss, self.plot_smooth_window),
            color="#d62728",
            linewidth=1.8,
            label=f"Value MA({self.plot_smooth_window})",
        )
        axes[1].plot(
            steps,
            self._moving_average(policy_loss, self.plot_smooth_window),
            color="#2ca02c",
            linewidth=1.8,
            label=f"Policy MA({self.plot_smooth_window})",
        )
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
