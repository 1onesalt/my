"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import time
import numpy as np
import torch
from runner.shared.base_runner import Runner

# import imageio


def _t2n(x):
    return x.detach().cpu().numpy()

# 新增：用于将字典的 batch 数据转换为 tensor 输入需要的格式
def _flatten_dict_obs(obs_batch_dict):
    """
    将 buffer 中的字典数据 (Key -> [n_threads, n_agents, ...]) 
    转换为网络需要的 (Key -> [n_threads * n_agents, ...])
    """
    flat_obs = {}
    for k, v in obs_batch_dict.items():
        # 假设 v 的形状是 [n_threads, n_agents, C, H, W] 或 [n_threads, n_agents, dim]
        # 我们需要将其合并为 [n_threads * n_agents, ...]
        flat_obs[k] = np.concatenate(v) 
    return flat_obs

class EnvRunner(Runner):  #继承Runner
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def _build_share_obs(self, obs):
        """Build centralized critic input from local observations."""
        if self.use_centralized_V:
            max_agents = self.num_agents
            active_agents = max_agents
            if hasattr(self.envs, "envs") and len(self.envs.envs) > 0:
                active_agents = int(getattr(self.envs.envs[0], "active_n_agents", max_agents))
            active_agents = max(0, min(active_agents, max_agents))

            active_mask = np.zeros((self.n_rollout_threads, max_agents, 1), dtype=np.float32)
            active_mask[:, :active_agents, 0] = 1.0
            critic_tokens = np.concatenate([obs, active_mask], axis=-1)
            share_obs = critic_tokens.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        return share_obs

    def run(self):
        self.warmup()  #重置环境

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                # if self.env_name == "MPE":
                #     env_infos = {}
                #     for agent_id in range(self.num_agents):
                #         idv_rews = []
                #         for info in infos:
                #             if 'individual_reward' in info[agent_id].keys():
                #                 idv_rews.append(info[agent_id]['individual_reward'])
                #         agent_k = 'agent%i/individual_rewards' % agent_id
                #         env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.record_training_metrics(train_infos, total_num_steps)
                self.log_train(train_infos, total_num_steps)
                # self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):  #初始化观测(从reset函数中获得)和共享观测（）并将他们放入到Replay Buffer 中
        # reset env
        obs = self.envs.reset()  
       
        # # 构造 Share Obs (Critic 输入)
        # if self.use_centralized_V:
        #     # 简单策略：将所有智能体的局部观测拼接作为全局观测
        #     # 变形为 [n_threads, n_agents * obs_dim]
        #     share_obs = obs.reshape(self.n_rollout_threads, -1)
        #     # 扩展维度以适配 Buffer [n_threads, n_agents, global_dim]
        #     share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        # else:
        #     share_obs = obs

        # 构建初始 Share Obs，与每一步 insert 使用同一逻辑，避免前后不一致。
        share_obs = self._build_share_obs(obs)
        
        if isinstance(self.buffer.share_obs, dict):
             for key in self.buffer.share_obs:
                self.buffer.share_obs[key][0] = share_obs[key].copy()
        else:
            self.buffer.share_obs[0] = share_obs.copy()

        if isinstance(self.buffer.obs, dict):
             for key in self.buffer.obs:
                self.buffer.obs[key][0] = obs[key].copy()
        else:
            self.buffer.obs[0] = obs.copy()


        # #初始化缓冲区
        # self.buffer.share_obs[0] = share_obs.copy()
        # self.buffer.obs[0] = obs.copy()


    @torch.no_grad()
    def collect(self, step):   #
        self.trainer.prep_rollout()

        # obs_input = {}
        # for key, data in self.buffer.obs.items():
        #     obs_input[key] = np.concatenate(data[step]) # 结果 shape: [n_rollout * n_agents, ...]

        # share_obs_input = {}
        # for key, data in self.buffer.share_obs.items():
        #     share_obs_input[key] = np.concatenate(data[step])

        # share_obs = np.concatenate(self.buffer.share_obs[step])
        # obs = np.concatenate(self.buffer.obs[step])

        # Share Obs (From buffer)
        if isinstance(self.buffer.share_obs, dict):
            share_obs = {
                key: np.concatenate(self.buffer.share_obs[key][step]) 
                for key in self.buffer.share_obs
            }
        else:
            share_obs = np.concatenate(self.buffer.share_obs[step])
        
        # Local Obs
        if isinstance(self.buffer.obs, dict):
            obs = {
                key: np.concatenate(self.buffer.obs[key][step]) 
                for key in self.buffer.obs
            }
        else:
            obs = np.concatenate(self.buffer.obs[step])

        rnn_states_input = np.concatenate(self.buffer.rnn_states[step])
        rnn_states_critic_input = np.concatenate(self.buffer.rnn_states_critic[step])
        masks_input = np.concatenate(self.buffer.masks[step])

        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            share_obs,    # 修改点：传入字典
            obs,          # 修改点：传入字典
            rnn_states_input,
            rnn_states_critic_input,
            masks_input,
        )

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # [env_num, agent_num, 1]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))  # [env_num, agent_num, action_dim]
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )  # [env_num, agent_num, 1]
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        # rearrange action 网络输出的做一个后处理得到actions
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            # actions  --> actions_env : shape:[10, 1] --> [5, 2, 5]
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            # TODO 这里改造成自己环境需要的形式即可
            # TODO Here, you can change the shape of actions_env to fit your environment
            actions_env = actions
            # raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        if infos is not None:
            for env_i in range(self.n_rollout_threads):
                for agent_i in range(self.num_agents):
                    try:
                        is_active = bool(infos[env_i][agent_i].get("is_active", True))
                    except Exception:
                        is_active = True
                    if not is_active:
                        active_masks[env_i, agent_i, 0] = 0.0

        share_obs = self._build_share_obs(obs)


        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            active_masks=active_masks,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[
                        eval_actions[:, :, i]
                    ]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos["eval_average_episode_rewards"])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
