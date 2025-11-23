import torch
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import time

from .rollout_buffer import RolloutBuffer


@dataclass
class TrainingMetrics:
    mean_reward: float
    mean_length: float
    actor_loss: float
    critic_loss: float
    entropy_loss: float
    grad_norm: float
    clip_fraction: float
    explained_variance: float
    approx_kl: float
    fps: float
    rollout_time: float
    learn_time: float


class PPOAgent:
    def __init__(
        self,
        obs_shape,
        action_dim,
        network,
        n_steps: int,
        learning_rate=7e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ppo_epochs=4,
        batch_size=64,
        value_coef=0.5,
        entropy_coef=0.01,
        device="cuda",
        num_envs=1,
        lr_scheduler=None,
        entropy_scheduler=None,
        optimizer=None,
    ):
        self.device = device
        self.network = network.to(self.device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.num_envs = num_envs
        self.n_steps = n_steps

        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.network.parameters(), lr=learning_rate, weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer

        self.buffer = RolloutBuffer(
            self.n_steps, self.num_envs, obs_shape, action_dim, device=self.device
        )

        self.lr_scheduler = lr_scheduler
        self.entropy_scheduler = entropy_scheduler

    def learn(self, vec_env):
        rollout_start_time = time.time()

        obs, _ = vec_env.reset()

        current_ep_reward = torch.zeros(self.num_envs, device=self.device)
        current_ep_len = torch.zeros(self.num_envs, device=self.device)

        finished_ep_rewards = []
        finished_ep_lengths = []

        for _ in range(self.buffer.n_steps):
            observation = obs["observation"]
            action_mask = obs["action_mask"]

            with torch.no_grad():
                dist, values = self.network(observation, action_mask)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            next_obs, rewards, terminateds, truncateds, _ = vec_env.step(actions)

            dones = terminateds | truncateds

            self.buffer.add(
                observation, actions, rewards, values, log_probs, dones, action_mask
            )

            current_ep_reward += rewards
            current_ep_len += 1

            if dones.any():
                done_indices = torch.nonzero(dones).squeeze(1)

                finished_ep_rewards.extend(current_ep_reward[done_indices].tolist())
                finished_ep_lengths.extend(current_ep_len[done_indices].tolist())

                current_ep_reward[done_indices] = 0
                current_ep_len[done_indices] = 0

            obs = next_obs

        rollout_end_time = time.time()
        rollout_time = rollout_end_time - rollout_start_time
        total_steps = self.buffer.n_steps * self.num_envs
        fps = total_steps / rollout_time if rollout_time > 0 else 0.0

        with torch.no_grad():
            _, last_values_tensor = self.network(obs["observation"], obs["action_mask"])
            last_values = last_values_tensor.reshape(self.num_envs)

        self.buffer.compute_advantages_and_returns(last_values, self.gamma, self.gae_lambda)

        learn_start_time = time.time()
        metrics = self.update_networks()
        learn_end_time = time.time()
        learn_time = learn_end_time - learn_start_time

        if self.lr_scheduler:
            self.lr_scheduler.step()
        if self.entropy_scheduler:
            self.entropy_scheduler.step()
            self.entropy_coef = self.entropy_scheduler.get_last_coef()

        self.buffer.reset()

        mean_reward = np.mean(finished_ep_rewards) if finished_ep_rewards else 0.0
        mean_length = np.mean(finished_ep_lengths) if finished_ep_lengths else 0.0

        return TrainingMetrics(
            mean_reward=mean_reward,
            mean_length=mean_length,
            actor_loss=metrics[0],
            critic_loss=metrics[1],
            entropy_loss=metrics[2],
            grad_norm=metrics[3],
            clip_fraction=metrics[4],
            explained_variance=metrics[5],
            approx_kl=metrics[6],
            fps=fps,
            rollout_time=rollout_time,
            learn_time=learn_time,
        )

    def update_networks(self):

        total_actor_loss = torch.tensor(0.0, device=self.device)
        total_critic_loss = torch.tensor(0.0, device=self.device)
        total_entropy_loss = torch.tensor(0.0, device=self.device)
        total_grad_norm = torch.tensor(0.0, device=self.device)
        total_clip_frac = torch.tensor(0.0, device=self.device)
        total_explained_var = torch.tensor(0.0, device=self.device)
        total_approx_kl = torch.tensor(0.0, device=self.device)

        updates_count = 0

        for epoch in range(self.ppo_epochs):
            data_loader = self.buffer.get_data_loader(self.batch_size)

            for (
                observations,
                actions,
                old_log_probs,
                returns,
                advantages,
                action_masks,
                old_values,
            ) in data_loader:
                dist, values = self.network(observations, action_masks)
                values = values.squeeze()

                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(values, returns)
                entropy_loss = -entropy

                total_loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), max_norm=0.5
                )

                self.optimizer.step()

                with torch.no_grad():
                    updates_count += 1
                    total_actor_loss += actor_loss.detach()
                    total_critic_loss += critic_loss.detach()
                    total_entropy_loss += entropy_loss.detach()
                    total_grad_norm += grad_norm

                    clip_frac = (torch.abs(ratio - 1.0) > self.clip_range).float().mean()
                    total_clip_frac += clip_frac

                    log_ratio = new_log_probs - old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    total_approx_kl += approx_kl

                    returns_var = returns.var()
                    if returns_var > 1e-8:
                        explained_var = 1 - F.mse_loss(values, returns) / returns_var
                    else:
                        explained_var = torch.tensor(0.0, device=self.device)
                    total_explained_var += explained_var

        return (
            (total_actor_loss / updates_count).item(),
            (total_critic_loss / updates_count).item(),
            (total_entropy_loss / updates_count).item(),
            (total_grad_norm / updates_count).item(),
            (total_clip_frac / updates_count).item(),
            (total_explained_var / updates_count).item(),
            (total_approx_kl / updates_count).item(),
        )
