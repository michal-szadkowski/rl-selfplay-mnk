import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
from dataclasses import dataclass
import numpy as np

from .rollout_buffer import RolloutBuffer


@dataclass
class TrainingMetrics:
    """Training metrics from a single PPO learning iteration."""

    mean_reward: float
    mean_length: float
    actor_loss: float
    critic_loss: float
    entropy_loss: float
    grad_norm: float
    clip_fraction: float


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
        device="cpu",
        num_envs=1,
        lr_schedule=None,
    ):
        """
        Initializes the PPO agent.

        Args:
            network: The ActorCritic network.
            n_steps (int): Total size of the rollout buffer.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): GAE lambda parameter for advantage estimation.
            clip_range (float): PPO clip range for policy updates.
            ppo_epochs (int): Number of PPO epochs per rollout.
            batch_size (int): Size of the mini-batches for network updates.
            value_coef (float): Value loss coefficient.
            entropy_coef (float): Entropy bonus coefficient.
            device (str): The device to run the calculations on.
            num_envs (int): The number of parallel environments.
        """
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
        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=learning_rate, weight_decay=1e-4
        )

        self.buffer = RolloutBuffer(
            self.n_steps, self.num_envs, obs_shape, action_dim, device=self.device
        )

        # Initialize LR scheduler if warmup is configured
        self.lr_scheduler = None
        if lr_schedule:
            self.lr_scheduler = self._create_lr_scheduler(lr_schedule)

    def _create_lr_scheduler(self, lr_schedule):
        """Create learning rate scheduler with warmup."""
        warmup_steps = lr_schedule.get("warmup_steps", 0)

        if warmup_steps > 0:
            # Convert environment steps to iterations
            steps_per_iteration = self.num_envs * self.n_steps
            warmup_iterations = max(1, warmup_steps // steps_per_iteration)

            # Warmup phase: linear from 0 to target LR
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_iterations,
            )
            # Main training phase: constant LR
            main = ConstantLR(self.optimizer, factor=1.0)

            return SequentialLR(
                self.optimizer, schedulers=[warmup, main], milestones=[warmup_iterations]
            )
        return None

    def learn(self, vec_env):
        """
        The main training loop for the PPO agent, collecting rollouts from a
        vectorized environment.

        Returns:
            TrainingMetrics: Object containing all training metrics
        """
        obs, _ = vec_env.reset()
        ep_rewards = []
        ep_lengths = []

        for _ in range(self.buffer.n_steps):
            # 1. Collect one step of the rollout from all parallel environments
            observation = obs["observation"]
            action_mask = torch.as_tensor(
                obs["action_mask"], dtype=torch.bool, device=self.device
            )

            with torch.no_grad():
                dist, values = self.network(
                    torch.as_tensor(observation, dtype=torch.float32, device=self.device),
                    action_mask,
                )
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(
                actions.cpu().numpy()
            )
            dones = terminateds | truncateds

            self.buffer.add(
                observation, actions, rewards, values, log_probs, dones, action_mask
            )

            obs = next_obs

            if "_episode" in infos:
                for i, done_flag in enumerate(infos["_episode"]):
                    if done_flag:
                        r = infos["episode"]["r"][i]
                        l = infos["episode"]["l"][i]
                        t = infos["episode"]["t"][i]
                        ep_rewards.append(r)
                        ep_lengths.append(l)

        # 2. Compute advantages and returns for the full rollout
        with torch.no_grad():
            # Get the value of the last observation in each environment
            _, last_values_tensor = self.network(
                torch.as_tensor(obs["observation"], dtype=torch.float32, device=self.device)
            )
            last_values = last_values_tensor.view(self.num_envs)

        self.buffer.compute_advantages_and_returns(last_values, self.gamma, self.gae_lambda)

        # 3. Update networks using the full rollout
        actor_loss, critic_loss, entropy_loss, grad_norm, clip_fraction = (
            self.update_networks()
        )

        # 4. Step the learning rate scheduler
        if self.lr_scheduler:
            self.lr_scheduler.step()

        # 5. Reset the buffer for the next rollout
        self.buffer.reset()

        # Create and return TrainingMetrics object
        if len(ep_rewards) > 0:
            return TrainingMetrics(
                mean_reward=np.mean(ep_rewards),
                mean_length=np.mean(ep_lengths),
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                entropy_loss=entropy_loss,
                grad_norm=grad_norm,
                clip_fraction=clip_fraction,
            )
        return TrainingMetrics(
            mean_reward=0.0,
            mean_length=0.0,
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            entropy_loss=entropy_loss,
            grad_norm=grad_norm,
            clip_fraction=clip_fraction,
        )

    def update_networks(self):
        """
        Updates the Actor and Critic networks using PPO algorithm with multiple epochs
        and clipped objective function.
        """
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        grad_norms = []
        clip_fractions = []

        # Perform multiple PPO epochs on the same rollout data
        for epoch in range(self.ppo_epochs):
            # Get a DataLoader that will provide mini-batches (shuffled each epoch)
            data_loader = self.buffer.get_data_loader(self.batch_size)

            # Iterate over mini-batches in this epoch
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

                # Calculate new log probabilities and entropy
                new_log_probs = dist.log_prob(actions.squeeze())
                entropy = dist.entropy().mean()

                # Calculate ratio (pi_theta / pi_theta_old)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # PPO Clipped Objective
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Calculate clip fraction (percentage of clips)
                clip_fraction = (torch.abs(ratio - 1.0) > self.clip_range).float().mean()
                clip_fractions.append(clip_fraction.item())

                # Critic loss (value function loss)
                critic_loss = F.mse_loss(values.squeeze(), returns)

                # Entropy bonus
                entropy_loss = -entropy

                # Total loss with coefficients
                total_loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), max_norm=0.5
                )
                grad_norms.append(grad_norm.item())

                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())

        return (
            np.mean(actor_losses),
            np.mean(critic_losses),
            np.mean(entropy_losses),
            np.mean(grad_norms),
            np.mean(clip_fractions),
        )
