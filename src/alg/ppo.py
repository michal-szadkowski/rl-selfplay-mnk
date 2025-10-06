import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from .rollout_buffer import RolloutBuffer


class ActorCriticModule(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        # obs_shape is expected to be (channels, height, width), e.g., (2, 9, 9)
        channels, m, n = obs_shape

        # Store action_dim for use in initialization
        self.action_dim = action_dim

        # Convolutional body
        self.shared_body = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            flattened_size = self.shared_body(dummy_input).shape[1]

        # Actor and critic heads
        self.actor = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                if module.out_features == 1:
                    nn.init.orthogonal_(module.weight, gain=1.0)

                elif module.out_features == self.action_dim:
                    nn.init.orthogonal_(module.weight, gain=0.01)

                else:
                    nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, obs, action_mask=None):
        features = self.shared_body(obs)
        logits = self.actor(features)

        if action_mask is not None:
            # upewniamy się, że maska ma odpowiedni wymiar
            if action_mask.dim() == 1 and logits.dim() == 2:
                action_mask = action_mask.unsqueeze(0)

            logits = logits.clone()
            logits[~action_mask] = -torch.inf

            all_invalid = action_mask.sum(dim=-1) == 0
            if all_invalid.any():
                logits[all_invalid] = 0.0

        dist = Categorical(logits=logits)
        value = self.critic(features)
        return dist, value


class PPOAgent:
    def __init__(self,
                 obs_shape,
                 action_dim,
                 network,
                 n_steps: int,
                 learning_rate=7e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_range=0.3,
                 ppo_epochs=4,
                 batch_size=64,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 device='cpu',
                 num_envs=1):
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
        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate)

        self.buffer = RolloutBuffer(self.n_steps, self.num_envs, obs_shape, action_dim, device=self.device)

    def learn(self, vec_env):
        """
        The main training loop for the PPO agent, collecting rollouts from a
        vectorized environment.
        """
        obs, _ = vec_env.reset()
        ep_rewards = []
        ep_lengths = []

        for _ in range(self.buffer.n_steps):
            # 1. Collect one step of the rollout from all parallel environments
            observation = obs["observation"]
            action_mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=self.device)

            with torch.no_grad():
                dist, values = self.network(torch.as_tensor(observation, dtype=torch.float32, device=self.device), action_mask)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            next_obs, rewards, terminateds, truncateds, infos = vec_env.step(actions.cpu().numpy())
            dones = terminateds | truncateds

            self.buffer.add(observation, actions, rewards, values, log_probs, dones, action_mask)

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
            _, last_values_tensor = self.network(torch.as_tensor(obs["observation"], dtype=torch.float32, device=self.device))
            last_values = last_values_tensor.view(self.num_envs)

        self.buffer.compute_advantages_and_returns(last_values, self.gamma, self.gae_lambda)

        # 3. Update networks using the full rollout
        actor_loss, critic_loss, entropy_loss = self.update_networks()

        # 4. Reset the buffer for the next rollout
        self.buffer.reset()

        if len(ep_rewards) > 0:
            return np.mean(ep_rewards), np.mean(ep_lengths), actor_loss, critic_loss, entropy_loss
        return 0.0, 0.0, actor_loss, critic_loss, entropy_loss

    def update_networks(self):
        """
        Updates the Actor and Critic networks using PPO algorithm with multiple epochs
        and clipped objective function.
        """
        actor_losses = []
        critic_losses = []
        entropy_losses = []

        # Perform multiple PPO epochs on the same rollout data
        for epoch in range(self.ppo_epochs):
            # Get a DataLoader that will provide mini-batches (shuffled each epoch)
            data_loader = self.buffer.get_data_loader(self.batch_size)

            # Iterate over mini-batches in this epoch
            for observations, actions, old_log_probs, returns, advantages, action_masks, old_values in data_loader:
                dist, values = self.network(observations, action_masks)

                # Calculate new log probabilities and entropy
                new_log_probs = dist.log_prob(actions.squeeze())
                entropy = dist.entropy().mean()

                # Calculate ratio (pi_theta / pi_theta_old)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # PPO Clipped Objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss (value function loss)
                critic_loss = F.mse_loss(values.squeeze(), returns)

                # Entropy bonus
                entropy_loss = -entropy

                # Total loss with coefficients
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_losses)
