import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from src.alg.rollout_buffer import RolloutBuffer


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=512):
        super().__init__()
        # obs_shape is expected to be (channels, height, width), e.g., (2, 9, 9)
        channels, m, n = obs_shape

        self.shared_body = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1),
            nn.LayerNorm([64, m, n]),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LayerNorm([128, m, n]),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.LayerNorm([128, m, n]),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            flattened_size = self.shared_body(dummy_input).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

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


class A2CAgent:
    def __init__(self, obs_shape, action_dim, network: ActorCritic, n_steps: int, learning_rate=7e-4, gamma=0.99, batch_size=64,
                 device='cpu', num_envs=1):
        """
        Initializes the A2C agent with a batch size for updates.

        Args:
            network: The ActorCritic network.
            n_steps (int): Total size of the rollout buffer.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            batch_size (int): Size of the mini-batches for network updates.
            device (str): The device to run the calculations on.
            num_envs (int): The number of parallel environments.
        """
        self.device = device
        self.network = network.to(self.device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate)

        self.buffer = RolloutBuffer(self.n_steps, self.num_envs, obs_shape, action_dim, device=self.device)

    def learn(self, vec_env):
        """
        The main training loop for the A2C agent, collecting rollouts from a
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

        self.buffer.compute_advantages_and_returns(last_values, self.gamma)

        # 3. Update networks using the full rollout
        actor_loss, critic_loss, entropy_loss = self.update_networks()

        # 4. Reset the buffer for the next rollout
        self.buffer.reset()

        if len(ep_rewards) > 0:
            return np.mean(ep_rewards), np.mean(ep_lengths), actor_loss, critic_loss, entropy_loss
        return 0.0, actor_loss, critic_loss, entropy_loss

    def update_networks(self):
        """
        Updates the Actor and Critic networks by iterating through mini-batches
        from the rollout buffer.
        """
        # Get a DataLoader that will provide mini-batches
        data_loader = self.buffer.get_data_loader(self.batch_size)

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        # Iterate over mini-batches
        for observations, actions, old_log_probs, returns, advantages, action_masks in data_loader:
            dist, values = self.network(observations, action_masks)

            # Actor loss
            log_probs = dist.log_prob(actions.squeeze())
            actor_loss = -(log_probs * advantages).mean()

            # Critic loss
            critic_loss = F.mse_loss(values.squeeze(), returns)

            # Entropy loss to encourage exploration
            entropy_loss = -dist.entropy().mean()

            # Total loss and backpropagation
            total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_losses)
