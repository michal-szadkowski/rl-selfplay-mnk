import torch
from torch.utils.data import DataLoader, TensorDataset


class RolloutBuffer:
    def __init__(self, n_steps, num_envs, obs_shape, action_dim, device='cpu'):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self):
        """Resetuje bufor."""
        self.observations = torch.zeros((self.n_steps, self.num_envs, *self.obs_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.n_steps, self.num_envs), dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros((self.n_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.n_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.n_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((self.n_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((self.n_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.n_steps, self.num_envs), dtype=torch.bool, device=self.device)
        self.action_masks = torch.zeros((self.n_steps, self.num_envs, self.action_dim), dtype=torch.bool, device=self.device)
        self.ptr = 0

    def add(self, obs, action, reward, value, log_prob, done, action_mask):
        """Dodaje batch przejść z jednego kroku czasowego (wektorowe środowiska)."""
        if self.ptr >= self.n_steps:
            raise IndexError("Buffer is full.")

        self.observations[self.ptr].copy_(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        self.actions[self.ptr].copy_(torch.as_tensor(action, dtype=torch.long, device=self.device))
        self.rewards[self.ptr].copy_(torch.as_tensor(reward, dtype=torch.float32, device=self.device))
        self.values[self.ptr].copy_(torch.as_tensor(value, dtype=torch.float32, device=self.device).view(-1))
        self.log_probs[self.ptr].copy_(torch.as_tensor(log_prob, dtype=torch.float32, device=self.device))
        self.dones[self.ptr].copy_(torch.as_tensor(done, dtype=torch.bool, device=self.device))
        self.action_masks[self.ptr].copy_(torch.as_tensor(action_mask, dtype=torch.bool, device=self.device))
        self.ptr += 1

    def compute_advantages_and_returns(self, last_values, gamma=0.99, gae_lambda=0.95):
        """
        Oblicza GAE-Lambda i returns.
        Zakładamy:
        - self.values[t] = V(s_t)
        - self.dones[t] = czy epizod zakończył się po kroku t (czyli czy s_{t+1} jest terminalne)
        - last_values = V(s_T) dla ostatnich stanów (po rollout).
        """
        last_values = torch.as_tensor(last_values, dtype=self.values.dtype,
                                      device=self.device).reshape(self.num_envs)

        steps = self.ptr if self.ptr > 0 else self.n_steps
        advantages = torch.zeros((steps, self.num_envs),
                                 dtype=self.rewards.dtype, device=self.device)
        last_gae = torch.zeros(self.num_envs, dtype=self.rewards.dtype, device=self.device)

        for t in reversed(range(steps)):
            if t == steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            # ważne: next_non_terminal odpowiada "czy s_{t+1} nie jest terminalne"
            next_non_terminal = 1.0 - self.dones[t].to(dtype=self.rewards.dtype)

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages[:steps] = advantages
        self.returns[:steps] = advantages + self.values[:steps]

    def get_data_loader(self, batch_size, normalize_advantages=True):
        steps = self.ptr if self.ptr > 0 else self.n_steps
        num_samples = steps * self.num_envs

        # More memory-efficient reshaping
        observations = self.observations[:steps].view(num_samples, *self.obs_shape)
        actions = self.actions[:steps].view(num_samples)
        log_probs = self.log_probs[:steps].view(num_samples)
        returns = self.returns[:steps].view(num_samples)
        advantages = self.advantages[:steps].view(num_samples)
        action_masks = self.action_masks[:steps].view(num_samples, self.action_dim)

        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(
            observations,
            actions,
            log_probs,
            returns,
            advantages,
            action_masks
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
