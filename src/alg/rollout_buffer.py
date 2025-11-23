import torch


class RolloutBuffer:
    def __init__(self, n_steps, num_envs, obs_shape, action_dim, device="cpu"):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self):
        self.observations = torch.zeros(
            (self.n_steps, self.num_envs, *self.obs_shape),
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.zeros(
            (self.n_steps, self.num_envs), dtype=torch.long, device=self.device
        )
        self.log_probs = torch.zeros(
            (self.n_steps, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.rewards = torch.zeros(
            (self.n_steps, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.values = torch.zeros(
            (self.n_steps, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.returns = torch.zeros(
            (self.n_steps, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.advantages = torch.zeros(
            (self.n_steps, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.dones = torch.zeros(
            (self.n_steps, self.num_envs), dtype=torch.bool, device=self.device
        )
        self.action_masks = torch.zeros(
            (self.n_steps, self.num_envs, self.action_dim),
            dtype=torch.bool,
            device=self.device,
        )
        self.ptr = 0

    def add(self, obs, action, reward, value, log_prob, done, action_mask):
        if self.ptr >= self.n_steps:
            raise IndexError("Buffer was full.")

        self.observations[self.ptr].copy_(obs)
        self.actions[self.ptr].copy_(action)
        self.rewards[self.ptr].copy_(reward)
        self.values[self.ptr].copy_(value.view(-1))
        self.log_probs[self.ptr].copy_(log_prob)
        self.dones[self.ptr].copy_(done)
        self.action_masks[self.ptr].copy_(action_mask)
        self.ptr += 1

    def compute_advantages_and_returns(self, last_values, gamma=0.99, gae_lambda=0.95):
        last_values = last_values.reshape(self.num_envs)

        steps = self.ptr

        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for t in reversed(range(steps)):
            if t == steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t].float()

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

            self.advantages[t] = last_gae

        self.returns[:steps] = self.advantages[:steps] + self.values[:steps]

    def get_data_loader(self, batch_size, normalize_advantages=True):
        steps = self.ptr
        num_samples = steps * self.num_envs

        b_obs = self.observations[:steps].view(num_samples, *self.obs_shape)
        b_actions = self.actions[:steps].view(num_samples)
        b_log_probs = self.log_probs[:steps].view(num_samples)
        b_returns = self.returns[:steps].view(num_samples)
        b_advantages = self.advantages[:steps].view(num_samples)
        b_values = self.values[:steps].view(num_samples)
        b_masks = self.action_masks[:steps].view(num_samples, self.action_dim)

        if normalize_advantages:
            adv_mean = b_advantages.mean()
            adv_std = b_advantages.std()
            b_advantages = (b_advantages - adv_mean) / (adv_std + 1e-8)

        indices = torch.randperm(num_samples, device=self.device)

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield (
                b_obs[batch_idx],
                b_actions[batch_idx],
                b_log_probs[batch_idx],
                b_returns[batch_idx],
                b_advantages[batch_idx],
                b_masks[batch_idx],
                b_values[batch_idx],
            )
