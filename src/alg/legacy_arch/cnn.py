import torch
import torch.nn as nn
from torch.distributions import Categorical
from ..weight_init import initialize_actor_critic_weights


class CnnActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        # obs_shape is expected to be (channels, height, width), e.g., (2, 9, 9)
        channels, m, n = obs_shape

        # Store action_dim for use in initialization
        self.action_dim = action_dim

        # Architecture info for model export
        self._architecture_name = "cnn"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }

        # Convolutional body
        self.shared_body = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            flattened_size = self.shared_body(dummy_input).shape[1]

        # Actor and critic heads
        self.actor = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        initialize_actor_critic_weights(self)

    def forward(self, obs, action_mask=None):
        features = self.shared_body(obs)
        logits = self.actor(features)

        if action_mask is not None:
            # ensure mask has correct dimensions
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
