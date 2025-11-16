import torch
import torch.nn as nn
from torch.distributions import Categorical
from .weight_init import initialize_actor_critic_weights


class CnnConv1x1ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        channels, h, w = obs_shape
        self.action_dim = action_dim

        self._architecture_name = "cnn_conv1x1"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }

        self.shared_body = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        actor_flattened_size = 2 * h * w
        critic_flattened_size = 1 * h * w

        self.actor = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(actor_flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(critic_flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        initialize_actor_critic_weights(self)

    def forward(self, obs, action_mask=None):
        features = self.shared_body(obs)

        # Actor and critic paths
        logits = self.actor(features)
        value = self.critic(features)

        if action_mask is not None:
            # ensure mask has correct dimensions
            if action_mask.dim() == 1 and logits.dim() == 2:
                action_mask = action_mask.unsqueeze(0)

            logits[~action_mask] = -torch.inf

            all_invalid = action_mask.sum(dim=-1) == 0
            if all_invalid.any():
                logits[all_invalid] = 0.0

        dist = Categorical(logits=logits)

        return dist, value
