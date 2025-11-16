import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .weight_init import initialize_actor_critic_weights


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out, inplace=True)


class ResNetConv1x1ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        c, h, w = obs_shape
        self.action_dim = action_dim

        self._architecture_name = "resnet_conv1x1"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }

        self.conv_in = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        actor_flattened_size = 2 * h * w
        critic_flattened_size = 1 * h * w

        self.actor = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Flatten(),
            nn.BatchNorm1d(actor_flattened_size),
            nn.Linear(actor_flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Flatten(),
            nn.BatchNorm1d(critic_flattened_size),
            nn.Linear(critic_flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        initialize_actor_critic_weights(self)

    def forward_features(self, x):
        """Extract features from observation, returns shared features."""
        x = self.conv_in(x)
        x = self.res_blocks(x)
        return x

    def forward(self, obs, action_mask=None):
        features = self.forward_features(obs)

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
