import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class SimpleResNetActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        c, h, w = obs_shape
        self.action_dim = action_dim

        self._architecture_name = "simple_resnet_actor_critic"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }

        self.conv_in = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            flat_size = self.forward_features(dummy).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                if module.out_features == 1:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                elif module.out_features == self.action_dim:
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x):
        """Extract features from observation, returns flattened features."""
        x = self.conv_in(x)
        x = self.res_blocks(x)
        x = self.flatten(x)
        return x

    def forward(self, obs, action_mask=None):
        features = self.forward_features(obs)
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

  