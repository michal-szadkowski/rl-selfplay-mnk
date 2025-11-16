import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from ..weight_init import initialize_actor_critic_weights


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


class BaseResNetActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, channels=64, num_blocks=4):
        super().__init__()
        self.action_dim = action_dim

        self.conv_in = nn.Sequential(
            nn.Conv2d(obs_shape[0], channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Build res_blocks with configurable number of blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(ResidualBlock(channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        actor_flattened_size = 2 * obs_shape[1] * obs_shape[2]
        critic_flattened_size = 1 * obs_shape[1] * obs_shape[2]

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(actor_flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(critic_flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        initialize_actor_critic_weights(self)

    def forward_body(self, x):
        x = self.conv_in(x)
        features = self.res_blocks(x)
        return features

    def forward(self, obs, action_mask=None):
        features = self.forward_body(obs)

        logits = self.policy_head(features)
        value = self.value_head(features)

        if action_mask is not None:
            if action_mask.dim() == 1 and logits.dim() == 2:
                action_mask = action_mask.unsqueeze(0)

            logits[~action_mask] = -torch.inf

            all_invalid = action_mask.sum(dim=-1) == 0
            if all_invalid.any():
                logits[all_invalid] = 0.0

        dist = Categorical(logits=logits)
        return dist, value


class ResNetSActorCritic(BaseResNetActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, channels=64, num_blocks=4)
        self._architecture_name = "resnet_s"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }


class ResNetLActorCritic(BaseResNetActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, channels=128, num_blocks=6)
        self._architecture_name = "resnet_l"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }
