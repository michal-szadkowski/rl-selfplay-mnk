import torch
import torch.nn as nn
from torch.distributions import Categorical
from ..weight_init import initialize_actor_critic_weights


class BaseCnnActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, channels=[64, 64, 64]):
        super().__init__()
        self.action_dim = action_dim

        # Build shared body with configurable channels
        layers = []
        in_channels = obs_shape[0]

        for i, out_channels in enumerate(channels):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else channels[i - 1],
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        self.shared_body = nn.Sequential(*layers)

        # Actor and critic heads using 1x1 convolutions
        actor_flattened_size = 2 * obs_shape[1] * obs_shape[2]
        critic_flattened_size = 1 * obs_shape[1] * obs_shape[2]

        self.actor = nn.Sequential(
            nn.Conv2d(channels[-1], 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(actor_flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Conv2d(channels[-1], 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(critic_flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        initialize_actor_critic_weights(self)

    def forward(self, obs, action_mask=None):
        features = self.shared_body(obs)

        logits = self.actor(features)
        value = self.critic(features)

        if action_mask is not None:
            if action_mask.dim() == 1 and logits.dim() == 2:
                action_mask = action_mask.unsqueeze(0)

            logits[~action_mask] = -torch.inf

            all_invalid = action_mask.sum(dim=-1) == 0
            if all_invalid.any():
                logits[all_invalid] = 0.0

        dist = Categorical(logits=logits)
        return dist, value


class CnnSActorCritic(BaseCnnActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, channels=[64, 64, 64, 64])
        self._architecture_name = "cnn_s"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }


class CnnLActorCritic(BaseCnnActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, channels=[128, 128, 128, 128, 128, 128])
        self._architecture_name = "cnn_l"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }
