import torch
import torch.nn as nn
from torch.distributions import Categorical
from ..weight_init import initialize_actor_critic_weights


class BaseTransformerActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, embed_dim=128, num_layers=4, num_heads=4):
        super().__init__()
        c, h, w = obs_shape
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        num_tokens = h * w

        self.cell_embed = nn.Conv2d(c, embed_dim, kernel_size=1, stride=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * num_tokens, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * num_tokens, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        initialize_actor_critic_weights(self)

    def forward_body(self, x):
        x = self.cell_embed(x)
        # Convert (B, D, H, W) to (B, H*W, D) sequence format
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        return x

    def forward(self, obs, action_mask=None):
        features = self.forward_body(obs)

        logits = self.policy_head(features)
        value = self.value_head(features)

        if action_mask is not None:
            if action_mask.dim() == 1 and logits.dim() == 2:
                action_mask = action_mask.unsqueeze(0)

            logits = logits.clone()
            logits[~action_mask] = -torch.inf

            all_invalid = action_mask.sum(dim=-1) == 0
            if all_invalid.any():
                logits[all_invalid] = 0.0

        dist = Categorical(logits=logits)
        return dist, value


class TransformerSActorCritic(BaseTransformerActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, embed_dim=96, num_layers=3, num_heads=3)
        self._architecture_name = "transformer_s"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }


class TransformerLActorCritic(BaseTransformerActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, embed_dim=256, num_layers=6, num_heads=8)
        self._architecture_name = "transformer_l"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }
