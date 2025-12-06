import torch
import torch.nn as nn
from torch.distributions import Categorical
from ..weight_init import initialize_weights_explicit


class BaseTransformerActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, embed_dim=128, num_layers=4, num_heads=4, head_hidden_dim=256):
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
            nn.Conv1d(embed_dim, 2, kernel_size=1),
            nn.Flatten(),
            nn.LayerNorm(2 * num_tokens),
            nn.ReLU(),
            nn.Linear(2 * num_tokens, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Conv1d(embed_dim, 1, kernel_size=1),
            nn.Flatten(),
            nn.LayerNorm(1 * num_tokens),
            nn.ReLU(),
            nn.Linear(1 * num_tokens, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1),
            nn.Tanh(),
        )

        # Initialize parameters that won't be touched by the function
        nn.init.normal_(self.pos_embed, std=0.02)

        # Explicitly pass only what we want to initialize
        initialize_weights_explicit(
            modules_to_init=[self.cell_embed, self.transformer],
            actor_head=self.policy_head,
            critic_head=self.value_head,
        )

    def forward_body(self, x):
        x = self.cell_embed(x)
        # Convert (B, D, H, W) to (B, H*W, D) sequence format
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        return x

    def forward(self, obs, action_mask=None):
        features = self.forward_body(obs)
        # Convert (B, seq_len, embed_dim) to (B, embed_dim, seq_len) for Conv1d
        features_conv = features.transpose(1, 2)

        logits = self.policy_head(features_conv)
        value = self.value_head(features_conv)

        if action_mask is not None:
            if action_mask.dim() == 1 and logits.dim() == 2:
                action_mask = action_mask.unsqueeze(0)

            min_val = -torch.inf
            logits = torch.where(action_mask.bool(), logits, min_val)

            is_all_masked = logits.max(dim=1, keepdim=True)[0] == min_val
            logits = torch.where(is_all_masked, torch.zeros_like(logits), logits)

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
        super().__init__(obs_shape, action_dim, embed_dim=192, num_layers=5, num_heads=6)
        self._architecture_name = "transformer_l"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }
