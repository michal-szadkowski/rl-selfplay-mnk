import torch
import torch.nn as nn
from torch.distributions import Categorical
from ..weight_init import initialize_weights_explicit


class SGRBlock(nn.Module):
    """
    Stabilized Gated Residual Block - standard 2026 dla RL.
    Zastępuje standardowy blok Transformera mechanizmem bramkowania SGR.
    """

    def __init__(self, d_model, nhead, dim_feedforward, bias_init=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.gate1 = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.gate2 = nn.Linear(d_model, d_model)

        # Inicjalizacja Identity: bramki bliskie 0 na starcie, by przepuszczać skip-connection
        with torch.no_grad():
            nn.init.constant_(self.gate1.weight, 0)
            nn.init.constant_(self.gate1.bias, bias_init)
            nn.init.constant_(self.gate2.weight, 0)
            nn.init.constant_(self.gate2.bias, bias_init)

    def forward(self, x):
        # 1. Multi-Head Attention z bramką SGR
        res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        g1 = torch.sigmoid(self.gate1(attn_out))
        x = res + g1 * attn_out

        # 2. Feed Forward z bramką SGR
        res = x
        x = self.norm2(x)
        mlp_out = self.mlp(x)
        g2 = torch.sigmoid(self.gate2(mlp_out))
        x = res + g2 * mlp_out
        return x


class BaseTransformerActorCritic(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        head_hidden_dim=256,
    ):
        super().__init__()
        c, h, w = obs_shape
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        num_tokens = h * w

        # --- EMBEDDINGI (BEZ ZMIAN) ---
        self.cell_embed = nn.Conv2d(c, embed_dim, kernel_size=1, stride=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # --- CIAŁO TRANSFORMERA (MOJA PROPOZYCJA SGR) ---
        # Zamiast nn.TransformerEncoder, stosujemy listę bloków SGR
        self.transformer = nn.Sequential(
            *[SGRBlock(embed_dim, num_heads, embed_dim * 4) for _ in range(num_layers)]
        )

        # --- GŁOWICE (BEZ ZMIAN) ---
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

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cell_embed.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.cell_embed.bias, 0.0)

        initialize_weights_explicit(
            modules_to_init=[],
            actor_head=self.policy_head,
            critic_head=self.value_head,
        )

    def forward_body(self, x):
        x = self.cell_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)  # Przejście przez sekwencję bloków SGR
        return x

    def forward(self, obs, action_mask=None):
        features = self.forward_body(obs)
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


# Klasy S i L pozostają bez zmian w strukturze inicjalizacji
class TransformerSActorCritic(BaseTransformerActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(
            obs_shape, action_dim, embed_dim=56, num_layers=2, num_heads=4, head_hidden_dim=128
        )
        self._architecture_name = "transformer_c_s"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }


class TransformerLActorCritic(BaseTransformerActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, embed_dim=96, num_layers=5, num_heads=8)
        self._architecture_name = "transformer_c_l"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }
