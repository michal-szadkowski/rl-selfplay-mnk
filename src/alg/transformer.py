import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class TransformerActorCritic(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
    ):
        """
        Transformer-based Actor-Critic implementation.
        Treats each board cell as a separate token.

        :param obs_shape: Observation shape (C, H, W)
        :param action_dim: Number of possible actions
        :param embed_dim: Dimension of the embedding vector after cell embedding.
        :param num_layers: Number of Transformer Encoder blocks.
        :param num_heads: Number of self-attention heads (must divide embed_dim).
        """
        super().__init__()
        c, h, w = obs_shape
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        self._architecture_name = "transformer_actor_critic"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
            "embed_dim": int(embed_dim),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
        }

        # Number of tokens = number of cells on the board
        num_tokens = h * w

        # 1. Cell Embedding
        # Use 1x1 convolution to transform C channels of each cell
        # into a vector of dimension `embed_dim`. This is our "tokenizer".
        self.cell_embed = nn.Conv2d(c, embed_dim, kernel_size=1, stride=1)

        # 2. CLS Token and Positional Encoding
        # CLS token will serve as global representation of board state
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # Standard practice
            batch_first=True,  # Expect input (Batch, Seq_Len, Dim)
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Normalization layers (standard in Transformers)
        self.norm_in = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # 4. Actor and Critic heads
        # Feature vector is CLS token output of dimension `embed_dim`
        flat_size = embed_dim

        self.actor = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialization for CLS token and positions (standard ViT)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        # Orthogonal initialization for the rest of the network (good for RL)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                if module.out_features == 1:  # Critic output
                    nn.init.orthogonal_(module.weight, gain=1.0)
                elif module.out_features == self.action_dim:  # Actor output
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:  # Other layers
                    nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        """Extract features from observation, returns state representation (from CLS token)."""
        b_size = x.shape[0]

        # 1. Cell Embedding: (B, C, H, W) -> (B, D, H, W)
        #    (D is embed_dim)
        x = self.cell_embed(x)

        # 2. Flatten to sequence: (B, D, H, W) -> (B, D, N) -> (B, N, D)
        #    Where N = H * W (number of tokens/cells)
        x = x.flatten(2).transpose(1, 2)

        # 3. Add CLS token: (B, N, D) -> (B, N+1, D)
        cls_tokens = self.cls_token.expand(b_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 4. Add positional encoding and normalization
        x = x + self.pos_embed
        x = self.norm_in(x)

        # 5. Pass through Transformer Encoder
        x = self.transformer(x)

        # 6. Extract CLS token as state representation (B, D)
        #    Take only the first token (index 0)
        features = x[:, 0]
        features = self.norm_out(features)  # Final normalization

        return features

    def forward(self, obs, action_mask=None):
        features = self.forward_features(obs)
        logits = self.actor(features)

        if action_mask is not None:
            # Ensure mask has correct dimensions
            if action_mask.dim() == 1 and logits.dim() == 2:
                action_mask = action_mask.unsqueeze(0)

            logits = logits.clone()
            logits[~action_mask] = -torch.inf

            # Handle case when all moves are illegal (e.g., end of game)
            all_invalid = action_mask.sum(dim=-1) == 0
            if all_invalid.any():
                logits[all_invalid] = 0.0

        dist = Categorical(logits=logits)
        value = self.critic(features)
        return dist, value
