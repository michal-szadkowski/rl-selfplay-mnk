import torch.nn as nn


def initialize_actor_critic_weights(model):
    """
    Shared weight initialization for Actor-Critic networks.

    Args:
        model: The neural network model to initialize
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            if module.out_features == 1:  # Critic output
                nn.init.orthogonal_(module.weight, gain=1.0)
            elif (
                hasattr(model, "action_dim") and module.out_features == model.action_dim
            ):  # Actor output
                nn.init.orthogonal_(module.weight, gain=0.01)
            else:  # Hidden layers
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def initialize_transformer_weights(model):
    """
    Specialized weight initialization for Transformer architectures.
    Uses Xavier/Glorot initialization for attention and feedforward layers.

    Args:
        model: The Transformer model to initialize
    """
    # Initialize position embeddings with truncated normal
    if hasattr(model, "pos_embed"):
        nn.init.trunc_normal_(model.pos_embed, std=0.02, a=-0.04, b=0.04)

    # Initialize cell embedding (Conv2d)
    if hasattr(model, "cell_embed"):
        nn.init.xavier_uniform_(
            model.cell_embed.weight, gain=nn.init.calculate_gain("linear")
        )
        if model.cell_embed.bias is not None:
            nn.init.zeros_(model.cell_embed.bias)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Use Xavier for transformer internals, orthogonal for heads
            if (
                hasattr(model, "action_dim") and module.out_features == model.action_dim
            ):  # Actor output
                nn.init.orthogonal_(module.weight, gain=0.01)
            elif module.out_features == 1:  # Critic output
                nn.init.orthogonal_(module.weight, gain=1.0)
            else:  # Hidden layers and transformer internals
                nn.init.xavier_uniform_(
                    module.weight, gain=nn.init.calculate_gain("linear")
                )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv1d):
            # For policy/value heads - use small gain for stability
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.MultiheadAttention):
            # Xavier for attention weights
            nn.init.xavier_uniform_(
                module.in_proj_weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.xavier_uniform_(
                module.out_proj.weight, gain=nn.init.calculate_gain("linear")
            )
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
