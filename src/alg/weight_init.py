import torch.nn as nn


def initialize_actor_critic_weights(model):
    """
    Shared weight initialization for Actor-Critic networks.

    Args:
        model: The neural network model to initialize
    """
    # Transformer-specific initialization (check for attributes)
    if hasattr(model, "cls_token"):
        nn.init.normal_(model.cls_token, std=0.02)
    if hasattr(model, "pos_embed"):
        nn.init.normal_(model.pos_embed, std=0.02)

    # Common initialization for all layer types
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            if module.out_features == 1:  # Critic output
                nn.init.orthogonal_(module.weight, gain=1.0)
            elif module.out_features == model.action_dim:  # Actor output
                nn.init.orthogonal_(module.weight, gain=0.01)
            else:  # Hidden layers
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
