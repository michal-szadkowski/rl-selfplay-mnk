import torch.nn as nn


def find_last_linear(module):
    """
    Helper function to find the last linear layer in a module.
    """
    last_linear = None
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            last_linear = layer
            break
    return last_linear


def initialize_weights_explicit(modules_to_init, actor_head, critic_head):
    """
    Applies orthogonal initialization to passed modules.
    This allows full control over what gets initialized.
    """
    # List of all modules for general initialization
    all_modules = modules_to_init + [actor_head, critic_head]

    # 1. General initialization of all passed modules
    for module in all_modules:
        if module is None:
            continue
        # .modules() loop is needed here to reach layers inside nn.Sequential
        for submodule in module.modules():
            if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                gain = nn.init.calculate_gain("relu")
                nn.init.orthogonal_(submodule.weight, gain=gain)
                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)
            elif isinstance(submodule, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(submodule.weight)
                nn.init.zeros_(submodule.bias)

    # 2. Special OVERWRITE for the last layer of Actor head
    if actor_head:
        last_linear_actor = find_last_linear(actor_head)
        if last_linear_actor:
            nn.init.orthogonal_(last_linear_actor.weight, gain=0.01)
            nn.init.zeros_(last_linear_actor.bias)

    # 3. Special OVERWRITE for the last layer of Critic head
    if critic_head:
        last_linear_critic = find_last_linear(critic_head)
        if last_linear_critic:
            nn.init.orthogonal_(last_linear_critic.weight, gain=1.0)
            nn.init.zeros_(last_linear_critic.bias)
