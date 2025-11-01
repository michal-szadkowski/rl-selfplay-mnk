"""Learning rate scheduler factory for PPO training."""

import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR


def create_lr_scheduler(optimizer, config, num_envs, n_steps):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer to wrap
        config: Scheduler configuration dictionary
        num_envs: Number of parallel environments
        n_steps: Number of steps per rollout
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler or None
    """
    if not config:
        return None
    
    # Extract warmup_steps from either format
    warmup_steps = config.get("warmup_steps", 0)
    
    if warmup_steps <= 0:
        return None
    
    # Convert environment steps to iterations
    steps_per_iteration = num_envs * n_steps
    warmup_iterations = max(1, warmup_steps // steps_per_iteration)
    
    # Warmup phase: linear from 0 to target LR
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_iterations,
    )
    # Main training phase: constant LR
    main = ConstantLR(optimizer, factor=1.0)
    
    return SequentialLR(
        optimizer, schedulers=[warmup, main], milestones=[warmup_iterations]
    )