"""Learning rate scheduler factory for PPO training."""

import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR


def create_lr_scheduler(
    optimizer, warmup_steps, total_steps, num_envs, n_steps, decay=False
):
    """
    Create learning rate scheduler with warmup and optional decay.

    Args:
        optimizer: PyTorch optimizer to wrap
        warmup_steps: Number of warmup steps (0 to disable)
        total_steps: Total number of environment steps
        num_envs: Number of parallel environments
        n_steps: Number of steps per rollout
        decay: Whether to linearly decay LR to 0 after warmup

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None
    """
    steps_per_iteration = num_envs * n_steps
    total_iterations = total_steps // steps_per_iteration
    warmup_iterations = max(1, warmup_steps // steps_per_iteration)

    schedulers = []
    milestones = []

    # 1. Warmup Phase
    if warmup_steps > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,  # Start from very low LR
            end_factor=1.0,
            total_iters=warmup_iterations,
        )
        schedulers.append(warmup)
        milestones.append(warmup_iterations)
    else:
        # If no warmup, we assume we are at step 0 of "main" phase immediately
        warmup_iterations = 0

    # 2. Main Phase (Decay or Constant)
    if decay:
        decay_iterations = max(1, total_iterations - warmup_iterations)
        main = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,  # Decay to 1% of base LR
            total_iters=decay_iterations,
        )
    else:
        main = ConstantLR(optimizer, factor=1.0)

    schedulers.append(main)

    return SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones,
    )
