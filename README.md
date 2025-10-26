# MNK Game Self-Play Reinforcement Learning

Reinforcement learning system implementing self-play training for MNK games using Proximal Policy Optimization (PPO) with dynamic opponent pools.

## Overview

This project implements a reinforcement learning framework for MNK games, including popular variants like Gomoku (9×9×5) where players aim to place K stones in a row on an M×N board. The system features vectorized training with intelligent opponent management, performance-based sampling, and comprehensive validation against benchmark opponents.

## Features

- **Custom PPO Implementation**: Full PyTorch implementation with separate policy/value networks
- **Dynamic Opponent Pool**: Performance-based weighted sampling with adaptive eviction
- **Vectorized Training**: Parallel environments for efficient learning
- **Comprehensive Validation**: Automated benchmarking against random and historical agents
- **W&B Integration**: Experiment tracking and hyperparameter sweeps
- **Zero-Sum Rewards**: Proper competitive game reward assignment
- **Action Masking**: Ensures only valid moves are selected

## Development

- **Python 3.12+** with full type hints
- **UV package manager** for dependency management
- **Custom PyTorch implementation** (no external RL libraries)
- **W&B** for experiment tracking and visualization
- **PettingZoo** for environment interface

## Usage

### Training
```bash
# Install dependencies
uv sync

# Start training
uv run src/vec_train_mnk.py

# Hyperparameter sweeps
wandb sweep sweep_config.yaml
wandb agent <SWEEP_ID>
```

### Playing
```bash
# Human vs AI
uv run src/play.py --p1 human --p2 model.pt --m 9 --n 9 --k 5

# AI vs AI
uv run src/play.py --p1 model1.pt --p2 model2.pt --m 9 --n 9 --k 5

# Custom board size
uv run src/play.py --p1 model.pt --p2 random --m 7 --n 7 --k 4
```

### Configuration
- **Default**: 9×9×5 board, 512 steps/iteration, 3e-4 learning rate
- **Opponent Pool**: Max 10 agents with weighted sampling
- **Validation**: Every 50 iterations against benchmarks
- **Hardware**: GPU preferred with CPU fallback

## Project Structure

- `src/env/mnk_game_env.py` - PettingZoo AEC environment
- `src/alg/ppo.py` - Custom PPO implementation
- `src/selfplay/opponent_pool.py` - Dynamic opponent management
- `src/vec_train_mnk.py` - Main training loop
- `src/play.py` - Human vs AI interface
- `src/validation.py` - Validation utilities
