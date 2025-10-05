# MNK Game Self-Play Reinforcement Learning

Reinforcement learning system implementing self-play training for MNK games (generalized Tic-Tac-Toe) using Advantage Actor-Critic (A2C) algorithms.

## Overview

This project implements a reinforcement learning framework for MNK games, where players aim to place K stones in a row on an M×N board. The system features vectorized training with dynamic opponent pools for efficient self-play learning.

## Architecture

### Core Components

- **MNK Game Environment**: Custom PettingZoo environment implementing M×N×K game dynamics
- **A2C Agent**: Advantage Actor-Critic implementation with policy and value function optimization
- **Vectorized Self-Play**: Training system with dynamic opponent management
- **Validation System**: Benchmarking against random and previous agent versions

### Technical Stack

- **Framework**: PyTorch for deep learning
- **Environment**: PettingZoo/Gymnasium for RL environments
- **Training**: Vectorized environments for parallel training
- **Monitoring**: Weights & Biases for experiment tracking

## Usage

### Training

```bash
# Vectorized training with self-play
python -m src.vec_train_mnk
```

### Gameplay

```bash
# Human vs trained model
python -m src.play --p1 human --p2 path/to/model.pt --m 9 --n 9 --k 5

# Model vs random agent
python -m src.play --p1 path/to/model.pt --p2 random --m 9 --n 9 --k 5

# Model vs model
python -m src.play --p1 model1.pt --p2 model2.pt --m 9 --n 9 --k 5

# Custom MNK configuration
python -m src.play --p1 model.pt --p2 random --m 7 --n 7 --k 4
```

## Key Features

- **Self-Play Training**: Agents learn by playing against dynamically updated opponents
- **Vectorized Environments**: Parallel training for improved efficiency
- **Adaptive Difficulty**: Dynamic opponent pool with previous agent versions
- **Validation System**: Performance tracking against benchmarks
- **Action Masking**: Valid move enforcement for game rules compliance
- **Flexible Configurations**: Support for various MNK settings (M×N×K)
- **Model Persistence**: Save/load trained agent models

## Game Mechanics

MNK generalizes Tic-Tac-Toe where:
- **M**: Board width
- **N**: Board height  
- **K**: Stones required in a row to win

## Development Setup

```bash
# Install dependencies
uv sync
```

Dependencies include PyTorch, PettingZoo, Gymnasium, and Weights & Biases.

## Hyperparameter Sweeping with Weights & Biases

This project supports hyperparameter optimization using W&B sweeps. The sweep configuration is defined in `sweep_config.yaml`.

### Setting up a Sweep

1. First, ensure you have a W&B account and have logged in using `wandb login`
2. Create a new sweep using the configuration file:

```bash
wandb sweep sweep_config.yaml
```

This will output a sweep ID. Make note of it as you'll need it to launch sweep agents.

### Running Sweep Agents

To launch agents that will run the training with different hyperparameter combinations:

```bash
# Launch a sweep agent with the sweep ID from the previous step
wandb agent <SWEEP_ID>
```

Each agent will run a complete training run with different hyperparameter values sampled from the defined search space.

The sweep is configured to maximize the metric `validation/vs_benchmark/win_rate` using Bayesian optimization.
