# RL self-play MNK

Project implementing a machine learning system using "self-play" methods (playing against oneself) in MNK game (generalization of Tic-Tac-Toe).

## Project Description

MNK Selfplay is an implementation of the Advantage Actor-Critic (A2C) reinforcement learning algorithm for learning to play a generalized version of Tic-Tac-Toe. In MNK game, a player must place K stones in a row on an M×N board, allowing for game formats such as 9×9 with 5-in-a-row (common in Gomoku).

The project uses a game environment defined as a PettingZoo environment and trains an agent using self-play techniques, learning by playing against previous versions of itself or other strategies.

## Main Components

- **MNK Game Environment**: Implementation of the M×N board game with win condition - K stones in a line
- **A2C Algorithm**: Implementation of the Actor-Critic algorithm for learning optimal moves
- **Self-play**: System for learning by playing against oneself or other strategies
- **Validation**: System for testing agent performance against random strategies and benchmarks

## Features

- Agent training using the A2C algorithm
- Human vs computer and computer vs computer gameplay
- Ability to load saved neural network models
- Validation and tracking of learning progress using Weights & Biases
- Support for M×N×K format games (width × height × number of stones to win)
- Self-play system with dynamically updated opponents

## Technologies

- Python
- PyTorch
- PettingZoo
- Gymnasium
- Weights & Biases

## Usage

### Training an agent:

```bash
python -m src.train_mnk
```

### Playing against other agents or humans:

```bash
# Human vs model game
python -m src.play --p1 human --p2 path/to/model.pt --m 9 --n 9 --k 5

# Model vs random agent
python -m src.play --p1 path/to/model.pt --p2 random --m 9 --n 9 --k 5

# Model vs model game
python -m src.play --p1 model1.pt --p2 model2.pt --m 9 --n 9 --k 5
```

## Neural Network Architecture

The model consists of:
- Shared convolutional part (Conv2D + LayerNorm) for feature extraction from the board
- Actor layers for predicting move probabilities
- Critic layer for estimating state value

## Training Algorithm

- A2C (Advantage Actor-Critic) algorithm
- Parallelization across multiple environments
- Self-play with dynamically updated opponent pool
- Periodic validation against benchmarks