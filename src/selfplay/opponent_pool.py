import random
from copy import deepcopy
from typing import List, Optional, Dict, Any
import torch
import numpy as np

from .policy import VectorNNPolicy


class OpponentPool:
    def __init__(self, network_creator, max_size: int = 10, device: str = "cpu"):
        self.network_creator = network_creator
        self.max_size = max_size
        self.device = device
        self.opponents: List[dict] = []  # Store opponent data with mutable stats

    def _evict_worst_performers(self) -> None:
        """Remove opponents with worst performance when pool is full."""
        if len(self.opponents) <= self.max_size:
            return

        # Sort opponents by mean_reward (ascending - worst first)
        sorted_indices = sorted(
            range(len(self.opponents)), key=lambda i: self.opponents[i]["mean_reward"]
        )

        # Remove the worst performers to get to max_size
        num_to_remove = len(self.opponents) - self.max_size
        for i in range(num_to_remove):
            # Remove from worst to better (reverse to maintain indices)
            worst_idx = sorted_indices[i]
            self.opponents.pop(worst_idx)
            # Adjust remaining indices
            sorted_indices = [idx - 1 if idx > worst_idx else idx for idx in sorted_indices]

    def add_opponent(self, network: torch.nn.Module) -> int:
        """Add a copy of the network to the opponent pool. Returns opponent index."""
        state_dict = deepcopy(network.state_dict())
        state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}

        opponent_data = {
            "state_dict": state_dict_cpu,
            "mean_reward": 1,
            "games_played": 0,
            "total_reward": 0.0,
        }

        self.opponents.append(opponent_data)

        # Remove worst performers if pool exceeds max size
        self._evict_worst_performers()

        # Return index of newly added opponent
        return len(self.opponents) - 1

    def update_opponent_stats(self, opponent_idx: int, reward: float) -> None:
        """Update statistics for a specific opponent after games."""
        if opponent_idx < 0 or opponent_idx >= len(self.opponents):
            return

        opponent = self.opponents[opponent_idx]
        opponent["games_played"] += 1
        opponent["total_reward"] += reward

        # Update average reward
        opponent["mean_reward"] = opponent["total_reward"] / opponent["games_played"]

    def _calculate_weight(self, opponent: dict) -> float:
        """Calculate sampling weight for an opponent."""
        # Performance weight: harder opponents get higher weight
        performance_weight = opponent["mean_reward"] + 1.0

        return np.exp(performance_weight)

    def sample_opponent(self) -> tuple[Optional[VectorNNPolicy], Optional[int]]:
        """Sample and create a single opponent using weighted sampling. Returns (opponent, index)."""
        if not self.opponents:
            return None, None

        weights = [self._calculate_weight(opp) for opp in self.opponents]
        weights_array = np.array(weights)
        weights = weights_array / weights_array.sum()

        selected_idx = np.random.choice(len(self.opponents), p=weights)
        selected_state = self.opponents[selected_idx]["state_dict"]

        network = self._create_network_from_state(selected_state)
        opponent = VectorNNPolicy(network, device=self.device)

        return opponent, selected_idx

    def _create_network_from_state(self, state_dict: dict) -> torch.nn.Module:
        """Create a network from the saved state dictionary."""
        network = self.network_creator()

        if self.device != "cpu":
            state_dict = {k: v.to(self.device) for k, v in state_dict.items()}

        network.load_state_dict(state_dict)
        network.to(self.device)
        network.eval()
        return network

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the current pool."""
        if not self.opponents:
            return {"size": 0}

        return {
            "size": len(self.opponents),
            "avg_mean_reward": np.mean([opp["mean_reward"] for opp in self.opponents]),
            "avg_games_played": np.mean([opp["games_played"] for opp in self.opponents]),
            "total_games": sum([opp["games_played"] for opp in self.opponents]),
        }
