import gc
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from ..env.mnk_game_env import create_mnk_env
from ..selfplay.policy import NNPolicy
from ..validation import play_single_episode
from .model_loader import ModelInfo


@dataclass
class GameConfig:
    """Configuration for game environment."""
    m: int = 9  # Board width
    n: int = 9  # Board height
    k: int = 5  # Win condition
    device: str = "cpu"


class MatchRunner:
    """Runs individual matches between models."""

    def __init__(self, config: GameConfig):
        self.config = config
        self.env = create_mnk_env(m=config.m, n=config.n, k=config.k)

    def play_match(self, model1: ModelInfo, model2: ModelInfo,
                   games_per_pair: int = 50) -> Tuple[float, float, Dict[str, Any]]:
        """
        Play a series of games between two models.

        Returns:
            Tuple of (player1_score, player2_score, game_stats)
        """
        player1_wins = 0
        player2_wins = 0
        draws = 0
        game_lengths = []

        # Load models and create policies
        p1_model = model1.load_model(self.config.device)
        p2_model = model2.load_model(self.config.device)

        p1_policy = NNPolicy(p1_model, device=self.config.device)
        p2_policy = NNPolicy(p2_model, device=self.config.device)

        # Play half the games with player 1 as first, half with player 2 as first
        games_as_first = games_per_pair // 2
        games_as_second = games_per_pair - games_as_first

        def update_results(result: Dict[str, Any]):
            """Update win counters based on game result."""
            nonlocal player1_wins, player2_wins, draws
            if result['outcome'] == 'win':
                player1_wins += 1
            elif result['outcome'] == 'loss':
                player2_wins += 1
            else:
                draws += 1

        # Games where player 1 goes first
        for game_idx in range(games_as_first):
            result = play_single_episode(
                self.env, p1_policy, p2_policy,
                agent_is_first=True
            )
            update_results(result)

            # Games where player 1 goes second
        for game_idx in range(games_as_second):
            result = play_single_episode(
                self.env, p1_policy, p2_policy,
                agent_is_first=False
            )
            update_results(result)

        # Calculate scores
        total_games = games_per_pair
        player1_score = (player1_wins + 0.5 * draws) / total_games
        player2_score = (player2_wins + 0.5 * draws) / total_games

        game_stats = {
            'total_games': total_games,
            'player1_wins': player1_wins,
            'player2_wins': player2_wins,
            'draws': draws,
            'player1_avg_reward': result.get('agent_reward', 0) if total_games > 0 else 0,
            'player2_avg_reward': result.get('opponent_reward', 0) if total_games > 0 else 0,
            'player1_score': player1_score,
            'player2_score': player2_score
        }

        # Cleanup - let ModelInfo handle its own memory management
        del p1_policy, p2_policy

        # Notify ModelInfo objects to handle GPU memory cleanup
        model1.unload_model()
        model2.unload_model()

        # CUDA-specific cleanup
        if self.config.device != 'cpu':
            import torch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        return player1_score, player2_score, game_stats