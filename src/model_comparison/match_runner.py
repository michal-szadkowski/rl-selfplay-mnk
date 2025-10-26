import gc
import pandas as pd
from typing import List
from dataclasses import dataclass
from tqdm import tqdm

from env.mnk_game_env import create_mnk_env
from selfplay.policy import NNPolicy
from validation import play_single_episode
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

    def _play_single_match(
        self, model1: ModelInfo, model2: ModelInfo, games_per_pair: int
    ) -> pd.DataFrame:
        """Play games between two models and return results."""
        # Load models and create policies
        p1_model = model1.load_model(self.config.device)
        p2_model = model2.load_model(self.config.device)
        p1_policy = NNPolicy(p1_model, device=self.config.device)
        p2_policy = NNPolicy(p2_model, device=self.config.device)

        # Track results
        player1_wins = player2_wins = draws = 0

        # Play games with alternating who goes first
        games_as_first = games_per_pair // 2
        games_as_second = games_per_pair - games_as_first

        for _ in range(games_as_first):
            result = play_single_episode(self.env, p1_policy, p2_policy, agent_is_first=True)
            if result["outcome"] == "win":
                player1_wins += 1
            elif result["outcome"] == "loss":
                player2_wins += 1
            else:
                draws += 1

        for _ in range(games_as_second):
            result = play_single_episode(self.env, p1_policy, p2_policy, agent_is_first=False)
            if result["outcome"] == "win":
                player1_wins += 1
            elif result["outcome"] == "loss":
                player2_wins += 1
            else:
                draws += 1

        # Calculate scores
        total_games = games_per_pair
        player1_score = (player1_wins + 0.5 * draws) / total_games
        player2_score = (player2_wins + 0.5 * draws) / total_games

        # Cleanup
        del p1_policy, p2_policy
        model1.unload_model()
        model2.unload_model()

        if self.config.device != "cpu":
            import torch

            torch.cuda.empty_cache()
        gc.collect()

        # Return results
        return pd.DataFrame(
            [
                {
                    "player1_unique_id": model1.unique_id,
                    "player2_unique_id": model2.unique_id,
                    "player1_run_name": model1.run_name,
                    "player2_run_name": model2.run_name,
                    "player1_iteration": model1.iteration,
                    "player2_iteration": model2.iteration,
                    "total_games": total_games,
                    "player1_wins": player1_wins,
                    "player2_wins": player2_wins,
                    "draws": draws,
                    "player1_score": player1_score,
                    "player2_score": player2_score,
                }
            ]
        )

    def run_tournament(
        self, models: List[ModelInfo], games_per_pair: int = 50
    ) -> pd.DataFrame:
        """Run a round-robin tournament between all models."""
        from itertools import combinations

        model_pairs = list(combinations(models, 2))
        all_results = []

        pbar = tqdm(model_pairs, desc="Tournament matches")
        for model1, model2 in pbar:
            pbar.set_postfix({"match": f"{model1.unique_id} vs {model2.unique_id}"})
            match_result = self._play_single_match(model1, model2, games_per_pair)
            all_results.append(match_result)

            # Update postfix with result
            p1_wins = match_result["player1_wins"].iloc[0]
            p2_wins = match_result["player2_wins"].iloc[0]
            draws = match_result["draws"].iloc[0]
            pbar.set_postfix(
                {
                    "match": f"{model1.unique_id} vs {model2.unique_id}",
                    "result": f"{p1_wins}-{p2_wins}-{draws}",
                }
            )

        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
