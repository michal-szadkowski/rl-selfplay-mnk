import gc
import pandas as pd
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from env.mnk_game_env import create_mnk_env
from selfplay.policy import BatchNNPolicy
from selfplay.vector_mnk_self_play import VectorMnkSelfPlayWrapper
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

    def _cleanup(
        self,
        p1_policy: BatchNNPolicy,
        p2_policy: BatchNNPolicy,
        model1: ModelInfo,
        model2: ModelInfo,
    ) -> None:
        """Clean up resources."""
        del p1_policy, p2_policy
        model1.unload_model()
        model2.unload_model()

        if self.config.device != "cpu":
            import torch

            torch.cuda.empty_cache()
        gc.collect()

    def _play_vectorized_batch(
        self, model1: ModelInfo, model2: ModelInfo, games_per_pair: int
    ) -> pd.DataFrame:
        """Play games between two models using vectorized environment."""

        p1_model = model1.load_model(self.config.device)
        p2_model = model2.load_model(self.config.device)
        p1_policy = BatchNNPolicy(p1_model, device=self.config.device)
        p2_policy = BatchNNPolicy(p2_model, device=self.config.device)

        result = self._play_match_with_policies(
            p1_policy, p2_policy, model1, model2, games_per_pair
        )

        self._cleanup(p1_policy, p2_policy, model1, model2)

        return result

    def _play_match_with_policies(
        self,
        p1_policy: BatchNNPolicy,
        p2_policy: BatchNNPolicy,
        model1: ModelInfo,
        model2: ModelInfo,
        games_per_pair: int,
    ) -> pd.DataFrame:
        """Play match with already loaded policies."""

        games_as_first = games_per_pair // 2
        games_as_second = games_per_pair - games_as_first

        wins_first, losses_first, draws_first = self._play_batch_games(
            p1_policy, p2_policy, games_as_first, "black"
        )
        wins_second, losses_second, draws_second = self._play_batch_games(
            p1_policy, p2_policy, games_as_second, "white"
        )

        player1_wins = wins_first + wins_second
        player2_wins = losses_first + losses_second
        draws = draws_first + draws_second
        total_games = games_per_pair
        player1_score = (player1_wins + 0.5 * draws) / total_games
        player2_score = (player2_wins + 0.5 * draws) / total_games

        return self._create_result_df(
            model1,
            model2,
            total_games,
            player1_wins,
            player2_wins,
            draws,
            player1_score,
            player2_score,
        )

    def _play_batch_games(
        self, p1_policy: BatchNNPolicy, p2_policy: BatchNNPolicy, n_games: int, p1_color: str
    ) -> tuple[int, int, int]:
        """Play batch of games and return (wins, losses, draws) for p1."""
        if n_games == 0:
            return 0, 0, 0

        # Create environment
        env = VectorMnkSelfPlayWrapper(
            m=self.config.m, n=self.config.n, k=self.config.k, n_envs=n_games
        )
        env.players = np.array([p1_color] * n_games)
        env.set_opponent(p2_policy)

        # Play games
        obs, _ = env.reset()
        completed = np.zeros(n_games, dtype=bool)
        wins = losses = draws = 0

        while not np.all(completed):
            actions = p1_policy.act(obs)
            obs, rewards, terminations, truncations, _ = env.step(actions)

            # Count newly completed games
            just_completed = (terminations | truncations) & ~completed
            for i in np.where(just_completed)[0]:
                if rewards[i] > 0:
                    wins += 1
                elif rewards[i] < 0:
                    losses += 1
                else:
                    draws += 1

            completed |= terminations | truncations

        del env
        return wins, losses, draws

    def _create_result_df(
        self,
        model1: ModelInfo,
        model2: ModelInfo,
        total_games: int,
        player1_wins: int,
        player2_wins: int,
        draws: int,
        player1_score: float,
        player2_score: float,
    ) -> pd.DataFrame:
        """Create results DataFrame."""
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

    def run_tournament_batched(
        self, models: List[ModelInfo], games_per_pair: int, batch_size: int = 8
    ) -> pd.DataFrame:
        """Run tournament using batch loading."""
        all_results = []
        total_matches = len(models) * (len(models) - 1) // 2

        pbar = tqdm(total=total_matches, desc="Tournament matches (batched)")

        for row_start in range(0, len(models), batch_size):
            row_batch = models[row_start : row_start + batch_size]

            # Load row batch
            loaded_row = []
            for model in row_batch:
                model_data = model.load_model()
                policy = BatchNNPolicy(model_data, device=self.config.device)
                loaded_row.append((model, policy))

            for col_start in range(row_start, len(models), batch_size):
                col_batch = models[col_start : col_start + batch_size]

                # Load column batch
                if col_start == row_start:
                    loaded_col = loaded_row  # Same batch
                else:
                    loaded_col = []
                    for model in col_batch:
                        model_data = model.load_model()
                        policy = BatchNNPolicy(model_data, device=self.config.device)
                        loaded_col.append((model, policy))

                # Play all matches: row_batch × col_batch
                for i, (model1, policy1) in enumerate(loaded_row):
                    start_j = i + 1 if col_start == row_start else 0  # Skip diagonal
                    for model2, policy2 in loaded_col[start_j:]:
                        result = self._play_match_with_policies(
                            policy1, policy2, model1, model2, games_per_pair
                        )
                        all_results.append(result)

                        # Update progress
                        p1_wins = result["player1_wins"].iloc[0]
                        p2_wins = result["player2_wins"].iloc[0]
                        draws = result["draws"].iloc[0]
                        pbar.set_postfix(
                            {
                                "match": f"{model1.unique_id} vs {model2.unique_id}",
                                "result": f"{p1_wins}-{p2_wins}-{draws}",
                            }
                        )
                        pbar.update(1)

                # Cleanup column batch if different
                if col_start != row_start:
                    self._cleanup_batch_gpu(loaded_col)

            # Cleanup row batch
            self._cleanup_batch_gpu(loaded_row)

        pbar.close()
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    def _cleanup_batch_gpu(self, loaded_batch: List[Tuple[ModelInfo, BatchNNPolicy]]) -> None:
        """Clean up batch of GPU models."""
        for model, policy in loaded_batch:
            model.unload_model()
            del policy

        gc.collect()

    def _run_tournament_sequential(
        self, models: List[ModelInfo], games_per_pair: int
    ) -> pd.DataFrame:
        """Run tournament with original sequential approach."""
        from itertools import combinations

        model_pairs = list(combinations(models, 2))
        all_results = []

        pbar = tqdm(model_pairs, desc="Tournament matches (sequential)")
        for model1, model2 in pbar:
            pbar.set_postfix({"match": f"{model1.unique_id} vs {model2.unique_id}"})
            match_result = self._play_vectorized_batch(model1, model2, games_per_pair)
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
