import gc
import pandas as pd
import torch
from typing import List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from env.torch_vector_mnk_env import TorchVectorMnkEnv
from env.constants import PLAYER_WHITE
from selfplay.policy import Policy, NNPolicy
from .model_loader import ModelInfo


@dataclass
class GameConfig:

    m: int = 9
    n: int = 9
    k: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MatchRunner:
    def __init__(self, config: GameConfig):
        self.config = config

    def run_tournament_batched(
        self, models: List[ModelInfo], games_per_pair: int, batch_size: int = 8
    ) -> pd.DataFrame:
        all_results = []
        if len(models) < 2:
            return pd.DataFrame()

        total_matches = len(models) * (len(models) - 1) // 2
        pbar = tqdm(total=total_matches, desc="Tournament matches (batched)")

        for row_start in range(0, len(models), batch_size):
            row_batch = models[row_start : row_start + batch_size]

            loaded_row = []
            for model in row_batch:
                model_data = model.load_model(self.config.device)
                policy = NNPolicy(model_data)
                loaded_row.append((model, policy))

            for col_start in range(row_start, len(models), batch_size):
                col_batch = models[col_start : col_start + batch_size]

                if col_start == row_start:
                    loaded_col = loaded_row
                else:
                    loaded_col = []
                    for model in col_batch:
                        model_data = model.load_model(self.config.device)
                        policy = NNPolicy(model_data)
                        loaded_col.append((model, policy))

                for i, (model1, policy1) in enumerate(loaded_row):
                    start_j = i + 1 if col_start == row_start else 0

                    for model2, policy2 in loaded_col[start_j:]:
                        result = self._play_match_with_policies(
                            policy1, policy2, model1, model2, games_per_pair
                        )
                        all_results.append(result)

                        p1_wins = result["player1_wins"].iloc[0]
                        p2_wins = result["player2_wins"].iloc[0]
                        draws = result["draws"].iloc[0]
                        pbar.set_postfix(
                            {
                                "match": f"{model1.unique_id} vs {model2.unique_id}",
                                "res": f"{p1_wins}-{p2_wins}-{draws}",
                            }
                        )
                        pbar.update(1)

                if col_start != row_start:
                    self._cleanup_batch_gpu(loaded_col, hard=False)

            self._cleanup_batch_gpu(loaded_row, hard=True)

        pbar.close()
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    def _play_match_with_policies(
        self,
        p1_policy: Policy,
        p2_policy: Policy,
        model1: ModelInfo,
        model2: ModelInfo,
        games_per_pair: int,
    ) -> pd.DataFrame:

        games_as_first = games_per_pair // 2
        games_as_second = games_per_pair - games_as_first

        w1, l1, d1 = self._play_batch_games(
            p1_policy, p2_policy, games_as_first, p1_is_black=True
        )

        w2, l2, d2 = self._play_batch_games(
            p1_policy, p2_policy, games_as_second, p1_is_black=False
        )

        player1_wins = w1 + w2
        player2_wins = l1 + l2
        draws = d1 + d2
        total_games = games_per_pair

        player1_score = (player1_wins + 0.5 * draws) / max(1, total_games)
        player2_score = (player2_wins + 0.5 * draws) / max(1, total_games)

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
        self, p1_policy: Policy, p2_policy: Policy, n_games: int, p1_is_black: bool
    ) -> tuple[int, int, int]:
        if n_games == 0:
            return 0, 0, 0

        # Create GPU environment
        env = TorchVectorMnkEnv(
            m=self.config.m,
            n=self.config.n,
            k=self.config.k,
            num_envs=n_games,
            device=self.config.device,
        )

        obs = env.reset()
        dones = torch.zeros(n_games, dtype=torch.bool, device=self.config.device)

        wins = 0
        losses = 0
        draws = 0

        agent_side_val = 0 if p1_is_black else 1

        while not dones.all():
            current_player = env.current_player  # (num_envs,)

            active_games = ~dones

            is_p1_turn = (current_player == agent_side_val) & active_games
            is_p2_turn = (current_player != agent_side_val) & active_games

            if not is_p1_turn.any() and not is_p2_turn.any():
                break

            actions = torch.full((n_games,), 0, dtype=torch.long, device=self.config.device)

            if is_p1_turn.any():
                p1_obs_input = {
                    "observation": obs["observation"][is_p1_turn].clone(),
                    "action_mask": obs["action_mask"][is_p1_turn],
                }

                if agent_side_val == PLAYER_WHITE:
                    p1_obs_input["observation"] = torch.flip(
                        p1_obs_input["observation"], dims=(1,)
                    )

                with torch.no_grad():
                    act = p1_policy.act(p1_obs_input, deterministic=False)

                actions[is_p1_turn] = act

            if is_p2_turn.any():
                p2_obs_input = {
                    "observation": obs["observation"][is_p2_turn].clone(),
                    "action_mask": obs["action_mask"][is_p2_turn],
                }

                opponent_side = 1 - agent_side_val
                if opponent_side == PLAYER_WHITE:
                    p2_obs_input["observation"] = torch.flip(
                        p2_obs_input["observation"], dims=(1,)
                    )

                with torch.no_grad():
                    act = p2_policy.act(p2_obs_input, deterministic=False)

                actions[is_p2_turn] = act

            moving_indices = torch.nonzero(is_p1_turn | is_p2_turn).squeeze(1)
            active_actions = actions[moving_indices]

            obs, rewards, step_dones = env.step_subset(active_actions, moving_indices)

            just_finished_mask = step_dones & (~dones)

            if just_finished_mask.any():
                finished_indices = torch.nonzero(just_finished_mask).squeeze(1)

                winners_mask = (rewards == 1.0) & just_finished_mask
                draws_mask = (rewards == 0.0) & just_finished_mask

                p1_wins_mask = winners_mask & is_p1_turn
                p1_losses_mask = winners_mask & (~is_p1_turn)

                wins += p1_wins_mask.sum().item()
                losses += p1_losses_mask.sum().item()
                draws += draws_mask.sum().item()

                dones[just_finished_mask] = True

        del env
        return int(wins), int(losses), int(draws)

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

    def _cleanup_batch_gpu(
        self, loaded_batch: List[Tuple[ModelInfo, Policy]], hard: bool = False
    ) -> None:
        """Clean up batch of GPU models."""
        for model, policy in loaded_batch:
            model.unload_model(hard=hard)
            del policy

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
