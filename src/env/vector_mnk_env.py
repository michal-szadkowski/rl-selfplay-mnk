import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from numba import njit, prange


class VectorMnkEnv:
    def __init__(self, m: int, n: int, k: int, parallel: int):
        self.m = m
        self.n = n
        self.k = k
        self.parallel = parallel
        self.num_envs = parallel
        self.possible_agents = ["black", "white"]

        self.single_observation_space = spaces.Dict(
            {
                "observation": spaces.Box(0, 1, (2, m, n), dtype=np.int8),
                "action_mask": spaces.Box(0, 1, (m * n,), dtype=np.int8),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = spaces.Discrete(m * n)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.agent_selection = np.full(self.parallel, "black")
        self.boards = np.zeros((self.parallel, 2, m, n), dtype=np.int8)
        self.rewards = np.zeros((self.parallel, 2), dtype=np.float32)
        self.terminations = np.zeros(self.parallel, dtype=bool)
        self.infos: List[Optional[Dict[str, Any]]] = [None] * self.parallel

    def last(
        self,
    ) -> Tuple[
        Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]
    ]:
        agent_idx = (self.agent_selection == "white").astype(int)
        rewards = self.rewards[np.arange(self.parallel), agent_idx]
        truncations = np.zeros(self.parallel, dtype=bool)
        infos = [info or {} for info in self.infos]

        observations = self.observe()
        return observations, rewards, self.terminations, truncations, infos

    def observe(self) -> Dict[str, np.ndarray]:
        black_turn = self.agent_selection == "black"
        observations = np.empty_like(self.boards)

        observations[black_turn] = self.boards[black_turn]
        white_turn = ~black_turn
        if np.any(white_turn):
            observations[white_turn] = self.boards[white_turn][:, [1, 0]]

        action_masks = (self.boards[:, 0] == 0) & (self.boards[:, 1] == 0)
        return {
            "observation": observations,
            "action_mask": action_masks.reshape(self.parallel, -1).astype(np.int8),
        }

    def reset(self, envs: np.ndarray[np.int_]) -> None:
        self.boards[envs] = 0
        self.agent_selection[envs] = "black"
        self.rewards[envs] = np.zeros(2, dtype=np.float32)
        self.terminations[envs] = False
        for i in envs:
            self.infos[i] = None

    def step(self, actions: np.ndarray) -> None:
        # Get indices of environments with actual actions
        action_mask = actions != None
        action_indices = np.where(action_mask)[0]

        if len(action_indices) > 0:
            # Validate terminated environments
            if np.any(self.terminations[action_indices]):
                raise ValueError("Cannot perform actions in terminated environments")

            # Get only the actions that are not None and convert to int
            active_actions = actions[action_mask].astype(int)
            rows, cols = divmod(active_actions, self.n)

            if np.any(active_actions < 0) or np.any(active_actions >= self.m * self.n):
                raise ValueError(f"Actions must be in range [0, {self.m * self.n - 1}]")

            player_idx = (self.agent_selection[action_mask] != "black").astype(int)
            cell_occupied = (self.boards[action_indices, 0, rows, cols] == 1) | (
                self.boards[action_indices, 1, rows, cols] == 1
            )
            if np.any(cell_occupied):
                raise ValueError("Cannot place piece on occupied cell")

            self.boards[action_indices, player_idx, rows, cols] = 1

            wins = self._check_wins(np.column_stack([action_indices, rows, cols, player_idx]))
            self.rewards[action_indices] = 0

            # Handle wins
            for i, env_idx in enumerate(action_indices[wins]):
                winner = 0 if self.agent_selection[env_idx] == "black" else 1
                self.rewards[env_idx, winner] = 1
                self.rewards[env_idx, 1 - winner] = -1
                self.terminations[env_idx] = True

            # Check for draws in environments that didn't end in a win
            non_winning_envs = action_indices[~wins]
            if len(non_winning_envs) > 0:
                draws = self._check_draws(non_winning_envs)
                for i, env_idx in enumerate(non_winning_envs[draws]):
                    # Draw: both players get 0 reward, game terminates
                    self.rewards[env_idx] = 0  # Both players get 0
                    self.terminations[env_idx] = True

        # Switch turns only for environments that had actual actions or are terminated
        turn_switch_mask = action_mask | self.terminations
        self.agent_selection = np.where(
            turn_switch_mask,
            np.where(self.agent_selection == "black", "white", "black"),
            self.agent_selection
        )

    def _check_wins(self, moves: np.ndarray) -> np.ndarray:
        """Check wins from specific move positions."""
        wins = np.zeros(len(moves), dtype=np.bool_)

        for i, (action_indices, row, col, player_idx) in enumerate(moves):
            player_board = self.boards[action_indices, player_idx]

            # Horizontal
            h_start = max(0, col - self.k + 1)
            h_end = min(self.n, col + self.k)
            h_line = player_board[row, h_start:h_end]
            wins[i] |= self._check_line(h_line)

            # Vertical
            v_start = max(0, row - self.k + 1)
            v_end = min(self.m, row + self.k)
            v_line = player_board[v_start:v_end, col]
            wins[i] |= self._check_line(v_line)

            # Diagonal \
            diag_line = []
            r, c = row - min(row, col), col - min(row, col)
            while r < self.m and c < self.n:
                diag_line.append(player_board[r, c])
                r += 1
                c += 1
            wins[i] |= self._check_line(np.array(diag_line))

            # Anti-diagonal /
            anti_diag_line = []
            r, c = row - min(row, self.n - 1 - col), col + min(row, self.n - 1 - col)
            while r < self.m and c >= 0:
                anti_diag_line.append(player_board[r, c])
                r += 1
                c -= 1
            wins[i] |= self._check_line(np.array(anti_diag_line))

        return wins

    def _check_draws(self, env_indices: np.ndarray) -> np.ndarray:
        """Check if the game is a draw (board is full) for specified environments."""
        boards_subset = self.boards[env_indices]
        return np.all((boards_subset[:, 0] == 1) | (boards_subset[:, 1] == 1), axis=(1, 2))

    def _check_line(self, line: np.ndarray) -> bool:
        """Check if line contains k consecutive pieces using numba optimization."""
        return _check_line_numba(line, self.k)


@njit
def _check_line_numba(line: np.ndarray, k: int) -> bool:
    """Numba-optimized line checking."""
    if len(line) < k:
        return False

    count = 0
    for i in range(len(line)):
        if line[i] == 1:
            count += 1
            if count == k:
                return True
        else:
            count = 0
    return False
