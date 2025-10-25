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

        self.possible_agents = ["black", "white"]

        self.single_observation_space = spaces.Dict(
            {
                "observation": spaces.Box(0, 1, (2, m, n), dtype=np.int8),
                "action_mask": spaces.Box(0, 1, (m * n,), dtype=np.int8),
            }
        )
        self.observation_space = batch_space(self.single_observation_space, self.parallel)
        self.single_action_space = spaces.Discrete(m * n)
        self.action_space = batch_space(self.single_action_space, self.parallel)

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

    def reset(self, envs: np.ndarray) -> None:
        self.boards[envs] = 0
        self.agent_selection[envs] = "black"
        self.rewards[envs] = np.zeros(2, dtype=np.float32)
        self.terminations[envs] = False
        for i in envs:
            self.infos[i] = None

    def step(self, actions: np.ndarray) -> None:
        # Get indices of environments with actual actions
        action_mask = np.array([a is not None for a in actions])
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

            wins = _check_wins_vectorized(self.boards, np.column_stack([action_indices, rows, cols, player_idx]), self.m, self.n, self.k)
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
                draws = _check_draws_numba(self.boards, non_winning_envs)
                for i, env_idx in enumerate(non_winning_envs[draws]):
                    # Draw: both players get 0 reward, game terminates
                    self.rewards[env_idx] = 0  # Both players get 0
                    self.terminations[env_idx] = True

        # Switch turns only for environments that had actual actions or are terminated
        turn_switch_mask = action_mask | self.terminations
        self.agent_selection[turn_switch_mask] = np.where(
            self.agent_selection[turn_switch_mask] == "black", "white", "black"
        )






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


@njit(parallel=True)
def _check_wins_vectorized(boards: np.ndarray, moves: np.ndarray, m: int, n: int, k: int) -> np.ndarray:
    """Vectorized win checking using numba."""
    wins = np.zeros(len(moves), dtype=np.bool_)
    
    for i in prange(len(moves)):
        env_idx, row, col, player_idx = moves[i]
        player_board = boards[env_idx, player_idx]
        
        # Check all four directions
        wins[i] = (_check_horizontal_numba(player_board, row, col, k, n) |
                  _check_vertical_numba(player_board, row, col, k, m) |
                  _check_diagonal_numba(player_board, row, col, k, m, n) |
                  _check_anti_diagonal_numba(player_board, row, col, k, m, n))
    
    return wins


@njit
def _check_horizontal_numba(board: np.ndarray, row: int, col: int, k: int, n: int) -> bool:
    """Check horizontal line for win."""
    start = max(0, col - k + 1)
    end = min(n, col + k)
    return _check_line_numba(board[row, start:end], k)


@njit
def _check_vertical_numba(board: np.ndarray, row: int, col: int, k: int, m: int) -> bool:
    """Check vertical line for win."""
    start = max(0, row - k + 1)
    end = min(m, row + k)
    return _check_line_numba(board[start:end, col], k)


@njit
def _check_diagonal_numba(board: np.ndarray, row: int, col: int, k: int, m: int, n: int) -> bool:
    """Check diagonal (\\) for win."""
    # Calculate diagonal bounds
    start_r = row - min(row, col)
    start_c = col - min(row, col)
    
    # Calculate diagonal length
    max_len = min(m - start_r, n - start_c)
    
    # Extract diagonal without Python list
    diag = np.empty(max_len, dtype=np.int8)
    for i in range(max_len):
        diag[i] = board[start_r + i, start_c + i]
    
    return _check_line_numba(diag, k)


@njit
def _check_anti_diagonal_numba(board: np.ndarray, row: int, col: int, k: int, m: int, n: int) -> bool:
    """Check anti-diagonal (/) for win."""
    # Calculate anti-diagonal bounds
    start_r = row - min(row, n - 1 - col)
    start_c = col + min(row, n - 1 - col)
    
    # Calculate anti-diagonal length
    max_len = min(m - start_r, start_c + 1)
    
    # Extract anti-diagonal without Python list
    anti_diag = np.empty(max_len, dtype=np.int8)
    for i in range(max_len):
        anti_diag[i] = board[start_r + i, start_c - i]
    
    return _check_line_numba(anti_diag, k)


@njit
def _check_draws_numba(boards: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
    """Vectorized draw checking."""
    draws = np.zeros(len(env_indices), dtype=np.bool_)
    
    for i in range(len(env_indices)):
        env_idx = env_indices[i]
        board = boards[env_idx]
        # Check if any cell is empty
        draws[i] = np.all((board[0] == 1) | (board[1] == 1))
    
    return draws
