import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .constants import PLAYER_BLACK, PLAYER_WHITE


class TorchVectorMnkEnv:
    def __init__(self, m: int, n: int, k: int, num_envs: int, device: str = "cuda"):
        assert m >= k and n >= k, f"Board ({m}x{n}) is too small for k={k}"

        self.m = m
        self.n = n
        self.k = k
        self.num_envs = num_envs
        self.device = device

        self.boards = torch.zeros((num_envs, 2, m, n), dtype=torch.float32, device=device)
        self.current_player = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.move_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_moves = m * n

        self.env_indices = torch.arange(num_envs, device=device)

        self._init_kernels()

    def _init_kernels(self):
        self.h_kernel = torch.ones((1, 1, 1, self.k), device=self.device)
        self.v_kernel = torch.ones((1, 1, self.k, 1), device=self.device)

        d1 = torch.eye(self.k, device=self.device).reshape(1, 1, self.k, self.k)
        d2 = torch.fliplr(torch.eye(self.k, device=self.device)).reshape(1, 1, self.k, self.k)
        self.diag_kernels = torch.cat([d1, d2], dim=0)

    def reset(self, env_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if env_indices is None:
            self.boards.zero_()
            self.current_player.zero_()
            self.move_counts.zero_()
        else:
            self.boards[env_indices] = 0
            self.current_player[env_indices] = PLAYER_BLACK
            self.move_counts[env_indices] = 0

        return self.observe()

    def observe(self) -> Dict[str, torch.Tensor]:
        occupied = (self.boards != 0.0).any(dim=1)
        action_mask = (~occupied).flatten(1)

        return {
            "observation": self.boards.clone(),
            "action_mask": action_mask,
        }

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        return self.step_subset(actions, self.env_indices)

    def step_subset(
        self, actions: torch.Tensor, active_indices: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:

        rows = actions.div(self.n, rounding_mode="floor")
        cols = actions % self.n

        players = self.current_player[active_indices]
        self.boards[active_indices, players, rows, cols] = 1.0
        self.move_counts[active_indices] += 1

        winners = self._check_wins(active_indices, players)
        draws = (self.move_counts[active_indices] >= self.max_moves) & (~winners)
        local_dones = winners | draws

        rewards = torch.zeros(self.num_envs, device=self.device)
        if winners.any():
            rewards[active_indices[winners]] = 1.0

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dones[active_indices] = local_dones

        self.current_player[active_indices] ^= 1

        return self.observe(), rewards, dones

    def _validate_action_bounds(self, actions: torch.Tensor):
        if ((actions < 0) | (actions >= self.m * self.n)).any():
            bad_idx = torch.nonzero((actions < 0) | (actions >= self.m * self.n))[0].item()
            bad_val = actions[bad_idx].item()
            raise ValueError(
                f"Action out of bounds! Env received {bad_val}, expected [0, {self.m * self.n - 1}]"
            )

    def _validate_moves(self, active_indices, rows, cols):
        target_black = self.boards[active_indices, PLAYER_BLACK, rows, cols]
        target_white = self.boards[active_indices, PLAYER_WHITE, rows, cols]

        is_occupied = (target_black > 0.5) | (target_white > 0.5)
        if is_occupied.any():
            bad_local_idx = torch.nonzero(is_occupied)[0].item()
            bad_env_idx = active_indices[bad_local_idx].item()
            raise ValueError(
                f"Illegal Move: Env {bad_env_idx} tried to play in occupied cell."
            )

    def _check_wins(self, active_indices, players) -> torch.Tensor:
        player_boards = self.boards[active_indices, players].unsqueeze(1)
        threshold = self.k - 0.1

        h_wins = F.conv2d(player_boards, self.h_kernel) > threshold
        v_wins = F.conv2d(player_boards, self.v_kernel) > threshold
        d_wins = F.conv2d(player_boards, self.diag_kernels) > threshold

        batch_size = len(active_indices)
        return (
            h_wins.view(batch_size, -1).any(dim=1)
            | v_wins.view(batch_size, -1).any(dim=1)
            | d_wins.view(batch_size, -1).any(dim=1)
        )
