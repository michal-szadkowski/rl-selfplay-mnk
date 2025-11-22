import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class TorchVectorMnkEnv:
    def __init__(self, m: int, n: int, k: int, num_envs: int, device: str = "cuda"):
        assert m >= k and n >= k, (
            f"Plansza ({m}x{n}) jest za mała na warunek zwycięstwa k={k}"
        )

        self.m = m
        self.n = n
        self.k = k
        self.num_envs = num_envs
        self.device = device

        # Stan gry (Tensor w VRAM)
        # Shape: (num_envs, 2, m, n). Kanał 0: Black, Kanał 1: White
        self.boards = torch.zeros(
            (num_envs, 2, m, n), dtype=torch.float32, device=device
        )

        # 0 = Black, 1 = White
        self.current_player = torch.zeros(num_envs, dtype=torch.long, device=device)

        # OPTYMALIZACJA: Licznik ruchów zamiast sumowania planszy
        self.move_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_moves = m * n

        # Pre-alokacja pomocnicza
        self.env_indices = torch.arange(num_envs, device=device)

        # 1. Filtr Poziomy (1 x K)
        self.h_kernel = torch.ones((1, 1, 1, self.k), device=device)

        # 2. Filtr Pionowy (K x 1)
        self.v_kernel = torch.ones((1, 1, self.k, 1), device=device)

        # 3. Filtry Diagonalne (złączone, bo oba są K x K)
        d1 = torch.eye(self.k, device=device).reshape(1, 1, self.k, self.k)
        d2 = torch.fliplr(torch.eye(self.k, device=device)).reshape(
            1, 1, self.k, self.k
        )
        self.diag_kernels = torch.cat([d1, d2], dim=0)  # Shape: (2, 1, k, k)

    def reset(
        self, env_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if env_indices is None:
            self.boards.zero_()
            self.current_player.zero_()
            self.move_counts.zero_()
        else:
            self.boards[env_indices] = 0
            self.current_player[env_indices] = 0  # Zawsze zaczyna Black
            self.move_counts[env_indices] = 0

        return self.observe()

    def observe(self) -> Dict[str, torch.Tensor]:
        # Zwracamy SUROWY stan planszy - obiektywny stan świata.
        # Kanał 0: ZAWSZE Black
        # Kanał 1: ZAWSZE White

        # Maska akcji: 1 gdzie puste, 0 gdzie zajęte
        # Sprawdzamy czy cokolwiek stoi na polu (wartość != 0)
        occupied = (self.boards != 0.0).any(
            dim=1
        )  # (num_envs, m, n) - True gdzie coś stoi
        action_mask = (~occupied).flatten(1)  # (num_envs, m*n) bool - True gdzie puste

        return {
            "observation": self.boards.clone(),  # (N, 2, M, N) - kopia bezpieczeństwa
            "action_mask": action_mask,
        }

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Step all environments - convenience method that calls step_subset with all indices."""
        all_indices = torch.arange(self.num_envs, device=self.device)
        return self.step_subset(actions, all_indices)

    def step_subset(
        self, actions: torch.Tensor, active_indices: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Wykonaj ruchy tylko w wybranych środowiskach.

        Args:
            actions: Tensor (len(active_indices),) int64 - akcje dla aktywnych środowisk
            active_indices: Tensor (len(active_indices),) int64 - indeksy aktywnych środowisk

        Returns:
            observations, rewards, dones, info
        """
        # 1. Konwersja akcji na (row, col)
        rows = actions.div(self.n, rounding_mode="floor")
        cols = actions % self.n

        # --- VALIDATION START ---
        # Sprawdzenie czy pole jest puste
        current_players = self.current_player[active_indices]

        # Sprawdzamy czy cokolwiek tam stoi (Black=1 lub White=1)
        target_cells_black = self.boards[active_indices, 0, rows, cols]
        target_cells_white = self.boards[active_indices, 1, rows, cols]

        is_occupied = (target_cells_black > 0.5) | (target_cells_white > 0.5)

        if is_occupied.any():
            # Znajdź pierwszy błędny indeks dla debugowania
            bad_idx = torch.nonzero(is_occupied)[0].item()
            real_env_idx = active_indices[bad_idx].item()
            raise ValueError(
                f"Illegal Move: Env {real_env_idx} tried to play in occupied cell ({rows[bad_idx]}, {cols[bad_idx]})"
            )
        # --- VALIDATION END ---

        # Wykonanie ruchu (bezpieczne, bo sprawdziliśmy)
        self.boards[active_indices, current_players, rows, cols] = 1.0

        # OPTYMALIZACJA 1: Inkrementacja licznika zamiast sumowania planszy
        self.move_counts[active_indices] += 1

        # 2. Sprawdzenie Zwycięstwa (Logika ta sama, zapis krótszy)
        # Wyciągamy widok plansz tylko raz
        player_boards = self.boards[active_indices, current_players].unsqueeze(1)
        threshold = self.k - 0.1

        # Sploty
        h_wins = F.conv2d(player_boards, self.h_kernel) > threshold
        v_wins = F.conv2d(player_boards, self.v_kernel) > threshold
        d_wins = F.conv2d(player_boards, self.diag_kernels) > threshold

        # Scalanie wyników (flatten przestrzenny -> any)
        batch_size = len(active_indices)
        winners = (
            h_wins.view(batch_size, -1).any(dim=1)
            | v_wins.view(batch_size, -1).any(dim=1)
            | d_wins.view(batch_size, -1).any(dim=1)
        )

        # 3. Sprawdzenie Remisów
        # Remis jest wtedy, gdy plansza pełna I nikt nie wygrał
        draws = (self.move_counts[active_indices] >= self.max_moves) & (~winners)

        active_dones = winners | draws

        # 4. Przygotowanie Wyników (Bezpośredni zapis)
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Zamiast tworzyć rewards_subset, piszemy wprost:
        # Używamy active_indices przefiltrowanych przez winners
        if winners.any():
            winning_indices = active_indices[winners]
            rewards[winning_indices] = 1.0

        dones[active_indices] = active_dones

        # 5. Zmiana Gracza (OPTYMALIZACJA: Flip First)
        # Odwracamy gracza we WSZYSTKICH aktywnych środowiskach.
        self.current_player[active_indices] ^= 1

        # Zwracamy obserwację po wszystkich zmianach
        return self.observe(), rewards, dones
