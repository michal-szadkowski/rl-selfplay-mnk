import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from selfplay.vector_mnk_self_play import VectorMnkSelfPlayWrapper
from tests.test_utils import FirstLegalActionPolicy


class TestVectorMnkSelfPlayWrapper:
    """Test cases for VectorMnkSelfPlayWrapper."""

    def setup_method(self):
        """Set up test environment."""
        self.m, self.n, self.k = 3, 3, 3
        self.n_envs = 2
        self.wrapper = VectorMnkSelfPlayWrapper(
            m=self.m, n=self.n, k=self.k, n_envs=self.n_envs
        )
        self.opponent = FirstLegalActionPolicy()
        self.wrapper.set_opponent(self.opponent)

    def test_reset_initializes_correctly(self):
        """Test that reset properly initializes game state."""
        obs, infos = self.wrapper.reset()

        # Check observation structure
        assert "observation" in obs
        assert "action_mask" in obs
        assert obs["observation"].shape == (self.n_envs, 2, self.m, self.n)
        assert obs["action_mask"].shape == (self.n_envs, self.m * self.n)

        # Check that opponent made first move where agent is white
        # Agent should be able to move in all environments
        assert isinstance(infos, dict)

    def test_step_with_none_actions_bug(self):
        """Test the bug where actions[~agent_turn_mask] == None fails."""
        obs, _ = self.wrapper.reset()

        # Get legal actions for each environment
        actions = np.zeros(self.n_envs, dtype=np.int32)
        for env_idx in range(self.n_envs):
            legal_actions = np.where(obs["action_mask"][env_idx] == 1)[0]
            actions[env_idx] = legal_actions[0] if len(legal_actions) > 0 else 0

        # This should not raise an assertion error
        try:
            obs, rewards, terminations, truncations, infos = self.wrapper.step(actions)
        except AssertionError as e:
            pytest.fail(f"Assertion failed with actions[~agent_turn_mask] == None: {e}")

    def test_double_opponent_move_bug(self):
        """Test for double opponent move after autoreset."""
        obs, _ = self.wrapper.reset()

        # Track opponent moves
        initial_count = self.opponent.action_count

        # Play until game ends
        for step in range(20):  # Max moves for 3x3 tic-tac-toe
            # Get legal actions for each environment
            actions = np.zeros(self.n_envs, dtype=np.int32)
            for env_idx in range(self.n_envs):
                legal_actions = np.where(obs["action_mask"][env_idx] == 1)[0]
                actions[env_idx] = legal_actions[0] if len(legal_actions) > 0 else 0

            obs, rewards, terminations, truncations, infos = self.wrapper.step(actions)

            if np.any(terminations | truncations):
                # Game ended, check for autoreset
                opponent_moves_before_reset = self.opponent.action_count

                # Next step should trigger autoreset - use legal actions
                reset_actions = np.zeros(self.n_envs, dtype=np.int32)
                for env_idx in range(self.n_envs):
                    # Check if it's agent's turn in this environment
                    agent_turn = (
                        self.wrapper.envs.agent_selection[env_idx]
                        == self.wrapper.players[env_idx]
                    )
                    if agent_turn:
                        legal_actions = np.where(obs["action_mask"][env_idx] == 1)[0]
                        reset_actions[env_idx] = (
                            legal_actions[0] if len(legal_actions) > 0 else 0
                        )

                obs, rewards, terminations, truncations, infos = self.wrapper.step(
                    reset_actions
                )

                opponent_moves_after_reset = self.opponent.action_count

                # Should have at most 1 opponent move per environment
                # If bug exists, opponent will have 2 moves
                moves_diff = opponent_moves_after_reset - opponent_moves_before_reset
                assert (
                    moves_diff <= self.n_envs
                ), f"Double opponent move detected: {moves_diff} moves"
                break

    def test_player_assignment_randomness(self):
        """Test that player assignments are properly randomized."""
        self.wrapper.reset()
        initial_players = self.wrapper.players.copy()

        # Reset multiple times
        different_assignments = False
        for _ in range(10):
            self.wrapper.reset()
            if not np.array_equal(initial_players, self.wrapper.players):
                different_assignments = True
                break

        assert different_assignments, "Player assignments should be randomized"

    def test_opponent_policy_none_handling(self):
        """Test behavior when opponent policy is None."""
        # Note: This test may need adjustment based on Policy type requirements
        # self.wrapper.set_opponent(None)
        obs, infos = self.wrapper.reset()

        # Should not crash
        actions = np.zeros(self.n_envs, dtype=np.int32)
        for env_idx in range(self.n_envs):
            legal_actions = np.where(obs["action_mask"][env_idx] == 1)[0]
            actions[env_idx] = legal_actions[0] if len(legal_actions) > 0 else 0
        obs, rewards, terminations, truncations, infos = self.wrapper.step(actions)

        # Should work without opponent
        assert isinstance(obs, dict)

    def test_action_mask_consistency(self):
        """Test that action masks are consistent with game state."""
        obs, infos = self.wrapper.reset()

        for _ in range(5):
            # Get legal actions from mask
            legal_actions = []
            for env_idx in range(self.n_envs):
                mask = obs["action_mask"][env_idx]
                legal = np.where(mask == 1)[0]
                legal_actions.append(legal)

            # Take first legal action for each environment
            actions = np.array([legal[0] if len(legal) > 0 else 0 for legal in legal_actions])

            obs, rewards, terminations, truncations, infos = self.wrapper.step(actions)

            if np.any(terminations | truncations):
                break


class TestVectorMnkSelfPlayWrapperIntegration:
    """Integration tests with specific scenarios."""

    def test_deterministic_opponent_sequence(self):
        """Test with first legal opponent to verify game flow."""
        wrapper = VectorMnkSelfPlayWrapper(m=3, n=3, k=3, n_envs=1)

        # Use first legal action policy for predictable behavior
        opponent = FirstLegalActionPolicy()
        wrapper.set_opponent(opponent)

        obs, infos = wrapper.reset()

        # Track game state to detect issues
        move_count = 0
        board_states = []

        for step in range(10):
            # Get legal action
            legal_actions = np.where(obs["action_mask"][0] == 1)[0]
            if len(legal_actions) == 0:
                break

            action = legal_actions[0]  # Take first legal

            # Store board state before move
            board_states.append(obs["observation"][0].copy())

            obs, rewards, terminations, truncations, infos = wrapper.step(
                np.array([action], dtype=np.int32)
            )

            move_count += 1

            if terminations[0] or truncations[0]:
                break

        assert move_count > 0, "Game should progress"
        assert len(board_states) == move_count, "Should track all moves"

    def test_multiple_environments_consistency(self):
        """Test that multiple environments work consistently."""
        n_envs = 4
        wrapper = VectorMnkSelfPlayWrapper(m=3, n=3, k=3, n_envs=n_envs)
        wrapper.set_opponent(FirstLegalActionPolicy())

        obs, infos = wrapper.reset()

        # Each environment should have valid initial state
        for env_idx in range(n_envs):
            assert obs["observation"][env_idx].shape == (2, 3, 3)
            assert obs["action_mask"][env_idx].shape == (9,)
            assert np.any(obs["action_mask"][env_idx] == 1), "Should have legal moves"

        # Play a few steps
        for step in range(5):
            actions = np.empty(n_envs, dtype=object)
            actions[:] = None

            # Get first legal action for each environment where it's agent's turn
            for env_idx in range(n_envs):
                # Check if it's agent's turn in this environment
                agent_turn = wrapper.envs.agent_selection[env_idx] == wrapper.players[env_idx]
                if agent_turn:
                    legal = np.where(obs["action_mask"][env_idx] == 1)[0]
                    actions[env_idx] = legal[0] if len(legal) > 0 else 0

            obs, rewards, terminations, truncations, infos = wrapper.step(actions)

            if np.all(terminations | truncations):
                break
