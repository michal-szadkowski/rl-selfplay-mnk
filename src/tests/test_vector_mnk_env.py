import pytest
import numpy as np
from src.env.vector_mnk_env import VectorMnkEnv, _check_line_numba


class TestVectorMnkEnvInit:
    """Test initialization of VectorMnkEnv."""

    def test_init_basic(self):
        """Test basic initialization."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=4)

        assert env.m == 3
        assert env.n == 3
        assert env.k == 3
        assert env.parallel == 4
        assert env.num_envs == 4
        assert env.possible_agents == ["black", "white"]

    def test_init_spaces(self):
        """Test that observation and action spaces are correctly initialized."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # Check single spaces
        assert env.single_action_space.n == 9  # 3*3
        assert env.single_observation_space["observation"].shape == (2, 3, 3)
        assert env.single_observation_space["action_mask"].shape == (9,)

        # Check batched spaces
        assert env.action_space.shape == (2,)
        assert env.observation_space["observation"].shape == (2, 2, 3, 3)
        assert env.observation_space["action_mask"].shape == (2, 9)


class TestVectorMnkEnvReset:
    """Test reset functionality."""

    def test_reset_single_env(self):
        """Test resetting a single environment."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=4)

        # Modify state
        env.boards[1] = 1
        env.agent_selection[1] = "white"
        env.rewards[1] = [1, -1]
        env.terminations[1] = True
        env.infos[1] = {"test": "data"}

        # Reset env 1
        env.reset(np.array([1]))

        # Check reset
        assert np.all(env.boards[1] == 0)
        assert env.agent_selection[1] == "black"
        assert np.all(env.rewards[1] == 0)
        assert not env.terminations[1]
        assert env.infos[1] is None

        # Check others unchanged
        assert env.agent_selection[0] == "black"
        assert env.terminations[0] == False

    def test_reset_multiple_envs(self):
        """Test resetting multiple environments."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=4)

        # Modify state
        env.boards[:2] = 1
        env.agent_selection[:2] = "white"
        env.terminations[:2] = True

        # Reset first two envs
        env.reset(np.array([0, 1]))

        # Check reset
        for i in [0, 1]:
            assert np.all(env.boards[i] == 0)
            assert env.agent_selection[i] == "black"
            assert np.all(env.rewards[i] == 0)
            assert not env.terminations[i]
            assert env.infos[i] is None

        # Check others unchanged
        assert env.agent_selection[2] == "black"
        assert env.terminations[2] == False


class TestVectorMnkEnvObserve:
    """Test observation generation."""

    def test_observe_black_turn(self):
        """Test observation on black's turn."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # Place some pieces
        env.boards[0, 0, 0, 0] = 1  # Black piece
        env.boards[0, 1, 0, 1] = 1  # White piece
        env.agent_selection[0] = "black"

        obs = env.observe()

        # Black turn should see [black_board, white_board]
        assert obs["observation"][0, 0, 0, 0] == 1  # Black piece
        assert obs["observation"][0, 1, 0, 1] == 1  # White piece

        # Action mask should be correct
        expected_mask = np.ones(9, dtype=np.int8)
        expected_mask[0] = 0  # Occupied by black
        expected_mask[1] = 0  # Occupied by white
        np.testing.assert_array_equal(obs["action_mask"][0], expected_mask)

    def test_observe_white_turn(self):
        """Test observation on white's turn."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # Place some pieces
        env.boards[0, 0, 0, 0] = 1  # Black piece
        env.boards[0, 1, 0, 1] = 1  # White piece
        env.agent_selection[0] = "white"

        obs = env.observe()

        # White turn should see [white_board, black_board] (swapped)
        assert obs["observation"][0, 0, 0, 1] == 1  # White piece in first channel
        assert obs["observation"][0, 1, 0, 0] == 1  # Black piece in second channel

    def test_observe_mixed_turns(self):
        """Test observation with mixed agent selections."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=3)

        # Set different turns
        env.agent_selection = np.array(["black", "white", "black"])

        obs = env.observe()

        # Check shapes
        assert obs["observation"].shape == (3, 2, 3, 3)
        assert obs["action_mask"].shape == (3, 9)

        # All should have correct action masks (empty board = all True)
        assert np.all(obs["action_mask"] == 1)


class TestVectorMnkEnvStep:
    """Test step functionality."""

    def test_step_valid_move(self):
        """Test valid move execution."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)
        actions = np.array([0, 4])  # Place at (0,0) and (1,1)

        env.step(actions)

        # Check pieces are placed - both are black's first move
        assert env.boards[0, 0, 0, 0] == 1  # Black at (0,0) in env 0
        assert env.boards[1, 0, 1, 1] == 1  # Black at (1,1) in env 1

        # Check turn switched
        assert env.agent_selection[0] == "white"
        assert env.agent_selection[1] == "white"

    def test_step_none_actions_terminated(self):
        """Test None actions in terminated environments."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # Terminate first environment
        env.terminations[0] = True

        # Mix of None and valid actions
        actions = np.array([None, 0])
        env.step(actions)

        # Should not raise error
        assert env.agent_selection[0] == "white"  # Turn switched for terminated env
        assert env.agent_selection[1] == "white"  # Turn switched for actual action

    def test_step_none_actions_active_valid(self):
        """Test that None actions in active environments are valid."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # None action in active environment should be valid
        actions = np.array([None, 0])
        env.step(actions)  # Should not raise error

        # Check that only the second environment had a move
        assert env.boards[0, 0, 0, 0] == 0  # No move in env 0
        assert env.boards[1, 0, 0, 0] == 1  # Move in env 1
        
        # Check turn switching - only env 1 should have switched turn
        assert env.agent_selection[0] == "black"  # No turn switch for None action
        assert env.agent_selection[1] == "white"  # Turn switched for actual action

    def test_step_turn_switching_logic(self):
        """Test specific turn switching logic for None actions in active vs terminated environments."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=3)

        # Set up different states:
        # Env 0: active, None action - should NOT switch turn
        # Env 1: active, real action - should switch turn  
        # Env 2: terminated, None action - should switch turn
        env.terminations[2] = True

        # Initial turns should all be black
        assert np.all(env.agent_selection == "black")

        # Apply actions
        actions = np.array([None, 0, None])
        env.step(actions)

        # Check turn switching results
        assert env.agent_selection[0] == "black"  # Active + None = no turn switch
        assert env.agent_selection[1] == "white"  # Active + action = turn switch
        assert env.agent_selection[2] == "white"  # Terminated + None = turn switch

    def test_step_terminated_env_action_error(self):
        """Test that actions in terminated environments raise error."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # Terminate first environment
        env.terminations[0] = True

        # Action in terminated environment should raise
        actions = np.array([0, 1])
        with pytest.raises(ValueError, match="Cannot perform actions"):
            env.step(actions)

    def test_step_invalid_action_range(self):
        """Test invalid action range."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # Action out of range
        actions = np.array([9, 0])  # 9 is invalid for 3x3
        with pytest.raises(ValueError, match="Actions must be in range"):
            env.step(actions)

    def test_step_negative_action(self):
        """Test negative action."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # Negative action
        actions = np.array([-1, 0])
        with pytest.raises(ValueError, match="Actions must be in range"):
            env.step(actions)

    def test_step_occupied_cell(self):
        """Test placing piece on occupied cell."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # Place first piece
        env.boards[0, 0, 0, 0] = 1  # Black at (0,0)

        # Try to place on same cell
        actions = np.array([0, 1])
        with pytest.raises(ValueError, match="Cannot place piece on occupied cell"):
            env.step(actions)


class TestVectorMnkWinDetection:
    """Test win detection functionality."""

    def test_horizontal_win(self):
        """Test horizontal win detection."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=1)

        # Black: (0,0), (0,1), White: (1,0), Black: (0,2) -> Black wins
        env.boards[0, 0, 0, 0] = 1
        env.boards[0, 1, 1, 0] = 1
        env.boards[0, 1, 1, 0] = 1
        env.boards[0, 0, 0, 2] = 1

        actions = np.array([4])  # White plays elsewhere
        env.step(actions)

        # Now black plays winning move
        actions = np.array([2])  # (0,2) - already there, this is wrong approach

        # Better approach: simulate actual game
        env.reset(np.array([0]))
        actions = np.array([0])  # Black (0,0)
        env.step(actions)
        actions = np.array([3])  # White (1,0)
        env.step(actions)
        actions = np.array([1])  # Black (0,1)
        env.step(actions)
        actions = np.array([4])  # White (1,1)
        env.step(actions)
        actions = np.array([2])  # Black (0,2) - wins!
        env.step(actions)

        assert env.terminations[0]
        assert env.rewards[0, 0] == 1  # Black wins
        assert env.rewards[0, 1] == -1  # White loses

    def test_horizontal_win_edge(self):
        """Test horizontal win on board edge."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=1)

        # Win on top row: (0,0), (0,1), (0,2)
        moves = [0, 3, 1, 4, 2]  # Black wins on top edge
        for i, action in enumerate(moves):
            env.step(np.array([action]))

        assert env.terminations[0]
        assert env.rewards[0, 0] == 1  # Black wins

    def test_vertical_win_edge(self):
        """Test vertical win on board edge."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=1)

        # Win on left column: (0,0), (1,0), (2,0)
        moves = [0, 1, 3, 2, 6]  # Black wins on left edge
        for i, action in enumerate(moves):
            env.step(np.array([action]))

        assert env.terminations[0]
        assert env.rewards[0, 0] == 1  # Black wins

    def test_diagonal_win_edge(self):
        """Test diagonal win starting from corner."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=1)

        # Win from top-left corner: (0,0), (1,1), (2,2)
        moves = [0, 1, 4, 2, 8]  # Black wins from corner
        for i, action in enumerate(moves):
            env.step(np.array([action]))

        assert env.terminations[0]
        assert env.rewards[0, 0] == 1  # Black wins

    def test_vertical_win(self):
        """Test vertical win detection."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=1)

        # Simulate vertical win for black
        moves = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]  # Black wins
        for i, (r, c) in enumerate(moves):
            action = r * 3 + c
            env.step(np.array([action]))

        assert env.terminations[0]
        assert env.rewards[0, 0] == 1  # Black wins

    def test_diagonal_win(self):
        """Test diagonal win detection."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=1)

        # Simulate diagonal win for black
        moves = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]  # Black wins
        for i, (r, c) in enumerate(moves):
            action = r * 3 + c
            env.step(np.array([action]))

        assert env.terminations[0]
        assert env.rewards[0, 0] == 1  # Black wins

    def test_anti_diagonal_win(self):
        """Test anti-diagonal win detection."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=1)

        # Simulate anti-diagonal win for black
        moves = [(0, 2), (0, 1), (1, 1), (1, 0), (2, 0)]  # Black wins
        for i, (r, c) in enumerate(moves):
            action = r * 3 + c
            env.step(np.array([action]))

        assert env.terminations[0]
        assert env.rewards[0, 0] == 1  # Black wins


class TestVectorMnkDrawDetection:
    """Test draw detection."""

    def test_full_board_draw(self):
        """Test draw when board is full."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=1)

        # Simulate a full game without wins
        moves = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0), (1, 2), (2, 1), (2, 0), (2, 2)]
        for i, (r, c) in enumerate(moves):
            action = r * 3 + c
            env.step(np.array([action]))

        assert env.terminations[0]
        assert env.rewards[0, 0] == 0  # Draw
        assert env.rewards[0, 1] == 0  # Draw


class TestVectorMnkParallelGames:
    """Test parallel game functionality."""

    def test_different_game_states(self):
        """Test multiple environments with different states."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=3)

        # Different actions in each env
        actions = np.array([0, 1, 4])
        env.step(actions)

        # Check each env has different board state - all are black's first move
        assert env.boards[0, 0, 0, 0] == 1  # Env 0: black at (0,0)
        assert env.boards[1, 0, 0, 1] == 1  # Env 1: black at (0,1)
        assert env.boards[2, 0, 1, 1] == 1  # Env 2: black at (1,1)

        # All should have switched turns to white
        expected_turns = np.array(["white", "white", "white"])
        np.testing.assert_array_equal(env.agent_selection, expected_turns)

    def test_mixed_termination_states(self):
        """Test mixed termination states across environments."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # Let first env end quickly
        # This would require setting up a winning position manually
        env.boards[0, 0, 0, 0] = 1
        env.boards[0, 0, 0, 1] = 1
        env.boards[0, 0, 0, 2] = 1  # Black wins horizontally
        env.terminations[0] = True

        # Second env continues
        actions = np.array([None, 4])  # None for terminated, valid for active
        env.step(actions)

        assert env.terminations[0]  # Still terminated
        assert not env.terminations[1]  # Still active


class TestVectorMnkEnvLast:
    """Test last() method."""

    def test_last_method(self):
        """Test last() method returns correct format."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        obs, rewards, terminations, truncations, infos = env.last()

        # Check types and shapes
        assert isinstance(obs, dict)
        assert "observation" in obs
        assert "action_mask" in obs
        assert obs["observation"].shape == (2, 2, 3, 3)
        assert obs["action_mask"].shape == (2, 9)
        assert rewards.shape == (2,)
        assert terminations.shape == (2,)
        assert truncations.shape == (2,)
        assert len(infos) == 2
        assert np.all(truncations == False)  # No truncations in this env

    def test_last_rewards_by_agent(self):
        """Test that rewards are correctly selected by current agent."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # Set different agent selections
        env.agent_selection = np.array(["black", "white"])
        env.rewards[0] = [1, -1]  # Black won
        env.rewards[1] = [-1, 1]  # White won

        _, rewards, _, _, _ = env.last()

        # Should return reward for current agent
        assert rewards[0] == 1  # Black's reward in env 0
        assert rewards[1] == 1  # White's reward in env 1


class TestCheckLineNumba:
    """Test the numba-optimized line checking function."""

    def test_check_line_exact_k(self):
        """Test line with exactly k consecutive pieces."""
        line = np.array([1, 1, 1, 0, 1])
        result = _check_line_numba(line, 3)
        assert result == True

    def test_check_line_more_than_k(self):
        """Test line with more than k consecutive pieces."""
        line = np.array([1, 1, 1, 1, 0])
        result = _check_line_numba(line, 3)
        assert result == True

    def test_check_line_less_than_k(self):
        """Test line with less than k consecutive pieces."""
        line = np.array([1, 1, 0, 1, 1])
        result = _check_line_numba(line, 3)
        assert result == False

    def test_check_line_empty(self):
        """Test empty line."""
        line = np.array([0, 0, 0, 0, 0])
        result = _check_line_numba(line, 3)
        assert result == False

    def test_check_line_shorter_than_k(self):
        """Test line shorter than k."""
        line = np.array([1, 1])
        result = _check_line_numba(line, 3)
        assert result == False

    def test_check_line_exactly_k_length(self):
        """Test line exactly k length."""
        line = np.array([1, 1, 1])
        result = _check_line_numba(line, 3)
        assert result == True

    def test_check_line_multiple_segments(self):
        """Test line with multiple segments."""
        line = np.array([1, 1, 0, 1, 1, 1])
        result = _check_line_numba(line, 3)
        assert result == True

    def test_check_line_k_in_middle(self):
        """Test line with k consecutive pieces in the middle."""
        line = np.array([0, 1, 1, 1, 0])
        result = _check_line_numba(line, 3)
        assert result == True

    def test_check_line_zero_in_middle(self):
        """Test line with zero interrupting consecutive pieces."""
        line = np.array([1, 0, 1, 1, 1])
        result = _check_line_numba(line, 3)
        assert result == True

    def test_check_line_two_segments(self):
        """Test line with two segments of consecutive pieces."""
        line = np.array([1, 1, 0, 1, 1])
        result = _check_line_numba(line, 3)
        assert result == False


class TestVectorMnkEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_board(self):
        """Test smallest possible board (1x1)."""
        env = VectorMnkEnv(m=1, n=1, k=1, parallel=1)

        # First move should win
        actions = np.array([0])
        env.step(actions)

        assert env.terminations[0]
        assert env.rewards[0, 0] == 1  # Black wins


class TestVectorMnkObservationEdgeCases:
    """Test observation edge cases."""

    def test_observe_mixed_termination_states(self):
        """Test observe() with mixed terminated and active environments."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=3)

        # Set up different states
        # Env 0: active, black's turn, some pieces on board
        env.boards[0, 0, 0, 0] = 1  # Black piece
        env.boards[0, 1, 0, 1] = 1  # White piece
        env.agent_selection[0] = "black"

        # Env 1: terminated, white won
        env.boards[1, 0, 0, 0] = 1  # Black piece
        env.boards[1, 0, 0, 1] = 1  # Black piece
        env.boards[1, 0, 0, 2] = 1  # Black piece (wins)
        env.boards[1, 1, 1, 0] = 1  # White piece
        env.boards[1, 1, 1, 1] = 1  # White piece
        env.terminations[1] = True
        env.agent_selection[1] = "white"
        env.rewards[1] = [-1, 1]  # Black lost, white won

        # Env 2: active, white's turn, empty board
        env.agent_selection[2] = "white"

        obs = env.observe()

        # Check shapes
        assert obs["observation"].shape == (3, 2, 3, 3)
        assert obs["action_mask"].shape == (3, 9)

        # Check env 0 (black turn, should see [black, white])
        assert obs["observation"][0, 0, 0, 0] == 1  # Black piece in first channel
        assert obs["observation"][0, 1, 0, 1] == 1  # White piece in second channel
        expected_mask_0 = np.ones(9, dtype=np.int8)
        expected_mask_0[0] = 0  # Occupied by black
        expected_mask_0[1] = 0  # Occupied by white
        np.testing.assert_array_equal(obs["action_mask"][0], expected_mask_0)

        # Check env 1 (terminated, should still return observation)
        # Env 1 is white's turn, so observation is swapped: [white, black]
        assert obs["observation"][1, 0, 1, 0] == 1  # White pieces in first channel
        assert obs["observation"][1, 0, 1, 1] == 1
        assert obs["observation"][1, 1, 0, 0] == 1  # Black pieces in second channel
        assert obs["observation"][1, 1, 0, 1] == 1
        assert obs["observation"][1, 1, 0, 2] == 1
        # Action mask should reflect actual occupied cells
        expected_mask_1 = np.ones(9, dtype=np.int8)
        expected_mask_1[0] = 0  # (0,0) occupied by black
        expected_mask_1[1] = 0  # (0,1) occupied by black
        expected_mask_1[2] = 0  # (0,2) occupied by black
        expected_mask_1[3] = 0  # (1,0) occupied by white
        expected_mask_1[4] = 0  # (1,1) occupied by white
        np.testing.assert_array_equal(obs["action_mask"][1], expected_mask_1)

        # Check env 2 (white turn, should see [white, black] - swapped)
        assert np.all(obs["observation"][2] == 0)  # Empty board
        assert np.all(obs["action_mask"][2] == 1)  # All moves available


class TestVectorMnkErrorHandling:
    """Test error handling and validation."""

    def test_invalid_dimensions(self):
        """Test invalid board dimensions."""
        # Zero dimensions should fail in gymnasium.Discrete
        with pytest.raises(AssertionError):
            env = VectorMnkEnv(m=0, n=3, k=1, parallel=1)  # Zero rows

        with pytest.raises(AssertionError):
            env = VectorMnkEnv(m=3, n=0, k=1, parallel=1)  # Zero columns

    def test_k_larger_than_dimensions(self):
        """Test k larger than board dimensions."""
        env = VectorMnkEnv(m=3, n=3, k=5, parallel=1)

        # Game should be impossible to win (always draw)
        # Fill board
        moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        for i, (r, c) in enumerate(moves):
            action = r * 3 + c
            env.step(np.array([action]))

        assert env.terminations[0]  # Should terminate (draw)
        assert env.rewards[0, 0] == 0  # Draw reward


class TestVectorMnkIntegration:
    """Integration tests combining multiple features."""

    def test_complete_game_simulation(self):
        """Test complete game simulation with all features."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=2)

        # Simulate complete games in both environments
        # Env 0: Black wins horizontally
        # Env 1: Draw

        moves_env0 = [0, 3, 1, 4, 2]  # Black wins
        moves_env1 = [0, 1, 2, 4, 3, 5, 7, 6, 8]  # Draw

        max_moves = max(len(moves_env0), len(moves_env1))

        for step in range(max_moves):
            actions = [None, None]

            if step < len(moves_env0):
                actions[0] = moves_env0[step]
            if step < len(moves_env1):
                actions[1] = moves_env1[step]

            env.step(np.array(actions))

        # Check final states
        assert env.terminations[0]  # Both should be terminated
        assert env.terminations[1]

        assert env.rewards[0, 0] == 1  # Black won env 0
        assert env.rewards[0, 1] == -1

        assert env.rewards[1, 0] == 0  # Draw in env 1
        assert env.rewards[1, 1] == 0

    def test_reset_and_replay(self):
        """Test resetting environment and playing new game."""
        env = VectorMnkEnv(m=3, n=3, k=3, parallel=1)

        # Play and complete one game
        moves = [0, 1, 2, 4, 6, 8, 5, 3, 7]  # Draw
        for action in moves:
            env.step(np.array([action]))

        assert env.terminations[0]

        # Reset and play again
        env.reset(np.array([0]))

        assert not env.terminations[0]
        assert env.agent_selection[0] == "black"
        assert np.all(env.boards[0] == 0)

        # Play first move of new game
        env.step(np.array([0]))
        assert env.boards[0, 0, 0, 0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
