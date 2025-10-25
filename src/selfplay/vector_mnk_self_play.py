import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import gymnasium as gym
from gymnasium.vector.utils import batch_space
from gymnasium.vector import AutoresetMode

from src.env.vector_mnk_env import VectorMnkEnv
from src.selfplay.policy import VectorNNPolicy, BatchNNPolicy, Policy


class VectorMnkSelfPlayWrapper(gym.vector.VectorEnv):
    """Vector self-play wrapper optimized for VectorMnkEnv."""

    def __init__(self, m: int, n: int, k: int, n_envs: int = 1):
        """
        Initialize vector self-play wrapper for MNK game.

        Args:
            m, n, k: Board dimensions and win condition
            n_envs: Number of parallel environments
        """
        # Set required VectorEnv attributes
        self.metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

        self.m = m
        self.n = n
        self.k = k
        self.num_envs = n_envs

        # Create vector environments
        self.envs = VectorMnkEnv(m=m, n=n, k=k, parallel=n_envs)

        # Setup spaces
        self.single_observation_space = self.envs.single_observation_space
        self.observation_space = self.envs.observation_space
        self.single_action_space = self.envs.single_action_space
        self.action_space = self.envs.action_space

        # Initialize state
        self.opponent_policy: Optional[Policy] = None
        self.players = np.random.choice(
            ["black", "white"], n_envs
        )  # Who external agent plays in each env

        # Autoreset tracking
        self._autoreset_envs = np.zeros(n_envs, dtype=bool)

    def set_opponent(self, opponent_policy: Policy) -> None:
        """Set opponent policy for self-play."""
        self.opponent_policy = opponent_policy

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset all environments and start new games.

        Returns:
            observations: Batch of initial observations
            infos: Additional information
        """
        # Reset all environments
        env_indices = np.arange(self.num_envs)
        self.envs.reset(env_indices)

        # Randomize who external agent plays as in each environment
        self.players = np.random.choice(["black", "white"], self.num_envs)

        self._autoreset_envs = np.zeros(self.num_envs, dtype=bool)

        # Let opponent make first move where external agent is white (black starts first)
        self._opponent_step()

        # Return current observations from VectorMnkEnv
        obs, rewards, terminations, truncations, env_infos = self.envs.last()

        # Process infos like the original wrapper
        infos = {}
        valid_infos = [info for info in env_infos if info]
        for info in valid_infos:
            infos.update(info)

        return obs, infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute agent actions, then opponent responses.

        Args:
            actions: Agent actions (for environments where external agent's turn)

        Returns:
            observations: Batch of new observations
            rewards: Batch of rewards (from agent's perspective)
            terminations: Batch of termination flags
            truncations: Batch of truncation flags
            infos: Additional information
        """
        assert np.all(self.envs.agent_selection == self.players)

        assert np.all(
            actions[~self._autoreset_envs] != None
        ), "External agent provided None action for their action but env was not terminated"

        agent_turn_mask = ~self._autoreset_envs

        self._handle_autoreset()

        step_actions = np.full(self.num_envs, None, dtype=object)
        step_actions[agent_turn_mask] = actions[agent_turn_mask]

        self.envs.step(step_actions)

        self._opponent_step()

        obs, rewards, terminations, truncations, env_infos = self.envs.last()

        assert np.all(
            self.envs.agent_selection == self.players
        ), "Agent turn mismatch after opponent step"

        infos = {}
        valid_infos = [info for info in env_infos if info]
        for info in valid_infos:
            infos.update(info)

        self._autoreset_envs = terminations.astype(bool) | truncations.astype(bool)

        return obs, rewards, terminations, truncations, infos

    def _opponent_step(self) -> None:
        """Execute opponent move in all environments where opponent should play."""
        assert self.opponent_policy is not None

        # Get current state to check terminations
        _, terminations, truncations, _, _ = self.envs.last()

        # Determine which environments opponent should play in
        opponent_envs_mask = (
            (self.envs.agent_selection != self.players)
            & ~self._autoreset_envs
            & ~(terminations.astype(bool) | truncations.astype(bool))
        )

        # Get observations and prepare batch for opponent
        obs = self.envs.observe()
        opponent_observations = {
            "observation": obs["observation"][opponent_envs_mask],
            "action_mask": obs["action_mask"][opponent_envs_mask],
        }

        # Get and apply opponent actions
        opponent_actions = self.opponent_policy.act(opponent_observations)

        step_actions = np.full(self.num_envs, None, dtype=object)
        step_actions[opponent_envs_mask] = opponent_actions
        self.envs.step(step_actions)

    def _handle_autoreset(self) -> None:
        """Handle autoreset for terminated environments."""
        if not np.any(self._autoreset_envs):
            return

        # Reset terminated environments
        env_indices_to_reset = np.where(self._autoreset_envs)[0]
        self.envs.reset(env_indices_to_reset)

        # Reset autoreset flag
        self._autoreset_envs[env_indices_to_reset] = False

        # Randomize who external agent plays as in reset environments
        self.players[env_indices_to_reset] = np.random.choice(
            ["black", "white"], len(env_indices_to_reset)
        )
