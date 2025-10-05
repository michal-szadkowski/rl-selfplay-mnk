from pettingzoo import AECEnv
from .mnk_game import MnkGame, Color

import gymnasium
import numpy as np
from gymnasium import spaces

from pettingzoo.utils import AgentSelector, wrappers


def create_mnk_env(m: int, n: int, k: int, render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = MnkEnv(m, n, k, render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class MnkEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "mnk"}

    def __init__(self, m: int, n: int, k: int, render_mode=None):
        self.possible_agents = ["black", "white"]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.action_spaces = {i: spaces.Discrete(m * n) for i in self.possible_agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=[2, m, n], dtype=np.int8
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=[m * n], dtype=np.int8
                    ),
                }
            )
            for i in self.possible_agents
        }
        self.render_mode = render_mode

        self.m = m
        self.n = n
        self.k = k
        self.reset()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        return self.observations[agent]

    def set_observations(self):
        self.observations = {
            agent: {
                "observation": self.game.board(
                    Color.Black if agent == "black" else Color.White
                ).astype(np.int8),
                "action_mask": (~(self.game.black | self.game.white)).flatten(),
            }
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):

        self.game = MnkGame(self.m, self.n, self.k)

        self.agents = self.possible_agents[:]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.infos = {agent: {} for agent in self.agents}

        self.set_observations()

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        act = np.unravel_index(action, (self.m, self.n))

        finished = self.game.put(act[0], act[1])

        if finished:
            winner = self.game.get_winner()
            self.rewards[self.agents[0]] = (
                1 if winner == Color.Black else (0 if winner is None else -1)
            )
            self.rewards[self.agents[1]] = (
                1 if winner == Color.White else (0 if winner is None else -1)
            )

            self.terminations = {i: True for i in self.agents}
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()

        self.set_observations()

        if self.render_mode == "human":
            self.render(act)

    def render(self, act=None):
        """
        Renders the environment. In human mode, it prints a formatted board
        to the terminal, including row/column numbers and action indices for empty cells.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) != 2:
            print("Game over")
            return

        row_label_w = len(str(self.m - 1))
        cell_w = len(str(self.m * self.n - 1)) + 2

        # Header
        header = " " * (row_label_w + 1)
        for j in range(self.n):
            header += f"{j:^{cell_w+1}}"
        print(header)

        # Top border
        border = " " * row_label_w + "+" + ("-" * cell_w + "+") * self.n
        print(border)

        # Board rows
        for i in range(self.m):
            row_str = f"{i:>{row_label_w}}|"
            for j in range(self.n):
                content_char = ""
                if self.game.black[i, j] == 1:
                    content_char = "O"  # Player 1 (Black)
                elif self.game.white[i, j] == 1:
                    content_char = "X"  # Player 2 (White)
                else:
                    content_char = str(i * self.n + j)

                # Highlight the last move if 'act' is provided and matches current cell
                if act is not None and i == act[0] and j == act[1]:
                    content = f"[{content_char}]"
                else:
                    content = content_char
                
                row_str += f"{content:^{cell_w}}|"
            print(row_str)
            print(border)

        if act is not None:
            print(f"Last move: {act}")

    def close(self):
        pass
