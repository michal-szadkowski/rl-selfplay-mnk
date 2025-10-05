import gymnasium as gym
from pettingzoo.utils.env import AECEnv
import numpy as np


class RandomPlayWrapper(gym.Env):
    def __init__(self, env: AECEnv):
        self.env = env
        self.observation_space = self.env.observation_space(self.env.possible_agents[0])
        self.action_space = self.env.action_space(self.env.possible_agents[0])
        self.possible_agents = self.env.possible_agents

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        obs, _, _, _, _ = self.env.last()
        return obs, {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info.

        The observation is for the next agent (used to determine the next action), while the remaining
        items are for the agent that just acted (used to understand what just happened).
        """
        current_agent = self.agent_selection

        self.env.step(action)
        next_agent = self.agent_selection

        obs, reward, termination, truncation, info = self.env.last()

        if not (termination or truncation):
            action = (self.env.action_space(next_agent).sample(mask=obs['action_mask'].astype(np.int8)))
        else:
            action = None

        self.env.step(action)

        return (
            self.env.observe(current_agent),
            self.env._cumulative_rewards[current_agent],
            self.env.terminations[current_agent],
            self.env.truncations[current_agent],
            self.env.infos[current_agent],
        )

    @property
    def agent_selection(self):
        return self.env.agent_selection

    def render(self):
        return self.env.render()
