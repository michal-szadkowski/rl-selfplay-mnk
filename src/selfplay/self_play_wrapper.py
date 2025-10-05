import gymnasium as gym
from pettingzoo.utils.env import AECEnv
import numpy as np
from abc import ABC, abstractmethod

import torch


class Policy(ABC):
    @abstractmethod
    def act(self, obs):
        pass


class RandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        if isinstance(obs, dict):
            action_mask = obs["action_mask"].astype(np.int8)
        else:
            action_mask = None

        action = self.action_space.sample(mask=action_mask)
        return action


class NNPolicy(Policy):
    def __init__(self, model, device="cpu"):
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.device = device

    def act(self, obs):

        if isinstance(obs, dict):
            observation = obs["observation"]
            action_mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=self.device)
        else:
            observation = obs
            action_mask = None

        with torch.no_grad():
            dist, _ = self.model(
                torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0),
                action_mask,
            )
        action = dist.sample()
        return action.item()


class SelfPlayWrapper(gym.Env):
    def __init__(self, env: AECEnv):
        self.env = env
        self.observation_space = self.env.observation_space(self.env.possible_agents[0])
        self.action_space = self.env.action_space(self.env.possible_agents[0])
        self.possible_agents = self.env.possible_agents

        self.opponent_pool: list[Policy] = [RandomPolicy(self.action_space)]
        self.current_opponent: Policy = None

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)

        self.current_opponent = np.random.choice(self.opponent_pool)

        self.player = np.random.choice(self.env.possible_agents)

        for a in self.env.agent_iter():
            obs, _, _, _, _ = self.env.last()

            if a == self.player:
                return obs, {}

            action = self.current_opponent.act(obs)
            self.env.step(action)

        return obs, {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info.

        The observation is for the next agent (used to determine the next action), while the remaining
        items are for the agent that just acted (used to understand what just happened).
        """
        self.env.step(action)

        obs, reward, termination, truncation, info = self.env.last()
        if not (termination or truncation):
            action = self.current_opponent.act(obs)
        else:
            action = None
        self.env.step(action)

        return self.env.last()

    @property
    def agent_selection(self):
        return self.env.agent_selection

    def render(self):
        return self.env.render()
