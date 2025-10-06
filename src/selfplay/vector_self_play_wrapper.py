from typing import Any

import numpy as np
import torch
import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
from pettingzoo import AECEnv

from .self_play_wrapper import Policy


class BatchRandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        actions = []
        for o in obs:
            act = self.action_space.sample(mask=o['action_mask'].astype(np.int8))
            actions.append(act)
        return actions


class VectorNNPolicy(Policy):
    def __init__(self, model, device="cpu"):
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.device = device

    def act(self, obs):
        observation_batch = torch.stack([
            torch.as_tensor(o["observation"], device=self.device, dtype=torch.float32) 
            for o in obs
        ])
        mask_batch = torch.stack([
            torch.as_tensor(o["action_mask"], device=self.device, dtype=torch.bool) 
            for o in obs
        ])

        with torch.no_grad():
            dist, _ = self.model(observation_batch, mask_batch)

        action = dist.sample()
        return action.cpu().numpy()


class VectorSelfPlayWrapper(gym.vector.VectorEnv):
    def __init__(self, env_fn, n_envs=1):
        self.metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

        self.num_envs = n_envs

        self.envs: list[AECEnv] = [env_fn() for _ in range(self.num_envs)]

        self.possible_agents = self.envs[0].possible_agents

        self.single_observation_space = self.envs[0].observation_space(self.possible_agents[0])
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = self.envs[0].action_space(self.possible_agents[0])
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.players = [None for _ in range(self.num_envs)]

        self.autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        self.opponent = None

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        for i in range(self.num_envs):
            self.envs[i].reset(seed=seed)
            self.players[i] = np.random.choice(self.possible_agents)

        self.opponent_step()

        self.env_obs = [None for _ in range(self.num_envs)]
        self.rewards = np.zeros((self.num_envs,), dtype=np.float32)
        self.terminations = np.zeros((self.num_envs,), dtype=np.bool_)

        self.truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self.autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        infos = {}
        for i in range(self.num_envs):
            self.env_obs[i], _, self.terminations[i], self.truncations[i], info = self.envs[i].last()
            infos = self._add_info(infos, info, i)

        observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )

        observations = concatenate(
            self.single_observation_space, self.env_obs, observations
        )

        return observations, infos

    def opponent_step(self):
        envs_to_step = [v for i, v in enumerate(self.envs) if self.players[i] == v.agent_selection and not self.autoreset_envs[i]]

        obs = [env.last()[0] for env in envs_to_step]

        act = self.opponent.act(obs)

        for i, env in enumerate(envs_to_step):
            _,_,term,trunc,info = env.last()
            if term or trunc:
                env.step(None)
            else:
                env.step(act[i])

    def step(self, actions):

        for i, env in enumerate(self.envs):
            if self.autoreset_envs[i]:
                env.reset()
                self.players[i] = np.random.choice(self.possible_agents)
            else:
                env.step(actions[i])

        self.opponent_step()

        infos = {}
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.last()

            self.env_obs[i] = o
            self.rewards[i] = r
            self.terminations[i] = term
            self.truncations[i] = trunc
            infos = self._add_info(infos, info, i)

        self.autoreset_envs = np.logical_or(self.terminations, self.truncations)

        observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )

        observations = concatenate(
            self.single_observation_space, self.env_obs, observations
        )

        return observations, self.rewards, self.terminations, self.truncations, infos
