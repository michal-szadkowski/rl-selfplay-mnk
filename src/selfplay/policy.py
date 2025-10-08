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
