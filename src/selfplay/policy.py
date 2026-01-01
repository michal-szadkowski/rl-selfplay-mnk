from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn


class Policy(ABC):
    @abstractmethod
    def act(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        pass


class RandomPolicy(Policy):
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def act(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        mask = obs["action_mask"]
        probs = mask.float()

        row_sums = probs.sum(dim=1, keepdim=True)
        zero_rows = row_sums == 0
        if zero_rows.any():
            probs = probs + zero_rows.float() * 1e-8

        if deterministic:
            return torch.argmax(probs, dim=1)
        else:
            return torch.multinomial(probs, num_samples=1).squeeze(1)


class NNPolicy(Policy):
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def act(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        observation = obs["observation"]
        action_mask = obs["action_mask"]

        if observation.dim() == 3:
            observation = observation.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        with torch.no_grad():
            dist, _ = self.model(observation, action_mask)

            if deterministic:
                action = torch.argmax(dist.logits, dim=1)
            else:
                action = dist.sample()

        return action
