import random
from copy import deepcopy
from typing import List, Optional, Callable
import torch

from .policy import VectorNNPolicy


class OpponentPool:
    def __init__(self, network_creator: Callable, max_size: int = 10, device: str = "cpu"):
        self.network_creator = network_creator
        self.max_size = max_size
        self.device = device
        self.opponents: List[dict] = []  # Store network states on CPU
    
    def add_opponent(self, network: torch.nn.Module) -> None:
        """Add a copy of the network to the opponent pool."""
        state_dict = deepcopy(network.state_dict())
        state_dict_cpu = {k: v.cpu() for k, v in state_dict.items()}
        self.opponents.append(state_dict_cpu)
        
        # Remove oldest opponent if pool exceeds max size
        if len(self.opponents) > self.max_size:
            self.opponents.pop(0)
    
    def _create_network_from_state(self, state_dict: dict) -> torch.nn.Module:
        """Create a network from the saved state dictionary."""
        network = self.network_creator()
        
        # Move state dict to target device if not CPU
        if self.device != "cpu":
            state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
        
        network.load_state_dict(state_dict)
        network.to(self.device)
        network.eval()
        return network
    
    def sample_opponent(self) -> Optional[VectorNNPolicy]:
        """Sample and create a single opponent from the pool."""
        if not self.opponents:
            return None
            
        selected_state = random.choice(self.opponents)
        network = self._create_network_from_state(selected_state)
        return VectorNNPolicy(network, device=self.device)
    
    def get_latest_opponent(self) -> Optional[VectorNNPolicy]:
        """Get the most recent opponent from the pool."""
        if not self.opponents:
            return None
            
        latest_state = self.opponents[-1]
        network = self._create_network_from_state(latest_state)
        return VectorNNPolicy(network, device=self.device)