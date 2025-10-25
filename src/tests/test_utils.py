import numpy as np
from typing import Dict, Any


class FirstLegalActionPolicy:
    """Simple opponent policy that always picks the first legal action."""

    def __init__(self, name: str = "first_legal"):
        self.name = name
        self.action_count = 0

    def act(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Choose first legal action for each environment.

        Args:
            observations: Dict with 'observation' and 'action_mask'

        Returns:
            Array of actions (first legal action for each env)
        """
        action_masks = observations["action_mask"]
        batch_size = action_masks.shape[0]
        actions = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            legal_actions = np.where(action_masks[i] == 1)[0]
            if len(legal_actions) > 0:
                actions[i] = legal_actions[0]
            else:
                # Should not happen in valid game state
                actions[i] = 0

        self.action_count += batch_size
        return actions

    def reset(self) -> None:
        """Reset action counter."""
        self.action_count = 0
