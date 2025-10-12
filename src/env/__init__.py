"""
Environment module for MNK games.

Contains PettingZoo-compatible environments for M-N-K games.
"""

from .mnk_game_env import create_mnk_env

__all__ = ["create_mnk_env"]