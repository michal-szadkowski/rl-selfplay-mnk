"""
Self-play module for training agents through self-competition.

Contains opponent pooling, policy implementations, and training utilities.
"""

from .policy import Policy, NNPolicy, RandomPolicy
from .opponent_pool import OpponentPool

__all__ = ["Policy", "NNPolicy", "RandomPolicy", "OpponentPool"]