from collections import deque
import random


class OpponentPool:
    """FIFO ring buffer for opponent policies with last N snapshots."""

    def __init__(self, max_size=5):
        self.max_size = max_size
        self.pool = deque(maxlen=max_size)

    def add_opponent(self, opponent):
        """Add opponent to pool (removes oldest if at capacity)."""
        self.pool.append(opponent)

    def get_random_opponent(self):
        """Get random opponent from pool."""
        if not self.pool:
            return None
        return random.choice(self.pool)

    def size(self):
        """Get current pool size."""
        return len(self.pool)
