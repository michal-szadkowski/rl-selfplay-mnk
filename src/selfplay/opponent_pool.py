from collections import deque
import random


class OpponentPool:
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.pool = deque(maxlen=max_size)

    def add_opponent(self, opponent):
        self.pool.append(opponent)

    def get_random_opponent(self):
        if not self.pool:
            return None
        return random.choice(self.pool)

    def size(self):
        return len(self.pool)
