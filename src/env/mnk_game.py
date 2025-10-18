from enum import Enum
import numpy as np
from numba import jit

Color = Enum("Color", [("Black", 0), ("White", 1)])


class MnkGame:
    def __init__(self, m: int, n: int, k: int):
        self.m = m
        self.n = n
        self.k = k
        self.white = np.zeros(shape=(m, n), dtype=np.bool)
        self.black = np.zeros(shape=(m, n), dtype=np.bool)
        self.turn = 0
        self.finished = False

    def board(self, color: Color):
        if color == Color.Black:
            return np.stack([self.black, self.white])
        else:
            return np.stack([self.white, self.black])

    def put(self, x: int, y: int):
        if self.finished:
            raise Exception("Illegal move, game is finished")
        if self.white[x, y] == 1 or self.black[x, y] == 1:
            raise Exception("Illegal move, space taken")
        if self.turn % 2 == 0:
            self.black[x, y] = 1
        else:
            self.white[x, y] = 1
        if self._check_for_win(x, y):
            self.finished = True
            return True
        self.turn += 1
        return self.turn == self.m * self.n

    def get_winner(self):
        if not self.finished:
            return None
        if self.turn == self.m * self.n:
            return None
        return Color.Black if self.turn % 2 == 0 else Color.White

    def _check_for_win(self, x, y):
        if self.turn % 2 == 0:
            board = self.black
        else:
            board = self.white

        return _check_win_numba(board, x, y, self.k, self.m, self.n)


@jit(nopython=True)
def _check_win_numba(board, x, y, k, m, n):
    """
    Numba-compiled win checking function.
    """
    count = 1  # Current piece

    # Check horizontal (dx=1, dy=0)
    # Positive direction
    for i in range(1, k):
        x_cur = x + i
        if x_cur >= m or board[x_cur, y] == 0:
            break
        count += 1
    # Negative direction
    for i in range(1, k):
        x_cur = x - i
        if x_cur < 0 or board[x_cur, y] == 0:
            break
        count += 1
    if count >= k:
        return True

    # Reset count for vertical
    count = 1
    # Check vertical (dx=0, dy=1)
    # Positive direction
    for i in range(1, k):
        y_cur = y + i
        if y_cur >= n or board[x, y_cur] == 0:
            break
        count += 1
    # Negative direction
    for i in range(1, k):
        y_cur = y - i
        if y_cur < 0 or board[x, y_cur] == 0:
            break
        count += 1
    if count >= k:
        return True

    # Reset count for diagonal
    count = 1
    # Check diagonal (dx=1, dy=1)
    # Positive direction
    for i in range(1, k):
        x_cur = x + i
        y_cur = y + i
        if x_cur >= m or y_cur >= n or board[x_cur, y_cur] == 0:
            break
        count += 1
    # Negative direction
    for i in range(1, k):
        x_cur = x - i
        y_cur = y - i
        if x_cur < 0 or y_cur < 0 or board[x_cur, y_cur] == 0:
            break
        count += 1
    if count >= k:
        return True

    # Reset count for anti-diagonal
    count = 1
    # Check anti-diagonal (dx=1, dy=-1)
    # Positive direction
    for i in range(1, k):
        x_cur = x + i
        y_cur = y - i
        if x_cur >= m or y_cur < 0 or board[x_cur, y_cur] == 0:
            break
        count += 1
    # Negative direction
    for i in range(1, k):
        x_cur = x - i
        y_cur = y + i
        if x_cur < 0 or y_cur >= n or board[x_cur, y_cur] == 0:
            break
        count += 1
    if count >= k:
        return True

    return False
