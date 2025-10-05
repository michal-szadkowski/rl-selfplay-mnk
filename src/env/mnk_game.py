from enum import Enum
import numpy as np

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

        return (
            self._count_in_line(x, y, 1, 0, board) >= self.k
            or self._count_in_line(x, y, 0, 1, board) >= self.k
            or self._count_in_line(x, y, 1, 1, board) >= self.k
            or self._count_in_line(x, y, 1, -1, board) >= self.k
        )

    def _count_in_line(self, x, y, dx, dy, board):
        return (
            self._count_direction(x, y, dx, dy, board)
            + self._count_direction(x, y, -dx, -dy, board)
            - 1
        )

    def _count_direction(self, x, y, dx, dy, board):
        sum = 0
        for i in range(self.k):
            x_cur = x + i * dx
            y_cur = y + i * dy

            if x_cur < 0 or x_cur >= self.m:
                break
            if y_cur < 0 or y_cur >= self.n:
                break

            if board[x_cur, y_cur] == 1:
                sum += 1
            else:
                break
        return sum
