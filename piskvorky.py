import numpy as np
import random
from variables import EMPTY, X, O


class Piskvorky():

    def __init__(self, size):
        self.size = size
        self.EMPTY = EMPTY
        self.X = X
        self.O = O
        self.turn = self.X
        self.wait = self.O
        self.state = np.zeros((self.size, self.size), dtype=np.int8)

    def __str__(self):  # return the board as string
        string = ""
        for i in self.state:
            for j in i:
                if j == self.X:
                    string += "X"
                elif j == self.O:
                    string += "O"
                else:
                    string += "_"
            string += "\n"
        return string

    def reset(self):  # resets board
        self.state = np.zeros((self.size, self.size), dtype=np.int8)
        self.turn = self.X
        self.wait = self.O

    def switch_turn(self):
        self.turn, self.wait = self.wait, self.turn

    def index_to_xy(self, index):
        x = index % self.size
        y = index // self.size
        return x, y

    def xy_to_index(self, xy):
        x, y = xy
        index = self.size * y + x
        return index

    def move(self, xy):
        x, y = xy
        if not self.is_legal(xy):
            raise Exception("Illegal move")

        self.state[y, x] = self.turn

        self.switch_turn()

    def is_legal(self, xy):
        x, y = xy

        if not self.is_in_bounds(xy):
            return False
        if self.state[y, x] != self.EMPTY:
            return False

        return True
    def is_in_bounds(self,xy):
        x, y = xy

        if not 0 <= x < self.size:
            return False
        if not 0 <= y < self.size:
            return False
        return True
    def is_near(self,xy):
        def empty_or_out_of_bounds(y,x):
            xy = (x,y)
            if not self.is_in_bounds(xy):
                return True
            if self.state[y,x] == self.EMPTY:
                return True
            return False


        x,y = xy

        if not (
            empty_or_out_of_bounds(y+1,x) and
            empty_or_out_of_bounds(y+1, x+1) and
            empty_or_out_of_bounds(y+1, x-1) and
            empty_or_out_of_bounds(y-1, x) and
            empty_or_out_of_bounds(y-1, x+1) and
            empty_or_out_of_bounds(y-1, x-1) and
            empty_or_out_of_bounds(y, x+1) and
            empty_or_out_of_bounds(y, x-1)
        ):
            return True

        return False
    def insert_empty(self, xy):
        x, y = xy
        self.state[y, x] = self.EMPTY
        self.switch_turn()

    def left_diag_points(self, x, y):

        points = 1

        for i in range(1, 6):
            if y + i > self.size - 1 or x + i > self.size - 1:
                break
            if self.state[y + i, x + i] != self.state[y, x]:
                break
            points += 1

        for i in range(1, 6):
            if y - i < 0 or x - i < 0:
                break
            if self.state[y - i, x - i] != self.state[y, x]:
                break
            points += 1

        return points

    def right_diag_points(self, x, y):

        points = 1

        for i in range(1, 6):
            if y + i > self.size - 1 or x - i < 0:
                break
            if self.state[y + i, x - i] != self.state[y, x]:
                break
            points += 1

        for i in range(1, 6):
            if y - i < 0 or x + i > self.size - 1:
                break
            if self.state[y - i, x + i] != self.state[y, x]:
                break
            points += 1

        return points

    def row_points(self, x, y):

        points = 1

        for i in range(1, 6):

            if y + i > self.size - 1:
                break
            if self.state[y + i, x] != self.state[y, x]:
                break
            points += 1

        for i in range(1, 6):
            if y - i < 0:
                break
            if self.state[y - i, x] != self.state[y, x]:
                break
            points += 1

        return points

    def column_points(self, x, y):

        points = 1

        for i in range(1, 6):
            if x + i > self.size - 1:
                break
            if self.state[y, x + i] != self.state[y, x]:
                break
            points += 1

        for i in range(1, 6):
            if x - i < 0:
                break
            if self.state[y, x - i] != self.state[y, x]:
                break
            points += 1
        return points

    def hash(self):
        res = 0
        for y, row in enumerate(self.state):
            for x, space in enumerate(row):
                res *= self.size
                res += space

        return res

    def end(self, xy):
        x, y = xy
        row = self.row_points(x, y)
        column = self.column_points(x, y)
        left_diag = self.left_diag_points(x, y)
        right_diag = self.right_diag_points(x, y)
        if right_diag >= 5 or left_diag >= 5 or row >= 5 or column >= 5:
            return self.wait

        elif np.count_nonzero(self.state == self.EMPTY) == 0:

            return "0"
        else:
            return False


def list_of_possible_moves(game):
    move_list = []

    for y, i in enumerate(game.state):
        for x, j in enumerate(i):
            if j == game.EMPTY:
                move_list.append((x, y))

    return move_list
