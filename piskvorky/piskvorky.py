import numpy as np
import time
from variables import EMPTY, X, O
from piskvorky.functions import left_diag_points,right_diag_points,column_points,row_points


class Piskvorky():

    def __init__(self, size):
        self.size = size
        self.EMPTY = EMPTY
        self.X = X
        self.O = O
        self.turn = self.X
        self.waiting = self.O
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
        self.waiting = self.O

    def switch_turn(self):
        self.turn, self.waiting = self.waiting, self.turn

    def index_to_xy(self, index):
        x = index % self.size
        y = index // self.size
        return x, y

    def xy_to_index(self, xy):
        x, y = xy
        index = self.size * y + x
        return index

    def move(self, xy:tuple):
        x, y = xy
        if not self.is_legal(xy):
            raise Exception("Illegal move")

        self.state[y, x] = self.turn

        self.switch_turn()

    def is_legal(self, xy:tuple,state = False):
        x, y = xy
        if state is False:
            state = self.state

        if not self.is_in_bounds(xy):
            return False
        if state[y, x] != self.EMPTY:
            return False

        return True
    def is_in_bounds(self,xy:tuple):
        x, y = xy

        if not 0 <= x < self.size:
            return False
        if not 0 <= y < self.size:
            return False
        return True

    def insert_empty(self, xy):
        x, y = xy
        self.state[y, x] = self.EMPTY
        self.switch_turn()

    def left_diag_points(self, x, y):

        return left_diag_points(self.state, x,y)

    def right_diag_points(self, x, y):


        return right_diag_points(self.state,x,y)

    def row_points(self, x, y):

        return row_points(self.state,x,y)

    def column_points(self, x, y):

        return column_points(self.state,x,y)

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
            return self.waiting

        elif np.count_nonzero(self.state == self.EMPTY) == 0:

            return "0"
        else:
            return False


