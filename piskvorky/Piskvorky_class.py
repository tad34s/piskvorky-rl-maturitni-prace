
import numpy as np
import time
from variables import EMPTY, X, O
import piskvorky.functions as fns


class Piskvorky():

    def __init__(self, size: int):
        self.size = size
        self.EMPTY = EMPTY
        self.X = X
        self.O = O
        self.turn = self.X
        self.waiting = self.O
        self.state = np.zeros((self.size, self.size), dtype=np.int8)

    def __str__(self) -> str:

        """
        :return: Board as a string
        """
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

    def reset(self) -> None:
        """
        Resets the game
        :return:
        """
        self.state = np.zeros((self.size, self.size), dtype=np.int8)
        self.turn = self.X
        self.waiting = self.O

    def switch_turn(self) -> None:
        """
        Changes whose turn it is
        :return:
        """
        self.turn, self.waiting = self.waiting, self.turn

    def index_to_xy(self, index: int) -> tuple:
        """
        Takes index of the space when the board was flattened into a single array, returns indices for a 2d array
        :param index:
        :return: xy position
        """
        x = index % self.size
        y = index // self.size
        return x, y

    def xy_to_index(self, xy: tuple) -> int:
        """
        takes indices return single index, same as index_to_xy but reverse
        :param xy:
        :return: index
        """
        x, y = xy
        index = self.size * y + x
        return index

    def move(self, xy: tuple) -> None:
        """
        takes in tuple of indices of the move, checks if move isn`t illegal, than makes the move
        :param xy:
        :return:
        """
        x, y = xy
        if not self.is_legal(xy):
            raise Exception("Illegal move")

        self.state[y, x] = self.turn

        self.switch_turn()

    def is_legal(self, xy: tuple, state=False) -> bool:
        """
        Checks if move isn't illegal
        :param xy:
        :param state:
        :return:
        """
        x, y = xy
        if state is False:
            state = self.state

        if not self.is_in_bounds(xy):
            return False
        if state[y, x] != self.EMPTY:
            return False

        return True

    def is_in_bounds(self, xy: tuple) -> bool:
        """
        Checks if move is in bounds of the board
        :param xy:
        :return:
        """

        x, y = xy

        if not 0 <= x < self.size:
            return False
        if not 0 <= y < self.size:
            return False
        return True

    def insert_empty(self, xy: tuple) -> None:
        """
        Deletes symbol on the space, and switches turn, effectively reverses a move
        :param xy:
        :return:
        """
        x, y = xy
        self.state[y, x] = self.EMPTY
        self.switch_turn()

    def end(self, xy: tuple):
        """
        Checks if move ended the game, returns the side that won.
        Should be called after calling move(xy).

        :param xy:
        :return:side that won, "0" if draw or False
        """
        row = fns.row_points(self.state, xy)
        column = fns.column_points(self.state,xy)
        left_diag = fns.left_diag_points(self.state,xy)
        right_diag = fns.right_diag_points(self.state,xy)
        if right_diag >= 5 or left_diag >= 5 or row >= 5 or column >= 5:
            # game is won
            return self.waiting

        elif np.count_nonzero(self.state == self.EMPTY) == 0:
            # game is drawn
            return "0"
        else:
            # game did not end
            return False
