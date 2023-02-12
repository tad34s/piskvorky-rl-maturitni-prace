import numpy as np
import random


class Piskvorky():

    def __init__(self, size):
        self.size = size
        self.EMPTY = 0
        self.X = 1
        self.O = 2
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

    def reset(self):  # resets boarc
        self.state = np.zeros((self.size, self.size), dtype=np.int8)
        self.turn = self.X
        self.wait = self.O

    def switchturn(self):
        self.turn, self.wait = self.wait, self.turn

    def indextoxy(self, index):
        x = index % self.size
        y = index // self.size
        return x, y

    def xytoindex(self, xy):
        x, y = xy
        index = self.size * y + x
        return index

    def move(self, xy):
        x, y = xy
        if not self.islegal(xy):
            raise Exception("Illegal move")

        self.state[y, x] = self.turn

        self.switchturn()

    def listofpossiblemoves(self):
        movelist = []

        for y, i in enumerate(self.state):
            for x, j in enumerate(i):
                if j == self.EMPTY:
                    movelist.append((x, y))
        return movelist

    def random_move(self):

        return random.choice(self.listofpossiblemoves())

    def islegal(self, xy):
        x, y = xy

        if not 0 <= x < self.size:
            print("Out of rows")
            return False
        if not 0 <= y < self.size:
            print("Out of columns")
            return False
        if self.state[y, x] != self.EMPTY:
            return False

        return True

    def insertempty(self, xy):
        x, y = xy
        self.state[y, x] = self.EMPTY
        self.switchturn()

    def leftdiagpoints(self, x, y):

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

    def rightdiagpoints(self, x, y):

        points = 1

        for i in range(1, 6):
            if y + i > self.size - 1 or x - i < 0:
                break
            if self.state[y + i, x - 1] != self.state[y, x]:
                break
            points += 1

        for i in range(1, 6):
            if y - i < 0 or x + i > self.size - 1:
                break
            if self.state[y - i, x + i] != self.state[y, x]:
                break
            points += 1

        return points

    def rowpoints(self, x, y):

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

    def columnpoints(self, x, y):

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
        """
        Encode the current state of the game (board positions) as an integer. Will be used for caching evaluations
        : A collision free hash value representing the current board state
        """
        res = 0
        for y, row in enumerate(self.state):
            for x, space in enumerate(row):
                res *= self.size
                res += space

        return res

    def end(self, xy):
        x, y = xy
        rada = self.rowpoints(x, y)
        sloupec = self.columnpoints(x, y)
        levadiagonala = self.leftdiagpoints(x, y)
        pravadiagonala = self.rightdiagpoints(x, y)
        if pravadiagonala >= 5 or levadiagonala >= 5 or rada >= 5 or sloupec >= 5:
            return self.wait

        elif np.count_nonzero(self.state == self.EMPTY) == 0:

            return "0"
        else:
            return False
