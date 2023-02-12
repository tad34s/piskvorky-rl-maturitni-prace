from CNNPlayer import CNNetwork
import torch
import numpy as np
from copy import deepcopy


class CombPlayer():

    def __init__(self,depth,size,name,model,load = None):
        self.depth = depth
        self.size = size
        self.to_train = False
        self.name = "Comb "+name
        self.model = model
        if load:
            self.model = self.loadmodel(load)
        self.EMPTY = 0

    def newgame(self,side,other):
        self.side = side
        self.wait = other

    def encode(self, state):
        nparray = np.array([
            [(state == self.side).astype(int),
             (state == self.wait).astype(int),
             (state == self.EMPTY).astype(int)]
        ])
        output = torch.tensor(nparray, dtype=torch.float32)

        return output

    def minimax(self, board, depth, alpha, beta, maximizing_player, mov):

        if depth == 0:
            encodeboard = self.encode(board.state)
            probs, q_values = self.model.probs(encodeboard)
            value = np.amax(q_values.numpy())
            if maximizing_player:
                value = value * -1
            return value, None

        # Base case: if the depth is 0 or the board is a terminal state, return the value of the board
        if mov is not None:
            vysledek = board.end(mov)
            if depth == 0 or vysledek:
                value = 1
                if maximizing_player:
                    value = value * -1
                if vysledek == "0":
                    value = 0
                return value, None

        # Initialize the best value to a large negative number for the maximizing player and a large positive number for the minimizing player
        if maximizing_player:
            best_value = float("-inf")
            best_move = None
        else:
            best_value = float("inf")
            best_move = None

        # Generate a list of all possible next boards

        # Recursively call minimax on each of the next boards

        for move in board.listofpossiblemoves():

            board.move(move)
            value, _ = self.minimax(board, deepcopy(depth - 1), alpha, beta, not maximizing_player, move)
            board.insertempty(move)
            if maximizing_player:
                # Choose the maximum value
                if value > best_value:
                    best_value = value
                    best_move = move

                alpha = max(alpha, value)
            else:
                # Choose the minimum value
                if value < best_value:
                    best_value = value
                    best_move = move

                beta = min(beta, value)

            # Prune the search tree if possible
            if alpha >= beta:
                break

        # Add the board and its value to the cache
        return best_value, best_move

    def move(self, game,enemy_move):
        value, xy = self.minimax(game, deepcopy(self.depth), float("-inf"), float("inf"), True, mov=None)
        print(value,xy)
        game.move(xy)

        return xy

    def loadmodel(self, load):
        model = CNNetwork(size=self.size)
        model.load_state_dict(torch.load(load))
        model.eval()
        return model
