from bot.Players.DQN.Networks import CNNetwork_preset
from bot.Player_abstract_class import Player
from piskvorky import mask_invalid_moves
import torch
import numpy as np
from copy import deepcopy
from bot.Players.Minimax_player import minimax


class CombPlayer(Player):
    """
    Player that combines the Minimax algorithm with DQN
    """
    def __init__(self, depth:int, size:int, name:str, model, restrict_movement=False):
        """
        :param depth: maximal depth
        :param size: game size
        :param name:
        :param model: model used to calculate state value
        :param restrict_movement: search spaces near symbol
        """
        self.depth = depth
        self.size = size
        self.to_train = False
        self.name = "Comb " + name
        self.model = model
        self.restrict_movement = restrict_movement
        self.move_count = 0

    def new_game(self, side, other):
        self.side = side
        self.wait = other
        self.move_count = 0

    def encode(self, state:np.ndarray)->torch.Tensor:
        """
        Encode state for the model
        :param state:
        :return:
        """
        nparray = np.array([
            [(state == self.side).astype(int),
             (state == self.wait).astype(int)]
        ])
        output = torch.tensor(nparray, dtype=torch.float32)
        return output

    def move(self, game, enemy_move):
        xy = minimax(game, deepcopy(self.depth), self.heuristic,self.restrict_movement)
        game.move(xy)
        self.move_count += 1
        return xy

    def heuristic(self, game, move)->float:
        """
        The heuristic is now replaced with model prediction
        :param game:
        :param move:
        :return:
        """
        q_values, probs = self.model.probs(self.encode(game.state))
        mask = mask_invalid_moves(game.state, self.restrict_movement)
        probs *= mask
        index = np.argmax(probs) # index of the best move
        return q_values[index].item() # value of the best move

