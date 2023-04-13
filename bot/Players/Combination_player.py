from bot.Players.DQN.Networks import CNNetwork_preset
from bot.Player_abstract_class import Player
from piskvorky import mask_invalid_moves
import torch
import numpy as np
from copy import deepcopy
from bot.Players.Minimax_player import minimax


class CombPlayer(Player):

    def __init__(self, depth, size, name, model, restrict_movement=False):
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

    def encode(self, state):
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

    def heuristic(self, game, move):
        q_values, probs = self.model.probs(self.encode(game.state))
        mask = mask_invalid_moves(game.state, self.restrict_movement)
        probs *= mask
        index = np.argmax(probs)
        return q_values[index].item()

    def loadmodel(self, load):
        model = CNNetwork_preset(size=self.size)
        model.load_state_dict(torch.load(load))
        model.eval()
        return model
