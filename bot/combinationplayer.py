from bot.CNNPlayer import CNNetwork
import torch
import numpy as np
from copy import deepcopy
from bot.mmplayer import minimax

class CombPlayer():

    def __init__(self,depth,size,name,model):
        self.depth = depth
        self.size = size
        self.to_train = False
        self.name = "Comb "+name
        self.model = self.loadmodel(model)
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


    def move(self, game,enemy_move):
        xy = minimax(game,deepcopy(self.depth),self.heuristic)
        game.move(xy)

        return xy
    def heuristic(self,game,move):
        nparray = np.array([
            [(game.state == self.side).astype(int),
             (game.state == self.wait).astype(int),
             (game.state == self.EMPTY).astype(int)]
        ])
        x = torch.tensor(nparray, dtype=torch.float32)
        q_values = self.model.forward(x)
        value = np.argmax(q_values.detach().numpy())
        return value
    def loadmodel(self, load):
        model = CNNetwork(size=self.size)
        model.load_state_dict(torch.load(load))
        model.eval()
        return model
