import numpy as np
from variables import EMPTY, GAME_SIZE
import torch
from piskvorky import row_points,column_points,left_diag_points,right_diag_points


def reward_if_terminal(state:np.ndarray, xy:tuple)->int or None:
    """
    Reward function for MCTS
    :param state:
    :param xy:
    :return: reward
    """
    x, y = xy
    row = row_points(state,xy)
    column = column_points(state,xy)
    left_diag = left_diag_points(state,xy)
    right_diag = right_diag_points(state,xy)
    if right_diag >= 5 or left_diag >= 5 or row >= 5 or column >= 5:
        return -1

    elif np.count_nonzero(state == EMPTY) == 0:
        return 0
    else:
        return None

def encode(state:np.ndarray,side:int,other:int)->torch.Tensor:
    """
    Our encode function for Alpha Zero
    :param state:
    :param side:
    :param other:
    :return:
    """
    nparray = np.array([
        [(state == side).astype(int),
         (state == other).astype(int)]
    ])
    output = torch.tensor(nparray, dtype=torch.float32)
    return output