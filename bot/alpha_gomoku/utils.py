import numpy as np
from variables import EMPTY, VELIKOST
import torch
from piskvorky import row_points,column_points,left_diag_points,right_diag_points


def reward_if_terminal(state, xy):
    x, y = xy
    row = row_points(state,x, y)
    column = column_points(state,x, y)
    left_diag = left_diag_points(state,x, y)
    right_diag = right_diag_points(state,x, y)
    if right_diag >= 5 or left_diag >= 5 or row >= 5 or column >= 5:
        return -1

    elif np.count_nonzero(state == EMPTY) == 0:
        return 0
    else:
        return None

def encode(state,side,other):

    nparray = np.array([
        [(state == side).astype(int),
         (state == other).astype(int)]
    ])
    output = torch.tensor(nparray, dtype=torch.float32)
    return output