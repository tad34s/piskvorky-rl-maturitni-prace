import numpy as np
from variables import EMPTY, VELIKOST



def index_to_xy(size, index):
    x = index % size
    y = index // size
    return x, y


def play_game(game, player1, player2):
    game.reset()
    player1.new_game(side=game.X, other=game.O)
    player2.new_game(side=game.O, other=game.X)
    turn = player1
    waiting = player2
    move = None
    while True:
        move = turn.move(game, move)
        print(str(game))
        if game.end(move):
            result = game.end(move)
            break
        turn, waiting = waiting, turn
    player1.game_end(result)
    player2.game_end(result)
    return result


def xy_to_index(size: int, xy: tuple):
    x, y = xy
    index = size * y + x
    return index


def mask_invalid_moves(state: np.ndarray, restrict_movements=False):
    if restrict_movements:
        if not np.any(state):
            return np.ones(state.shape, dtype=np.float32).flatten()
        mask = np.zeros(state.shape, dtype=np.float32)
        for y, row in enumerate(state):
            for x, space in enumerate(row):
                if space != EMPTY:
                    if y < state.shape[0] - 1:
                        mask[y + 1, x] = 1
                        if x > 0:
                            mask[y + 1, x - 1] = 1
                        if x < state.shape[0] - 1:
                            mask[y + 1, x + 1] = 1
                    if y > 0:
                        mask[y - 1, x] = 1
                        if x > 0:
                            mask[y - 1, x - 1] = 1
                        if x < state.shape[0] - 1:
                            mask[y - 1, x + 1] = 1
                    if x > 0:
                        mask[y, x - 1] = 1
                    if x < state.shape[0] - 1:
                        mask[y, x + 1] = 1
        mask *= state == EMPTY
    else:
        mask = state == EMPTY

    return mask.flatten()


def list_of_possible_moves(state,restrict_movement = False):
    mask = mask_invalid_moves(state,restrict_movement)
    indices = np.nonzero(mask)
    move_list = [index_to_xy(VELIKOST,x) for x in indices[0]]

    return move_list


def left_diag_points(state, x, y):
    points = 1

    for i in range(1, 6):
        if y + i > VELIKOST - 1 or x + i > VELIKOST - 1:
            break
        if state[y + i, x + i] != state[y, x]:
            break
        points += 1

    for i in range(1, 6):
        if y - i < 0 or x - i < 0:
            break
        if state[y - i, x - i] != state[y, x]:
            break
        points += 1

    return points


def right_diag_points(state, x, y):
    points = 1

    for i in range(1, 6):
        if y + i > VELIKOST - 1 or x - i < 0:
            break
        if state[y + i, x - i] != state[y, x]:
            break
        points += 1

    for i in range(1, 6):
        if y - i < 0 or x + i > VELIKOST - 1:
            break
        if state[y - i, x + i] != state[y, x]:
            break
        points += 1

    return points


def row_points(state, x, y):
    points = 1

    for i in range(1, 6):

        if y + i > VELIKOST - 1:
            break
        if state[y + i, x] != state[y, x]:
            break
        points += 1

    for i in range(1, 6):
        if y - i < 0:
            break
        if state[y - i, x] != state[y, x]:
            break
        points += 1

    return points


def column_points(state, x, y):
    points = 1

    for i in range(1, 6):
        if x + i > VELIKOST - 1:
            break
        if state[y, x + i] != state[y, x]:
            break
        points += 1

    for i in range(1, 6):
        if x - i < 0:
            break
        if state[y, x - i] != state[y, x]:
            break
        points += 1
    return points
