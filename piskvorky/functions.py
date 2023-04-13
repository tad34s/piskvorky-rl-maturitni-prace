import numpy as np
from variables import EMPTY, GAME_SIZE
import bot.Player_abstract_class as plr

def play_game(game, player1: plr.Player, player2: plr.Player, visible=False) -> int:
    """
    Plays a game between 2 players, returns the symbol that won.
    visible turns on printing game state in the command line

    :param game:
    :param player1:
    :param player2:
    :param visible:
    :return: X, O or "0"
    """
    game.reset()
    player1.new_game(side=game.X, other=game.O)
    player2.new_game(side=game.O, other=game.X)
    turn = player1
    waiting = player2
    move = None
    while True:
        move = turn.move(game, move)
        if visible:
            print(str(game))
        if game.end(move):
            result = game.end(move)
            break
        turn, waiting = waiting, turn
    player1.game_end(result)
    player2.game_end(result)
    return result


def index_to_xy(size: int, index: int) -> tuple:
    """
    Takes index of the space when the board was flattened, returns indices for a 2d array
    :param size: size of the game board
    :param index: 1d array index
    :return: 2d array indices
    """

    x = index % size
    y = index // size
    return x, y


def xy_to_index(size: int, xy: tuple) -> int:
    """
    Takes indices return single index, same as index_to_xy but reverse
    :param size: size of the game board
    :param xy: 2d array indices
    :return: 1d array index
    """
    x, y = xy
    index = size * y + x
    return index


def mask_invalid_moves(state: np.ndarray, restrict_movements=False) -> np.ndarray:
    """
    Returns a numpy array, where legal moves are 1 and illegal 0.

    When restrict movements is True all empty spaces not near a symbol are also illegal.
    :param state: state of the board state of the board
    :param restrict_movements:
    :return: 1d numpy array
    """

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


def list_of_possible_moves(state, restrict_movement=False) -> list:
    """
    Takes in state of the board, returns indices of all legal moves.
    :param state: state of the board
    :param restrict_movement:
    :return: list of legal moves
    """

    mask = mask_invalid_moves(state, restrict_movement)
    indices = np.nonzero(mask)
    move_list = [index_to_xy(GAME_SIZE, x) for x in indices[0]]

    return move_list


def left_diag_points(state: np.ndarray, xy: tuple) -> int:
    """
    Counts the number of symbols connected diagonally to the left.
    :param xy: counting starting point
    :param state: state of the board
    :return: number of symbols connected
    """
    x, y = xy
    points = 1
    for i in range(1, 6):
        if y + i > GAME_SIZE - 1 or x + i > GAME_SIZE - 1:
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


def right_diag_points(state: np.ndarray, xy: tuple) -> int:
    """
    Counts the number of symbols connected diagonally to the right.
    :param xy: counting starting point
    :param state: state of the board
    :return: number of symbols connected
    """

    points = 1
    x, y = xy
    for i in range(1, 6):
        if y + i > GAME_SIZE - 1 or x - i < 0:
            break
        if state[y + i, x - i] != state[y, x]:
            break
        points += 1

    for i in range(1, 6):
        if y - i < 0 or x + i > GAME_SIZE - 1:
            break
        if state[y - i, x + i] != state[y, x]:
            break
        points += 1

    return points


def row_points(state: np.ndarray, xy: tuple) -> int:
    """
    Counts the number of symbols connected in the row.
    :param xy: counting starting point
    :param state: state of the board
    :return: number of symbols connected
    """
    points = 1
    x, y = xy
    for i in range(1, 6):

        if y + i > GAME_SIZE - 1:
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


def column_points(state: np.ndarray, xy: tuple) -> int:
    """
    Counts the number of symbols connected in the column.
    :param xy: counting starting point
    :param state: state of the board
    :return: number of symbols connected
    """
    points = 1
    x, y = xy
    for i in range(1, 6):
        if x + i > GAME_SIZE - 1:
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
