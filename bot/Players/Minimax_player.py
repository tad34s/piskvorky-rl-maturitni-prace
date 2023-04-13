import random
from bot.Player_abstract_class import Player
from piskvorky import Piskvorky, list_of_possible_moves
from variables import EMPTY


class MinimaxPlayer(Player):
    """
    Player that uses the minimax algorithm to find the best move
    """

    def __init__(self, depth: int, name: str, restrict_movement: bool = False):
        """

        :param depth: maximal depth the minimax algorithm will search too
        :param name:
        :param restrict_movement: search only spaces near symbols
        """

        super().__init__(name)
        self.depth = depth
        self.name = "Minim " + name
        self.to_train = False
        self.restrict_movement = restrict_movement

    def move(self, game, enemy_move):
        xy = minimax(game, self.depth, self.heuristic, self.restrict_movement)
        game.move(xy)
        return xy

    def heuristic(self, game: Piskvorky, move: tuple) -> float:
        """
        Heuristic to determine the value of a state, when minimax hits maximal depth
        :param game:
        :param move:
        :return:
        """
        row = game.row_points(move[0], move[1])
        column = game.column_points(move[0], move[1])
        leftdiag = game.left_diag_points(move[0], move[1])
        rightdiag = game.right_diag_points(move[0], move[1])

        value = max((row, column, leftdiag, rightdiag))
        value = value / 6

        return value


def minimax(game: Piskvorky, depth: int, heuristic, restrict_movement: bool):
    """
    Minimax algorith, uses alpha-beta pruning
    :param game:
    :param depth: maximal depth
    :param heuristic: function for value of a state when maximal depth was hit
    :param restrict_movement:  search only spaces near symbols
    :return:
    """

    def maxx(alpha: float, beta: float, depth: int, maxdepth: int):
        """
        The maximazing decision maker
        :param alpha:
        :param beta:
        :param depth: current depth
        :param maxdepth:
        :return:
        """
        maxv = -2000
        maxx = None
        maxy = None
        list = list_of_possible_moves(game.state, restrict_movement)
        for mov in list:

            game.move(mov)

            depth += 1

            if depth > maxdepth:
                # we hit max depth
                value = heuristic(game, mov)
                game.insert_empty(mov)
                depth -= 1
                return value, mov[0], mov[1]

            if game.end(mov) != "0" and game.end(mov):
                # we found a winning move
                game.insert_empty(mov)
                depth -= 1
                return 10, mov[0], mov[1]
            if not EMPTY in game.state:
                # board is full, game is drawn
                game.insert_empty(mov)
                depth -= 1
                return 0, mov[0], mov[1]

            # going deeper
            val, x, y = minn(alpha, beta, depth, maxdepth)

            game.insert_empty(mov)

            if val > maxv:
                # updating the optimal move
                maxv = val
                maxx = mov[0]
                maxy = mov[1]

            # alpha-beta pruning
            if maxv >= beta:
                depth -= 1
                return maxv, maxx, maxy

            if maxv > alpha:
                alpha = maxv
            depth -= 1
        return maxv, maxx, maxy

    def minn(alpha: float, beta: float, depth: int, maxdepth: int):
        """
        The minimizing decision maker
        :param alpha:
        :param beta:
        :param depth: current depth
        :param maxdepth:
        :return:
        """

        minv = 2000
        minx = None
        miny = None

        for mov in list_of_possible_moves(game.state, restrict_movement):
            game.move(mov)

            depth += 1

            if depth > maxdepth:
                # we hit max depth
                value = heuristic(game, mov)
                game.insert_empty(mov)
                depth -= 1
                return value * (-1), mov[0], mov[1]

            if game.end(mov) != "0" and game.end(mov):
                # we found a winning move
                game.insert_empty(mov)
                depth -= 1
                return -10, mov[0], mov[1]
            if not EMPTY in game.state:
                # game is drawn
                game.insert_empty(mov)
                depth -= 1
                return 0, mov[0], mov[1]

            # going deeper
            val, x, y = maxx(alpha, beta, depth, maxdepth)
            game.insert_empty(mov)

            # updating optimal move
            if val < minv:
                minv = val
                minx = mov[0]
                miny = mov[1]

            # alpha-beta pruning
            if minv <= alpha:
                depth -= 1

                return minv, minx, miny

            if minv < beta:
                beta = minv

            depth -= 1
        return minv, minx, miny

    # starting the minimax
    val, x, y = maxx(-200, 200, 0, depth)

    if val == 0:
        return random.choice(list_of_possible_moves(game.state, restrict_movement))

    return x, y
