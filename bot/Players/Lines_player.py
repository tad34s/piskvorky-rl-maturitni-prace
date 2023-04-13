import random
from copy import copy
from bot.Player_abstract_class import Player
from piskvorky import Piskvorky


class LinesPlayer(Player):
    """
    Player that just picks a lines and tries to fill it, if its blocked picks the next best line
    """
    def __init__(self, game_size:int, name:str):
        super().__init__(name="Lines Player" + name)
        self.game_size = game_size
        self.line_list = self.generate_line_list()
        self.to_train = False
        self.game_length = 0

    def new_game(self, side:int, other:int):
        self.line_list = self.generate_line_list()
        self.game_length = 0

    def reward(self, game, enemy_move: tuple):
        """
        Function that replaces the reward function in CNNPlayer when it is trying to train blocking.
        Calculates the reward, by the number of symbols it blocked
        :param game:
        :param enemy_move:
        :return:
        """

        points = []
        for index, line in enumerate(self.line_list):
            if enemy_move in line[0]:
                points.append(copy(line[1]))
                self.line_list[index][1] = 0 # the line is now blocked so we set the number of symbols in line to zero

        points.sort(reverse=True)
        reward_points = sum([(x ** 2) / (1 * (i + 1))
                             for i, x in enumerate(points)])
        reward_points += self.game_length / 80
        return reward_points

    def move(self, game: Piskvorky, enemy_move:tuple):
        """
        Goes through line_list and finds the line with the most symbols filled, then pics a random spot on that line.
        :param game:
        :param enemy_move:
        :return:
        """
        moves = []
        max_points = -1
        for line, points in self.line_list:
            if points > max_points:
                # found a better line
                new_moves = []
                for x in line:
                    # add all the legal moves
                    if game.is_legal(x):
                        new_moves.append(x)
                if new_moves:
                    # replace the old moves, and points with new ones
                    moves = new_moves
                    max_points = points

            elif points == max_points:
                # found as good
                for move in line:
                    # add legal moves to the existing list
                    if not move in moves and game.is_legal(move):
                        moves.append(move)

        # return a random move form the picked moves
        output = random.choice(moves)

        # add a point to the lines in line_list
        for index, line in enumerate(self.line_list):
            if output in line[0]:
                self.line_list[index][1] += 1

        self.game_length += 1
        game.move(output)
        return output

    def generate_line_list(self):
        """
        Generates a list of all the possible lines. Each lines has two elements, a list of the spaces in the line and
        a number of symbols put in the line.

        :return:
        """
        line_list = []
        for row in range(self.game_size):
            for column in range(self.game_size):

                if column >= 4:
                    line = [(column - x, row) for x in range(5)]
                    line_list.append([line, 0])

                if row >= 4:
                    line = [(column, row - x) for x in range(5)]
                    line_list.append([line, 0])

                if row >= 4 and column >= 4:
                    line = [(column - x, row - x) for x in range(5)]
                    line_list.append([line, 0])
                    line = [(column - 4 + x, row - x) for x in range(5)]
                    line_list.append([line, 0])

        return line_list
