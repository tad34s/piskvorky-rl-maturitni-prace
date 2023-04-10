import random
from copy import copy
from bot.Players.Player_abstract_class import Player


class LinesPlayer(Player):
    # player that just picks a lines and tries to fill it, if its blocked picks the next best line
    def __init__(self, game_size, name):
        super().__init__(name="Lines Player" + name)
        self.game_size = game_size
        self.line_list = self.generate_line_list()
        self.to_train = False
        self.game_length = 0

    def new_game(self, side, other):
        self.line_list = self.generate_line_list()
        self.game_length = 0

    def reward(self, game, enemy_move: tuple):
        reward_points = 0
        points = []
        for index, line in enumerate(self.line_list):
            if enemy_move in line[0]:
                points.append(copy(line[1]))
                self.line_list[index][1] = 0
        points.sort(reverse=True)
        reward_points = sum([(x ** 2) / (30 * (i + 1))
                             for i, x in enumerate(points)])
        reward_points += self.game_length / 80
        return reward_points

    def move(self, game, enemy_move: tuple):
        moves = []
        max_points = -1
        for line, points in self.line_list:
            if points > max_points:
                new_moves = []
                for x in line:
                    if game.is_legal(x):
                        new_moves.append(x)
                if new_moves:
                    moves = new_moves
                    max_points = points
            elif points == max_points:
                for move in line:
                    if not move in moves and game.is_legal(move):
                        moves.append(move)

        output = random.choice(moves)
        for index, line in enumerate(self.line_list):
            if output in line[0]:
                self.line_list[index][1] += 1
        self.game_length += 1
        game.move(output)
        return output

    def generate_line_list(self):
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
