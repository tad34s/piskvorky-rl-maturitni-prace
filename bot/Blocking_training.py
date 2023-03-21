from CNNPlayer import CNNPLayer
from variables import VELIKOST
import random
from piskvorky import Piskvorky




def main():
    piskvorky = Piskvorky(VELIKOST)
    n_cnn_players = 181
    cnn_players = []
    episodes = 100
    waiting = list(range(n_cnn_players))
    while True:
        #pick = random.sample(waiting, k=1)
        train_blocking(piskvorky, 181)


def train_blocking(game: Piskvorky,cnn_number: int):

    n_games = 100
    teacher = TeacherPlayer(game.size)
    player = CNNPLayer(VELIKOST, memory_size=50, name=str(cnn_number), load=False, to_train=True, block_training=teacher.reward)
    for i in range(n_games):
        print(i)
        game.reset()
        turn = teacher
        waiting = player
        teacher.new_game(game.X,game.O)
        player.new_game(game.O,game.X)
        move = None
        while True:
            move = turn.move(game, move)
            print(str(game))
            print(move)
            if game.end(move):
                vysledek = game.end(move)
                break
            turn, waiting = waiting, turn

        player.train(vysledek, epochs=20, n_recalls=0)
    player.save_model()

class TeacherPlayer():

    def __init__(self, game_size):
        self.game_size = game_size
        self.line_list = self.generate_line_list()
        self.to_train = False
        self.game_length = 0

    def new_game(self,side,other):
        self.line_list = self.generate_line_list()
        self.game_length = 0
        print(self.line_list)
    def reward(self, game: Piskvorky, enemy_move: tuple):
        reward_points = 0
        points = []
        for index, line in enumerate(self.line_list):
            if enemy_move in line[0]:
                points.append(line[1])
                self.line_list[index][1] = 0
        reward_points = sum([(x**3)/(25 *(i**2+1))
                             for i,x in enumerate(points)])
        reward_points += self.game_length/60
        print(reward_points)
        return reward_points

    def move(self, game: Piskvorky, enemy_move: tuple):
        moves = []
        max_points = -1
        for line, points in self.line_list:
            if points > max_points:
                new_moves = []
                for x in line:
                    if game.islegal(x):
                        new_moves.append(x)
                if new_moves:
                    moves = new_moves
                    max_points = points
            elif points == max_points:
                for move in line:
                    if not move in moves and game.islegal(move):
                        moves.append(move)

        output = random.choice(moves)
        for index, line in enumerate(self.line_list):
            if output in line[0]:
                self.line_list[index][1] +=1

        self.game_length += 1
        game.move(output)
        return output
    def generate_line_list(self):
        line_list = []
        for row in range(self.game_size):
            for column in range(self.game_size):
                room_right = False
                room_left = False
                room_up = False
                room_down = False

                if column >= 4:
                    line = [(column - x, row) for x in range(5)]
                    line_list.append([line, 0])
                    room_left = True

                if row >= 4:
                    line = [(column, row - x) for x in range(5)]
                    line_list.append([line, 0])
                    room_up = True

                if self.game_size - column >= 5:
                    line = [(column + x, row) for x in range(5)]
                    line_list.append([line, 0])
                    room_right = True

                if self.game_size - row >= 5:
                    line = [(column, row + x) for x in range(5)]
                    line_list.append([line, 0])
                    room_down = True

                if room_up and room_left:
                    line = [(column - x, row - x) for x in range(5)]
                    line_list.append([line, 0])

                if room_up and room_right:
                    line = [(column + x, row - x) for x in range(5)]
                    line_list.append([line, 0])

                if room_down and room_left:
                    line = [[column - x, row + x] for x in range(5)]
                    line_list.append([line, 0])

                if room_down and room_right:
                    line = [(column + x, row + x) for x in range(5)]
                    line_list.append([line, 0])
        return line_list

if __name__ == "__main__":
    main()