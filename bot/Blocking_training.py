from CNNPlayer import CNNPLayer
from variables import VELIKOST
import random
from piskvorky import Piskvorky
from copy import copy

class TeacherPlayer():

    def __init__(self, game_size):
        self.game_size = game_size
        self.line_list = self.generate_line_list()
        self.to_train = False
        self.game_length = 0

    def new_game(self,side,other):
        self.line_list = self.generate_line_list()
        self.game_length = 0
    def reward(self, game: Piskvorky, enemy_move: tuple):
        reward_points = 0
        points = []
        for index, line in enumerate(self.line_list):
            if enemy_move in line[0]:
                points.append(copy(line[1]))
                self.line_list[index][1] = 0
        points.sort(reverse=True)
        print(points)
        reward_points = sum([(x**2)/(30 *(i+1))
                             for i,x in enumerate(points)])
        reward_points += self.game_length/80
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
                print(line)
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


                if row >= 4 and column >=4:
                    line = [(column - x, row - x) for x in range(5)]
                    line_list.append([line, 0])
                    line = [(column -4 + x, row - x) for x in range(5)]
                    line_list.append([line, 0])

        return line_list



def main():
    piskvorky = Piskvorky(VELIKOST)
    n_cnn_players = 10
    cnn_players = []
    episodes = 100
    #waiting = list(range(n_cnn_players))
    teacher = TeacherPlayer(piskvorky.size)
    player = CNNPLayer(VELIKOST, memory_size=100, name=str(184), load=False, to_train=True,pretraining= True,
                       block_training=teacher.reward)
    counter = 0
    for i in range(100):
        #pick = random.sample(waiting, k=1)
        #print(pick)
        print("episode",counter)
        if counter == 6:
            player.pretraining = False
        train_blocking(piskvorky, player,teacher)
        counter+=1


def train_blocking(game: Piskvorky,player: CNNPLayer, teacher: TeacherPlayer,):

    n_games = 100
    for i in range(n_games):
        game.reset()
        print(i)
        turn = teacher
        waiting = player
        teacher.new_game(game.X,game.O)
        player.new_game(game.O,game.X)
        move = None
        while True:
            move = turn.move(game, move)
            print(str(game))
            if game.end(move):
                vysledek = game.end(move)
                break
            turn, waiting = waiting, turn

        player.train(vysledek, epochs=10, n_recalls=60)
    player.save_model()


if __name__ == "__main__":
    main()