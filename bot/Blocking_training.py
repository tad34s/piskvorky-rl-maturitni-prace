from CNNPlayer import CNNPLayer
from variables import VELIKOST
import random
from piskvorky import Piskvorky


class TeacherPlayer():

    def __init__(self,game_size):
        self.game_size = game_size
        self.line_list = self.generate_line_list(game_size)

    def generate_line_list(self):
        pass

    def reward(self,game:Piskvorky,enemy_move:tuple):
        reward_points = 0
        for index,line,points in enumerate(self.line_list):
            if enemy_move in line:
                reward_points += points
                self.line_list[index][1] = 0
        reward_points /= 10
        return reward_points

    def move(self,game:Piskvorky,enemy_move:tuple):
        moves = []
        max_points = -1
        for line,points in self.line_list:
            if points > max_points:
                moves = line
            elif points == max_points:
                for move in line:
                    if not move in moves:
                        moves.append(move)

def reward(game:Piskvorky, move:tuple) -> float:
    pass
# replaces the reward function in the cnn player
# the total points it got from blocking the line
def main():
    piskvorky = Piskvorky(VELIKOST)
    n_cnn_players = 10
    cnn_players = []
    episodes = 100
    waiting = list(range(n_cnn_players))
    for i in range(n_cnn_players):
        cnn_players.append(CNNPLayer(VELIKOST, memory_size=50, name=str(i), load=True, to_train=True,block_training=reward))
    for i in range(episodes):
        picks = random.sample(waiting, k=1)
        cnnp1 = cnn_players[picks[0]]
        print(cnnp1.name)
        game_record, games_len = train_blocking(piskvorky, cnnp1)
        cnnp1.save_model()


def train_blocking(game: Piskvorky,player: CNNPLayer):

    n_games = 1000
    teacher = TeacherPlayer()
    for i in range(n_games):
        list_of_lines = generate_list_lines(game)

        while True:
            pass
def generate_list_lines(game: Piskvorky):
    pass

if __name__ == "__main__":
    main()