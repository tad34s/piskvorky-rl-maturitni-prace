from CNNPlayer import CNNPLayer
from variables import VELIKOST
import random
from piskvorky import Piskvorky

def reward(game:Piskvorky, move:tuple) -> float:
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



if __name__ == "__main__":
    main()