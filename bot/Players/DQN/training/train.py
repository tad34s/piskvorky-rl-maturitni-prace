# Press the green button in the gutter to run the script.
import random
from piskvorky import Piskvorky,play_game
from bot.Players import CNNPlayer
from variables import VELIKOST,X,O
from bot.Players.DQN.Networks import CNNetwork_big


def main():
    piskvorky = Piskvorky(VELIKOST)
    n_cnn_players= 10
    cnn_players = []
    waiting = list(range(n_cnn_players))
    for i in range(n_cnn_players):
        model = CNNetwork_big(VELIKOST,str(i))
        cnn_players.append(CNNPlayer(VELIKOST, model = model,memory_size=50, name=str(i), to_train=True, minimax_prob=0))
    while True:

        picks = random.sample(waiting, k=2)
        cnnp1 = cnn_players[picks[0]]
        cnnp2 = cnn_players[picks[1]]
        print(cnnp1.name, cnnp2.name)
        game_record, games_len = train(piskvorky, cnnp1, cnnp2)
        cnnp1.model.save()
        cnnp2.model.save()


def train(game, player1, player2):
    n_of_matches = 100
    starter = player1
    waiting = player2
    who_won = []

    for match in range(1, n_of_matches + 1):

        print(f"match: {match}")
        result = play_game(game, starter, waiting)
        if result == X:
            who_won.append(starter.name)
        elif result == O:
            who_won.append(waiting.name)
        else:
            who_won.append(0)
        n_recalls = 10
        if player1.to_train:
            player1.train(result, epochs=20, n_recalls=n_recalls)
        if player2.to_train:
            player2.train(result, epochs=20, n_recalls=n_recalls)


        player1, player2 = player2, player1

    return who_won


if __name__ == '__main__':
    main()
