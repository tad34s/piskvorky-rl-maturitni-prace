# Press the green button in the gutter to run the script.
import random

from piskvorky import Piskvorky
from bot.Players.CNNPlayer import CNNPLayer
from variables import VELIKOST


def main():
    piskvorky = Piskvorky(VELIKOST)
    n_cnn_players= 10
    cnn_players = []
    waiting = list(range(n_cnn_players))
    for i in range(n_cnn_players):
        cnn_players.append(CNNPLayer(VELIKOST,memory_size=50, name="18",load=False, to_train=True,minimax_prob=0))
    while True:

        picks = random.sample(waiting, k=2)
        cnnp1 = cnn_players[picks[0]]
        cnnp2 = cnn_players[picks[1]]
        print(cnnp1.name, cnnp2.name)
        game_record, games_len = train(piskvorky, cnnp1, cnnp2)
        cnnp1.save_model()
        cnnp2.save_model()


def train(game, player1, player2):
    number_of_games = 1
    games_won = []
    games_len = []

    for i in range(1, number_of_games + 1):
        print(i)
        game.reset()
        starter = player1.name
        seconder = player2.name
        turn = player1
        waiting = player2
        player1.new_game(side=game.X, other=game.O)
        player2.new_game(side=game.O, other=game.X)
        vysledek = 0
        n_of_moves = 0
        move = None

        while True:
            move = turn.move(game, move)
            print(str(game))
            n_of_moves += 1
            if game.end(move):
                vysledek = game.end(move)
                break
            turn, waiting = waiting, turn

        if vysledek == 1:

            games_won.append(starter)
        elif vysledek == 2:
            games_won.append(seconder)
        else:
            games_won.append(vysledek)

        multiplier = 10
        if player1.to_train:
            player1.train(vysledek, epochs=20, n_recalls=multiplier)
        if player2.to_train:
            player2.train(vysledek, epochs=20, n_recalls=multiplier)

        games_len.append(n_of_moves)

        player1, player2 = player2, player1

    return games_won, games_len


if __name__ == '__main__':
    main()
