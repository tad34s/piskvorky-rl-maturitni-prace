# Press the green button in the gutter to run the script.
import random
from piskvorky import Piskvorky, play_game
from bot.Players import CNNPlayer
from variables import GAME_SIZE, X, O
from bot.Players.DQN.Networks import CNNetwork_big


def train(game: Piskvorky, player1: CNNPlayer, player2: CNNPlayer, n_of_matches: int) -> list:
    """
    Play n matches against each other.

    :param game:
    :param player1:
    :param player2:
    :return: winning history
    """
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
    # Here we simply create a bunch of players and then pit them against each other for n matches, after each match
    # we train.
    piskvorky = Piskvorky(GAME_SIZE)
    n_cnn_players = 10
    cnn_players = []
    waiting = list(range(n_cnn_players))
    for i in range(n_cnn_players):
        model = CNNetwork_big(GAME_SIZE, str(i))
        cnn_players.append(
            CNNPlayer(GAME_SIZE, model=model, memory_size=50, name=str(i), to_train=True, minimax_prob=0))

    while True:
        # pick random players
        picks = random.sample(waiting, k=2)
        cnnp1 = cnn_players[picks[0]]
        cnnp2 = cnn_players[picks[1]]
        print(cnnp1.name, cnnp2.name)
        # play games
        game_record, games_len = train(piskvorky, cnnp1, cnnp2, 100)
        cnnp1.model.save()
        cnnp2.model.save()
