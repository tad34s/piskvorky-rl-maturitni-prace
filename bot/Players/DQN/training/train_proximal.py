from piskvorky import Piskvorky
from variables import GAME_SIZE, X, O
from bot.Player_abstract_class import Player
import bot.Players as player

from bot.Players.DQN.Networks import CNNetwork_preset
from copy import copy
import random
import concurrent.futures


def play_training_game(game:Piskvorky, player1:Player, player2:Player)->tuple:
    """
    Make 2 players play a game. If they are training return the log of states, also return final reward.
    :param game:
    :param player1:
    :param player2:
    :return:
    """
    game.reset()
    states1 = []
    states2 = []
    player1.new_game(side=game.X, other=game.O)
    player2.new_game(side=game.O, other=game.X)
    turn = player1
    waiting = player2
    move = None
    while True:
        move = turn.move(game, move)
        if game.end(move):
            result = game.end(move)
            break
        turn, waiting = waiting, turn
    player1.game_end(result)
    player2.game_end(result)
    if player1.to_train:
        states1.extend(player1.match_state_log)
    if player2.to_train:
        states2.extend(player2.match_state_log)
    if result == X:
        reward = 1
    elif result == O:
        reward = -1
    else:
        reward = 0
    return states1, states2, reward


def get_list_of_matches(n_cnn_players:int, n_all_players:int, n_matches:int):
    """
    Generate a list of tuples, where each tuple has 2 numbers, that correspond to indexes of some
    players in list all_players. One tuple represents one match. There has to be at least one CNNPlayer
    in each match.

    :param n_cnn_players:
    :param n_all_players:
    :param n_matches:
    :return:
    """
    list = []
    players1 = random.choices(range(0, n_cnn_players), k=n_matches)
    for index in players1:
        while True:
            index2 = random.choice(range(0, n_all_players))
            if index2 != index:
                break
        list.append((index, index2))
    return list


def play_n_games(game: Piskvorky, CNN_players: list, opponents: list, n: int)->None:

    all_players = CNN_players + opponents
    list_of_matches = get_list_of_matches(len(CNN_players), len(all_players), n)
    # parallelize the training
    with concurrent.futures.ProcessPoolExecutor() as executor:
        if __name__ == "__main__":
            processes = []
            for match in list_of_matches:
                print(f"match: {match}")
                processes.append(
                    executor.submit(play_training_game, copy(game), copy(all_players[match[0]]), copy(all_players[match[1]])))
            for e, process in enumerate(processes):
                states1, states2, reward = process.result()
                # add the data to their caches
                if states1:
                    CNN_players[list_of_matches[e][0]].cache.add_states(states1, reward)
                if states2:
                    CNN_players[list_of_matches[e][1]].cache.add_states(states2, reward * (-1))

    # train all model
    for CNN_player in CNN_players:
        CNN_player.train(5, )
        CNN_player.model.save()



if __name__ == "__main__":
    # Here we train the proximal CNNPlayer by creating a number of them and then pitting them against each other.
    # We will also add some other players to spicy things up.
    game = Piskvorky(GAME_SIZE)
    n_cnn_players = 10
    cnn_players = []
    for i in range(n_cnn_players):
        model = CNNetwork_preset(game.size, str(i))
        cnn_player = player.CNNPlayer_proximal(GAME_SIZE, str(i), model, to_train=True, restrict_movement=True,
                                               random_move_decrease=0.997, minimax_prob=0.0)
        cnn_players.append(cnn_player)

    opponents = [player.LinesPlayer(name="line", game_size=GAME_SIZE),
                 player.MinimaxPlayer(depth=3, name="minimax", restrict_movement=True),
                 player.RandomPlayer("1")]
    steps = 500 # number of trainings
    length = 100 # number of games played
    for step in range(steps):
        print(f"step: {step}")
        play_n_games(game, cnn_players, opponents, length)
