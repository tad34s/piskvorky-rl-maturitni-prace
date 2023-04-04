from piskvorky import Piskvorky
from variables import VELIKOST
import Players as players
def play_n_games(game:Piskvorky,CNN_player:players.CNNPlayer_proximal, opponent:players.Player, n:int):
    starter = opponent
    waiting = CNN_player
    for match in range(n):
        print(match)
        play_game(game,starter,waiting)
        starter,waiting = waiting,starter
    CNN_player.train(20)


def play_game(game: Piskvorky, player1:players.Player, player2:players.Player):
    game.reset()
    player1.new_game(side=game.X, other=game.O)
    player2.new_game(side=game.O, other=game.X)
    turn = player1
    waiting = player2
    move = None
    while True:
        move = turn.move(game, move)
        print(str(game))
        if game.end(move):
            result = game.end(move)
            break
        turn, waiting = waiting, turn
    player1.game_end(result)
    player2.game_end(result)

if __name__ == "__main__":
    game = Piskvorky(VELIKOST)
    cnn_player = players.CNNPlayer_proximal(size = VELIKOST, name = str(1),to_train = True,preset = True,minimax_prob=0.0)
    random_player = players.RandomPlayer(name = str(1))
    while True:
        play_n_games(game,cnn_player,random_player,20)
