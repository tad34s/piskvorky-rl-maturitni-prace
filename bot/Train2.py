from piskvorky import Piskvorky
from variables import VELIKOST, X,O
import Players as players
from matplotlib.pyplot import plot,show
def play_n_games(game:Piskvorky,CNN_player:players.CNNPlayer_proximal, opponent:players.Player, n:int):
    starter = opponent
    waiting = CNN_player
    who_won = []
    for match in range(n):
        print(f"match: {match}")
        result = play_game(game,starter,waiting)
        if result == X:
            who_won.append(starter.name)
        elif result == O:
            who_won.append(waiting.name)
        else:
            who_won.append(0)
        starter,waiting = waiting,starter
    CNN_player.train(20)
    return who_won


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
    return result

if __name__ == "__main__":
    game = Piskvorky(VELIKOST)
    cnn_player = players.CNNPlayer_proximal(size = VELIKOST, name = str(1),to_train = True,preset = True,minimax_prob=0.0,random_move_prob=0.9,random_move_decrease=0.96)
    random_player = players.LinesPlayer(name="line",game_size=VELIKOST)
    results_log = []
    steps = 150
    length = 150
    for step in range(steps):
        print(f"step: {step}")
        who_won = play_n_games(game,cnn_player,random_player,length)
        results_log.append(who_won.count(cnn_player.name))
    plot(range(len(results_log)), results_log)
    show()
