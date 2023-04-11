from piskvorky import Piskvorky, play_game
from variables import VELIKOST, X,O
import Players as players
from matplotlib.pyplot import plot,show
from alpha_gomoku.Alpha_player import AplhaPlayer

def play_n_games(game:Piskvorky,CNN_player:players.Player, opponent:players.Player, n:int):
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
        #starter,waiting = waiting,starter
    CNN_player.train(20)
    return who_won




if __name__ == "__main__":
    game = Piskvorky(VELIKOST)
    cnn_player = players.CNNPlayer_proximal(VELIKOST,"181",preset = True,to_train = True,load=False,pretraining=False,double_dqn=False,restrict_movement=True)
    random_player = players.LinesPlayer(name="line",game_size=VELIKOST)
    results_log = []
    steps = 200
    length = 100
    for step in range(steps):
        print(f"step: {step}")
        who_won = play_n_games(game,cnn_player,random_player,length)
        results_log.append(who_won.count(cnn_player.name))
    plot(range(len(results_log)), results_log)
    show()
