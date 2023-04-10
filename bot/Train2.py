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
    cnn_player = AplhaPlayer(VELIKOST,"1",100,True,True,False,False)
    random_player = players.LinesPlayer(name="line",game_size=VELIKOST)
    results_log = []
    steps = 100
    length = 50
    for step in range(steps):
        print(f"step: {step}")
        who_won = play_n_games(game,cnn_player,random_player,length)
        results_log.append(who_won.count(cnn_player.name))
    plot(range(len(results_log)), results_log)
    show()
