# Press the green button in the gutter to run the script.
from piskvorky import Piskvorky
from mmplayer import MinimaxPlayer
from CNNPlayer import CNNPLayer
import matplotlib.pyplot as plt
from utils import velikost
from combinationplayer import CombPlayer
from utils import displaystats


def main():
    piskvorky = Piskvorky(velikost)
    cnnp1 = CNNPLayer(velikost, name="4", to_train=True)
    cnnp2 = CNNPLayer(velikost,name="2",to_train = True)
    minimax2 = MinimaxPlayer(depth=6, name="2")
    minimax1 = MinimaxPlayer(depth=2, name="1")
   # comb = CombPlayer(size=velikost, depth=3, name="1", model=None, load="CNN 6")
    # CNNQplayer = CNNQPlayer(velikost)
    game_record,games_len = train(piskvorky, cnnp1, minimax1)
    cnnp1.save_model()
    displaystats(game_record, games_len, cnnp1.name, minimax1.name)


def train(game, player1, player2):
    number_of_games = 800
    games_won = []
    games_len =[]

    for i in range(1, number_of_games + 1):
        print(i)
        game.reset()
        starter = player1.name
        seconder = player2.name
        turn = player1
        waiting = player2
        player1.newgame(side=game.X, other=game.O)
        player2.newgame(side=game.O, other=game.X)
        vysledek = 0
        n_of_moves = 0
        move = None

        while True:
            move = turn.move(game,move)
            print(str(game))
            n_of_moves += 1
            if game.end(move):
                vysledek = game.end(move)
                break
            turn,waiting = waiting,turn

        if vysledek == 1:
            starter_reps = 5
            seconder_reps = 2
            games_won.append(starter)
        elif vysledek == 2:
            starter_reps = 2
            seconder_reps = 5
            games_won.append(seconder)
        else:
            starter_reps = 3
            seconder_reps = 3
            games_won.append(vysledek)

        if starter.to_train:
            starter.train(vysledek,epochs = 20, reps = starter_reps)
        if seconder.to_train:
            seconder.train(vysledek, epochs = 20, reps = seconder_reps)

        games_len.append(n_of_moves)



        player1, player2 = player2, player1


    return games_won,games_len

if __name__ == '__main__':
    main()
