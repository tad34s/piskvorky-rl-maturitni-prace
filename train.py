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
    minimax1 = MinimaxPlayer(depth=3, name="1")
    comb = CombPlayer(size=velikost, depth=3, name="1", model=None, load="CNN 6")
    # CNNQplayer = CNNQPlayer(velikost)
    game_record = train(piskvorky, cnnp1, minimax1)
    cnnp1.save_model()
    displaystats(game_record, cnnp1.name, minimax1.name)


def train(game, player1, player2):
    number_of_games = 100
    games_won = []

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
        move = None

        while True:
            move = turn.move(game,move)
            print(str(game))
            if game.end(move):
                vysledek = game.end(move)
                break
            turn,waiting = waiting,turn

        if player1.to_train:
            player1.train(vysledek,epochs = 20, reps = 20)
        if player2.to_train:
            player2.train(vysledek, epochs = 20, reps = 20)

        if vysledek == 1:
            games_won.append(starter)
        elif vysledek == 2:
            games_won.append(seconder)
        else:
            games_won.append(vysledek)

        player1, player2 = player2, player1


    return games_won

if __name__ == '__main__':
    main()
