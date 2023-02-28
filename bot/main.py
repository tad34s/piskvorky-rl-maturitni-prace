# This is a sample Python script.
from piskvorky import Piskvorky
from utils import velikost
from uselplayer import UserPlayer
from CNNPlayer import CNNPLayer
from mmplayer import MinimaxPlayer
from combinationplayer import CombPlayer
from utils import displaystats



# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    piskvorky = Piskvorky(velikost)
    cnn2 = CNNPLayer(velikost, name="CNN 2", load="CNN 7 8")
    cnn = CNNPLayer(velikost, name = "CNN 1", load = "CNN 3 8")
    #comb = CombPlayer(size=VELIKOST, depth=3,name="1", model=None, load="CNN 5")
    minim = MinimaxPlayer(3, name="Minimax 1")
    #user = UserPlayer()

    #play(piskvorky,player1,player2)
    test(piskvorky,10,cnn,cnn2)

def play(game, player1, player2):
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
        move = turn.move(game, move)
        print(str(game))
        if game.end(move):
            vysledek = game.end(move)
            break
        turn, waiting = waiting, turn

    print(vysledek, turn.side)
    if vysledek == turn.side:
        print(f"Vyhrál {turn.name}")
        return turn.name
    elif vysledek == waiting.side:
        print(f"Vyhrál {waiting.name}")
        return waiting.name
    elif vysledek == "0":
        print("Remizovali")
        return "0"
    else:
        raise Exception("Game.end returned weird value")


def test(game, length, player1, player2):
    log = []
    for i in range(length):
        log.append(play(game, player1, player2))
        player1,player2 = player2, player1
    displaystats(log, player1.name, player2.name)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
