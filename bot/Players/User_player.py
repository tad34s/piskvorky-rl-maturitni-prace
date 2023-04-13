from bot.Player_abstract_class import Player
from piskvorky import Piskvorky


class UserPlayer(Player):
    """
    Player used for playing against AI in the command line
    """
    def __init__(self):
        name_input = input("Jmeno: ")
        super().__init__("User " + name_input)


    def move(self, game:Piskvorky, enemy_move:tuple)->tuple:
        print(str(game))
        input_move = input("X Y:")
        x, y = [int(x) for x in input_move.split()]
        game.move((x, y))
        return x, y
