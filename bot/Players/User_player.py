from bot.Players.Player_abstract_class import Player


class UserPlayer(Player):

    def __init__(self):
        nameinput = input("Jmeno: ")
        super().__init__("User " + nameinput)

    def new_game(self, side, other):
        self.side = side
        self.wait = other

    def move(self, game,enemy_move):
        print(str(game))
        inputmove = input("X Y:")
        x, y = [int(x) for x in inputmove.split()]
        game.move((x, y))
        return x, y
