from bot.Player_abstract_class import Player


class UserPlayer(Player):
    # used for playing against AI in the command line
    def __init__(self):
        name_input = input("Jmeno: ")
        super().__init__("User " + name_input)

    def new_game(self, side: int, other: int):
        self.side = side
        self.wait = other

    def move(self, game,enemy_move):
        print(str(game))
        input_move = input("X Y:")
        x, y = [int(x) for x in input_move.split()]
        game.move((x, y))
        return x, y
