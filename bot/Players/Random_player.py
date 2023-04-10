from bot.Players.Player_abstract_class import Player
from piskvorky import list_of_possible_moves
import random

class RandomPlayer(Player):

    def __init__(self,name:str):
        super().__init__(name = "Random " + name)

    def move(self,game,enemy_move) ->tuple:
        move = random.choice(list_of_possible_moves(game.state))
        game.move(move)
        return move

    def new_game(self,side,other) ->None:
        self.side = side
        self.other = other