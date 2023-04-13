from bot.Player_abstract_class import Player
from piskvorky import list_of_possible_moves, Piskvorky
import random


class RandomPlayer(Player):
    """
    Player that makes random moves
    """
    def __init__(self, name: str):
        super().__init__(name="Random " + name)

    def move(self, game: Piskvorky, enemy_move: tuple) -> tuple:
        move = random.choice(list_of_possible_moves(game.state))
        game.move(move)
        return move

