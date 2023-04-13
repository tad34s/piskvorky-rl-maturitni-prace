from abc import ABC, abstractmethod
class Player(ABC):
    """
    Abstract class for all players
    """

    def __init__(self, name: str, to_train: bool = False):
        self.name = name
        self.to_train = to_train

    def new_game(self, side: int, other: int) -> None:
        """

        :param side: The symbol it will be playing as
        :param other: The opponents symbol
        :return:
        """
        self.side = side
        self.opponent = other

    @abstractmethod
    def move(self, game, enemy_move: tuple) -> tuple:
        """
        :param game: Piskvorky game object
        :param enemy_move: The move opponent made
        :return: The move it made
        """
        pass

    def game_end(self, result) -> None:
        """
        Called after game ended
        :param result:
        :return:
        """
        pass
