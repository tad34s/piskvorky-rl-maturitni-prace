from abc import ABC, abstractmethod,abstractproperty

class Player(ABC):

    def __init__(self,name,to_train = False):
        self.name = name
        self.to_train = to_train
    @abstractmethod
    def new_game(self,side,other)->None:
        pass

    @abstractmethod
    def move(self,game,enemy_move)->tuple:
        pass

    def game_end(self,result)->None:
        pass