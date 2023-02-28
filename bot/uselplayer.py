
class UserPlayer():

    def __init__(self):
        nameinput = input("Jmeno: ")
        self.name = "User " + nameinput
        pass

    def newgame(self,side,other):
        self.side = side
        self.wait = other

    def move(self,game):
        print(str(game))
        inputmove = input("X Y:")
        x,y = [int(x) for x in inputmove.split()]
        game.move((x,y))
        return x,y