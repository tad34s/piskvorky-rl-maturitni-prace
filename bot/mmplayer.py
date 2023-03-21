import random


class MinimaxPlayer():

    def __init__(self, depth, name):
        self.depth = depth
        self.name = "Minim " + name
        self.to_train = False
        self.cache = {}

    def newgame(self, side, other):
        self.cache = {}
        self.side = side
        self.other = other

    def listofpossiblemoves(self, game):
        movelist = []

        for y, i in enumerate(game.state):
            for x, j in enumerate(i):
                if j == game.EMPTY:
                    movelist.append((x, y))

        return movelist


    def move(self, game, enemy_move):
        xy = minimax(game,self.depth,heuristic)

        game.move(xy)
        return xy


def heuristic(game, move):
    row =game.rowpoints(move[0], move[1])
    column =game.columnpoints(move[0], move[1])
    leftdiag = game.leftdiagpoints(move[0], move[1])
    rightdiag = game.rightdiagpoints(move[0], move[1])

    value = max((row,column,leftdiag,rightdiag))
    value = value / 6

    return value


def minimax(game, depth,heuristic):
    def maxx(alpha, beta, depth, maxdepth):
        maxv = -2000
        maxx = None
        maxy = None

        for mov in listofpossiblemoves(game):

            game.move(mov)

            depth += 1

            # nechci delat cely game tree tak checkuju hloubku
            # pridat rozhodovani podle heuristiky
            if depth > maxdepth:
                value = heuristic(game, mov)
                game.insertempty(mov)
                depth -= 1
                return value, mov[0], mov[1]

            if game.end(mov) != "0" and game.end(mov):
                game.insertempty(mov)
                depth -= 1
                return 10, mov[0], mov[1]
            if not listofpossiblemoves(game):
                game.insertempty(mov)
                depth -= 1
                return 0, mov[0], mov[1]

            val, x, y = minn(alpha, beta, depth, maxdepth)

            game.insertempty(mov)

            if val > maxv:
                maxv = val
                maxx = x
                maxy = y

            if maxv >= beta:
                depth -= 1
                return maxv, maxx, maxy

            if maxv > alpha:
                alpha = maxv
            depth -= 1
        return maxv, maxx, maxy

    def minn(alpha, beta, depth, maxdepth):

        minv = 2000
        minx = None
        miny = None

        for mov in listofpossiblemoves(game):
            game.move(mov)

            depth += 1

            # nechci delat cely game tree tak checkuju hloubku
            if depth > maxdepth:
                if depth > maxdepth:
                    value = heuristic(game, mov)
                    game.insertempty(mov)
                    depth -= 1
                    return value *(-1), mov[0], mov[1]

            if game.end(mov) != "0" and game.end(mov):
                game.insertempty(mov)
                depth -= 1
                return -10, mov[0], mov[1]
            if not listofpossiblemoves(game):
                game.insertempty(mov)
                depth -= 1
                return 0, mov[0], mov[1]

            val, x, y = maxx(alpha, beta, depth, maxdepth)
            game.insertempty(mov)

            if val < minv:
                minv = val
                minx = x
                miny = y

            if minv <= alpha:
                depth -= 1

                return minv, minx, miny

            if minv < beta:
                beta = minv

            depth -= 1
        return minv, minx, miny

    val, x, y = maxx(-200, 200, 0, depth)

    if val == 0:
        print("lol")
        return random.choice(listofpossiblemoves(game))

    return x, y


def listofpossiblemoves(game):
    movelist = []

    for y, i in enumerate(game.state):
        for x, j in enumerate(i):
            if j == game.EMPTY:
                movelist.append((x, y))

    return movelist
