import pygame
import sys
from variables import VELIKOST
from piskvorky import Piskvorky
from bot.CNNPlayer import CNNPLayer
import os
from bot.mmplayer import MinimaxPlayer
from bot.combinationplayer import CombPlayer
from time import sleep


BLACK = (0, 0, 0)
GRID_COLOR = (19, 20, 20)
SQUARE_COLOR = (37, 40, 43)
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
CIRCLE_COLOR = (51, 163, 171)
X_COLOR = (209, 54, 78)
PADDING = 2


def main():
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    CLOCK = pygame.time.Clock()
    drawGrid()
    print(os.getcwd())

    game = Piskvorky(VELIKOST)
    game_ended = False
    new_game=True

    turn_user = True
    AI = CNNPLayer(VELIKOST,name="9", load = "..\\bot\\CNN 10 8",to_train = False)
    #AI = MinimaxPlayer(3, name="nicitel")
   # AI = CombPlayer(depth=2,size=VELIKOST,name="skolovac",model="..\\bot\\CNN 10 8")

    AI.newgame(side=game.O, other=game.X)
    vysledek = 0
    while True:
        if new_game:
            new_game = False
            drawGrid()

        if game_ended:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                        game.reset()
                        AI.newgame(side=game.O, other=game.X)
                        turn_user = True
                        game_ended = False
                        new_game = True
                        continue

            if vysledek==1:
                showMessage("Vyhrál jsi")
            elif vysledek==2:
                showMessage("Prohrál jsi")
            else:
                showMessage("Remíza")



            # text s výsledkem, čára přes vyhranou věc, nejde dál psát
        else:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:


                    if turn_user:
                        move = get_move(game)
                        if move:
                            game.move(move)
                            drawBoard(game)
                            turn_user = False
                            if vysledek := game.end(move):
                                print(vysledek)
                                game_ended = True
                            continue


                if not turn_user:
                    move = AI.move(game, False)
                    drawBoard(game)
                    if vysledek := game.end(move):
                        game_ended = True
                    turn_user = True

            # nakreslit z boardu z game.board
            # poslouchat event lick, získat pozici, konvertovat na move, ceknout jestli je legalni a je na rade jestli ano udelat ho
        pygame.display.update()


def drawGrid():
    SCREEN.fill(SQUARE_COLOR)
    blocks_size = int(WINDOW_WIDTH / VELIKOST)  # Set the size of the grid block
    for x in range(0, WINDOW_WIDTH, blocks_size):
        for y in range(0, WINDOW_HEIGHT, blocks_size):
            rect = pygame.Rect(x, y, blocks_size, blocks_size)
            pygame.draw.rect(SCREEN, GRID_COLOR, rect, 1)


def drawBoard(game):
    print(str(game))
    block_size = int(WINDOW_WIDTH / VELIKOST)  # Set the size of the grid block
    for ix, x in enumerate(range(0, WINDOW_WIDTH, block_size)):
        for iy, y in enumerate(range(0, WINDOW_HEIGHT, block_size)):
            if game.state[iy, ix] == game.X:
                drawX(SCREEN, X_COLOR, (x + PADDING, y + PADDING), (x + block_size - PADDING, y + block_size - PADDING))
            elif game.state[iy, ix] == game.O:
                pygame.draw.circle(SCREEN, CIRCLE_COLOR, (x + block_size / 2, y + block_size / 2),
                                   block_size / 2 - PADDING,width=10)


def drawX(screen, color, xy, end):
    x, y = xy
    ex, ey = end

    thickness = 10
    for i in range(thickness):
        pygame.draw.aaline(screen, color, (x + i, y),
                           (ex - thickness + i, ey))  # start_pos(x+thickness,y)---end_pos(width+x+thickness,height+y)
        pygame.draw.aaline(screen, color, (ex - thickness + i, y), (x + i, ey))

def get_move(game):
    blocks_size = int(WINDOW_WIDTH / VELIKOST)  # Set the size of the grid block
    x,y = pygame.mouse.get_pos()
    ix = x // blocks_size
    iy = y // blocks_size
    if game.state[iy,ix] == game.EMPTY:
        return ix,iy
    else:
        return None
    # gets the space from the mouse position
    # returns xy if not is illegal None
    pass

def showMessage(message):
    FONT = pygame.font.Font('freesansbold.ttf', 64)
    text = FONT.render((message), True, (255, 255, 255))
    SCREEN.blit(text, (WINDOW_WIDTH/2-150, WINDOW_HEIGHT/2-150))
    pass
# displays a message on board

if __name__ == "__main__":
    main()
