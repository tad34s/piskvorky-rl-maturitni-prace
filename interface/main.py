import pygame
import sys
from variables import VELIKOST
from piskvorky import Piskvorky

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
    SCREEN.fill(SQUARE_COLOR)
    CLOCK = pygame.time.Clock()
    drawGrid()

    game = Piskvorky(VELIKOST)
    game.move((1, 1))
    game.move((2, 2))
    game_ended = False

    turn_user = True
    vysledek = 0
    while True:
        if game_ended:
            if vysledek==1:
                showMessage("Vyhrál jsi")
            elif vysledek==2:
                showMessage("Prohrál jsi")
            else:
                showMessage("Remíza")
            pass
            # text s výsledkem, čára přes vyhranou věc, nejde dál psát
        else:
            drawBoard(game)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if turn_user:
                        move = get_move()
                        if move:
                            game.move(move)
                            if vysledek := game.end(move):
                                game_ended = True
                            turn_user = False

            if not turn_user:
                move = AI.move(game)
                if move:
                    game.move(move)
                    if vysledek := game.end(move):
                        game_ended = True
                    turn_user = True

            pass
            # nakreslit z boardu z game.board
            # poslouchat event lick, získat pozici, konvertovat na move, ceknout jestli je legalni a je na rade jestli ano udelat ho

        pygame.display.update()


def drawGrid():
    blocks_size = int(WINDOW_WIDTH / VELIKOST)  # Set the size of the grid block
    for x in range(0, WINDOW_WIDTH, blocks_size):
        for y in range(0, WINDOW_HEIGHT, blocks_size):
            rect = pygame.Rect(x, y, blocks_size, blocks_size)
            pygame.draw.rect(SCREEN, GRID_COLOR, rect, 1)


def drawBoard(game):
    block_size = int(WINDOW_WIDTH / VELIKOST)  # Set the size of the grid block
    for ix, x in enumerate(range(0, WINDOW_WIDTH - block_size, block_size)):
        for iy, y in enumerate(range(0, WINDOW_HEIGHT - block_size, block_size)):
            if game.state[iy, ix] == 1:
                drawX(SCREEN, X_COLOR, (x + PADDING, y + PADDING), (x + block_size - PADDING, y + block_size - PADDING))
            elif game.state[iy, ix] == 2:
                pygame.draw.circle(SCREEN, CIRCLE_COLOR, (x + block_size / 2, y + block_size / 2),
                                   block_size / 2 - PADDING,width=10)


def drawX(screen, color, xy, end):
    x, y = xy
    ex, ey = end

    thickness = 10
    for i in range(thickness):
        print((x + i, y), (ex - thickness + i, ey))
        print((ex - thickness + i, y), (x + i, ey))
        pygame.draw.aaline(screen, color, (x + i, y),
                           (ex - thickness + i, ey))  # start_pos(x+thickness,y)---end_pos(width+x+thickness,height+y)
        pygame.draw.aaline(screen, color, (ex - thickness + i, y), (x + i, ey))

def get_move():
    # gets the space from the mouse position
    # returns xy if not is illegal None
    pass

def showMessage(message):
    pass
# displays a message on board

if __name__ == "__main__":
    main()
