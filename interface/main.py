import pygame
import sys
import torch
from variables import GAME_SIZE, X, O
from piskvorky import Piskvorky
import os
from bot.Players.DQN.Networks import CNNetwork_preset
from bot.Players.Combination_player import CombPlayer
from bot.Players.Alpha_Zero.Alpha_player import AlphaPlayer
from bot.Players.Alpha_Zero.model import AlphaCNNetwork_preset
from bot.Players.DQN.CNNPlayer2 import CNNPlayer_proximal

# constants
GRID_COLOR = (19, 20, 20)
SQUARE_COLOR = (37, 40, 43)
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
CIRCLE_COLOR = (51, 163, 171)
X_COLOR = (209, 54, 78)
PADDING = 2  # padding for symbols so they do not cross grid lines


def main() -> None:
    global SCREEN, CLOCK

    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    CLOCK = pygame.time.Clock()
    drawGrid()
    print(os.getcwd())

    game = Piskvorky(GAME_SIZE)
    game_ended = False
    new_game = True
    turn_user = True  # user starts

    model = AlphaCNNetwork_preset(GAME_SIZE, "123", load="../bot/Players/Alpha_Zero/NNs_preset/Alpha-Zero-123-8.nn")
    AI = AlphaPlayer(GAME_SIZE, model, "123", 500, True, restrict_movement=True)

    # AI = MinimaxPlayer(3, name="nicitel")
    # model = CNNetwork_preset(GAME_SIZE,"184","..\\bot\\Players\\DQN\\NNs_preset\\CNN-proximal-184-8.nn")
    # AI = CNNPlayer_proximal(GAME_SIZE,"184",model,random_move_prob=0,restrict_movement=True)
    # AI = CombPlayer(depth=3,size=GAME_SIZE,name="skolovac",model=model,restrict_movement=True)

    AI.new_game(side=game.O, other=game.X)
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
                    AI.new_game(side=game.O, other=game.X)
                    turn_user = True
                    game_ended = False
                    new_game = True
                    continue

            if vysledek == X:
                showMessage("Vyhrál jsi")
            elif vysledek == O:
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

                if not turn_user and not game_ended:
                    move = AI.move(game, False)
                    drawBoard(game)
                    if vysledek := game.end(move):
                        print(vysledek)
                        game_ended = True
                    turn_user = True

            # nakreslit z boardu z game.board
            # poslouchat event lick, získat pozici, konvertovat na move, ceknout jestli je legalni a je na rade jestli ano udelat ho
        pygame.display.update()


def drawGrid() -> None:
    """
    Draws grid lines.
    :return:
    """
    SCREEN.fill(SQUARE_COLOR)
    blocks_size = int(WINDOW_WIDTH / GAME_SIZE)  # Set the size of the grid block
    for x in range(0, WINDOW_WIDTH, blocks_size):
        for y in range(0, WINDOW_HEIGHT, blocks_size):
            rect = pygame.Rect(x, y, blocks_size, blocks_size)
            pygame.draw.rect(SCREEN, GRID_COLOR, rect, 1)


def drawBoard(game: Piskvorky) -> None:
    """
    Draws the state of the board.
    :param game:
    :return:
    """
    print(str(game))
    block_size = int(WINDOW_WIDTH / GAME_SIZE)  # Set the size of the grid block
    for ix, x in enumerate(range(0, WINDOW_WIDTH, block_size)):
        for iy, y in enumerate(range(0, WINDOW_HEIGHT, block_size)):
            if game.state[iy, ix] == game.X:
                drawX(SCREEN, X_COLOR, (x + PADDING, y + PADDING), (x + block_size - PADDING, y + block_size - PADDING))
            elif game.state[iy, ix] == game.O:
                pygame.draw.circle(SCREEN, CIRCLE_COLOR, (x + block_size / 2, y + block_size / 2),
                                   block_size / 2 - PADDING, width=10)


def drawX(screen: pygame.Surface, color: tuple, xy: tuple, end: tuple):
    """
    Draws the X symbol
    :param screen:
    :param color:
    :param xy: start pixels location
    :param end: end pixels location
    :return:
    """
    x, y = xy
    ex, ey = end

    thickness = 10
    for i in range(thickness):
        pygame.draw.aaline(screen, color, (x + i, y),
                           (ex - thickness + i, ey))
        pygame.draw.aaline(screen, color, (ex - thickness + i, y), (x + i, ey))


def get_move(game: Piskvorky) -> tuple or None:
    """
    Gets the position of move on the board from the pixel location of cursor.
    :param game:
    :return: xy or None
    """
    blocks_size = int(WINDOW_WIDTH / GAME_SIZE)  # Set the size of the grid block
    x, y = pygame.mouse.get_pos()
    ix = x // blocks_size
    iy = y // blocks_size
    if game.state[iy, ix] == game.EMPTY:
        return ix, iy
    else:
        return None


def showMessage(message: str) -> None:
    """
    Displays a massage on the screen.
    :param message:
    :return:
    """
    FONT = pygame.font.Font('freesansbold.ttf', 64)
    text = FONT.render(message, True, (255, 255, 255))
    SCREEN.blit(text, (WINDOW_WIDTH / 2 - 150, WINDOW_HEIGHT / 2 - 150))


if __name__ == "__main__":
    main()
