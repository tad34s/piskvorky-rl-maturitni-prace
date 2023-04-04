from piskvorky import Piskvorky


def play_game(game: Piskvorky, player1, player2):
    game.reset()
    player1.new_game(side=game.X, other=game.O)
    player2.new_game(side=game.O, other=game.X)
    turn = player1
    waiting = player2
    while True:
        move = turn.move(game, move)
        print(str(game))
        if game.end(move):
            vysledek = game.end(move)
            break
        turn, waiting = waiting, turn
    player1.game_end()
    player2.game_end()