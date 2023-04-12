from bot.Players.CNNPlayer import CNNPlayer
from variables import VELIKOST
from piskvorky import Piskvorky
from matplotlib.pyplot import plot,show
from bot.Players.Lines_player import LinesPlayer
from Networks import CNNetwork_big


def main():
    piskvorky = Piskvorky(VELIKOST)
    n_cnn_players = 10
    cnn_players = []
    episodes = 100
    # waiting = list(range(n_cnn_players))
    teacher = LinesPlayer(piskvorky.size,"teacher")
    model = CNNetwork_big(VELIKOST, str(185))
    player = CNNPlayer(VELIKOST,model = model, memory_size=1000, name=str(185), to_train=True, pretraining=True,
                       block_training=teacher.reward, double_dqn=True, minimax_prob=0, random_move_prob=0.99, random_move_decrease=0.9997)
    counter = 0
    rewards_history = []
    for i in range(episodes):
        # pick = random.sample(waiting, k=1)
        # print(pick)
        print("episode", counter)
        if counter == 6:
            player.pretraining = False
        rewards = train_blocking(piskvorky, player, teacher)

        rewards_history.append(rewards)
        counter += 1

    plot(range(len(rewards_history)), rewards_history)
    show()


def train_blocking(game: Piskvorky, player: CNNPlayer, teacher: LinesPlayer, ):
    n_games = 100
    rewards = 0
    for i in range(n_games):
        game.reset()
        print(i)
        turn = teacher
        waiting = player
        teacher.new_game(game.X, game.O)
        player.new_game(game.O, game.X)
        move = None
        while True:
            move = turn.move(game, move)
            print(str(game))
            if game.end(move):
                vysledek = game.end(move)
                break
            turn, waiting = waiting, turn
        player.train(vysledek, epochs=20, n_recalls=200)
        rewards += sum(player.curr_match_reward_log)
    player.model.save()
    return rewards


if __name__ == "__main__":
    main()
