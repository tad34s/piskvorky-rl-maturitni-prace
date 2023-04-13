from bot.Players.DQN.CNNPlayer import CNNPlayer
from variables import GAME_SIZE
from piskvorky import Piskvorky
from matplotlib.pyplot import plot, show
from bot.Players.Lines_player import LinesPlayer
from bot.Players.DQN.Networks import CNNetwork_big


def train_blocking(game: Piskvorky, player: CNNPlayer, teacher: LinesPlayer) -> list:
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
        # to plot the rewards later, we will save them
        rewards += sum(player.curr_match_reward_log)
    player.model.save()
    return rewards


if __name__ == "__main__":
    # Here we will try to train the model, to block. We will use the LinesPlayer as a teacher, that means his reward
    # functions replaces, the normal reward function. Now our CNNPlayer will get rewards for blocking instead of
    # connecting.
    piskvorky = Piskvorky(GAME_SIZE)
    teacher = LinesPlayer(piskvorky.size, "teacher")
    model = CNNetwork_big(GAME_SIZE, str(185))
    player = CNNPlayer(GAME_SIZE, model=model, memory_size=1000, name=str(185), to_train=True, pretraining=True,
                       block_training=teacher.reward, double_dqn=True, minimax_prob=0, random_move_prob=0.99,
                       random_move_decrease=0.9997)
    counter = 0
    episodes = 100
    rewards_history = []
    for i in range(episodes):

        print("episode", counter)
        if counter == 6:
            player.pretraining = False
        rewards = train_blocking(piskvorky, player, teacher)

        rewards_history.append(rewards)
        counter += 1

    plot(range(len(rewards_history)), rewards_history)
    show()
