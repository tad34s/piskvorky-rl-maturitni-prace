from piskvorky import Piskvorky
from model import AlphaCNNetwork_preset
from trainer import Trainer
from variables import GAME_SIZE
from copy import deepcopy
import concurrent.futures


def learn(trainer,num_iters):
    for i in range(num_iters):

        print("{}/{}".format(i, num_iters))

        train_examples = []

        # get the training data paralely
        with concurrent.futures.ProcessPoolExecutor() as executor:
            if __name__ == "__main__":
                processes = []
                for eps in range(trainer.num_episodes):
                    print(f"episode: {eps}")
                    processes.append(executor.submit(trainer.exceute_episode))
                for process in processes:
                    train_examples.extend(process.result())

        train_examples = [list(x) for x in zip(*train_examples)]
        # prepare new model
        new_model = AlphaCNNetwork_preset(game.size, str(i), load=False)
        new_model.load_state_dict(trainer.model.state_dict())
        # train new model on the training data
        trainer.train(train_examples,new_model)
        new_model.save()
        is_better = trainer.model_eval(trainer.model, new_model, 2)
        if is_better:
            trainer.model = new_model
        else:
            print("not changing")


if __name__ == '__main__':
    # Train the Alpha Zero model, by self playing
    game = Piskvorky(GAME_SIZE)
    model = AlphaCNNetwork_preset(game.size, "0", load=False)

    trainer = Trainer(game, "trainer", model, restrict_movement=True, num_episodes=2)
    learn(trainer,num_iters=200)
