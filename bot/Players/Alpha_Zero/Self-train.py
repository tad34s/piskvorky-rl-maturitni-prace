from piskvorky import Piskvorky
from model import AlphaCNNetwork_preset
from trainer import Trainer
from variables import VELIKOST
from copy import deepcopy
import concurrent.futures

def learn(trainer):
    for i in range(trainer.num_iters):

        print("{}/{}".format(i, trainer.num_iters))

        train_examples = []
        old_model = deepcopy(trainer.model)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            if __name__ == "__main__":
                processes = []
                for eps in range(trainer.num_episodes):
                    print(f"episode: {eps}")
                    processes.append(executor.submit(trainer.exceute_episode))
                for process in processes:

                    train_examples.extend(process.result())

        train_examples = [list(x) for x in zip(*train_examples)]

        trainer.train(train_examples)
        is_better = trainer.model_eval(old_model, trainer.model, 2)
        if not is_better:
            print("not changing")
            trainer.model = old_model
        trainer.model.save()

if __name__ == '__main__':
    game = Piskvorky(VELIKOST)
    model = AlphaCNNetwork_preset(game.size, "123", load=False)


    trainer = Trainer(game,"123", model,restrict_movement=True,num_episodes=10)
    learn(trainer)