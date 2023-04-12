import numpy as np
import torch
from MCTS import MCTS
from piskvorky import index_to_xy,play_game
from utils import reward_if_terminal,encode
from variables import X,O
from random import shuffle
import os
from bot.alpha_gomoku.model import AlphaCNNetwork_preset
from bot.alpha_gomoku.Alpha_player import AplhaPlayer
from copy import deepcopy
class Trainer:

    def __init__(self, game,name, num_iters = 500,num_simulations = 500,num_epochs = 3,num_iters_per_example = 20, num_episodes = 50, restrict_movement = False, load = False):
        self.game = game
        self.name = name
        self.num_simulations = num_simulations
        self.num_episodes = num_episodes
        self.num_iters = num_iters
        self.num_epochs = num_epochs
        self.num_iters_per_example = num_iters_per_example

        self.restrict_movement = restrict_movement
        self.file = "NNs_preset\\" + self.name.replace(" ", "-") + ".nn"
        if load:
            if isinstance(load, str):
                self.model.load_state_dict(torch.load(load))
            else:
                self.model.load_state_dict(torch.load(self.file))
            self.model.eval()
        else:
            self.model = AlphaCNNetwork_preset(game.size)
        self.mcts = MCTS(self.game, self.model, num_simulations)
    def exceute_episode(self):

        train_examples = []
        self.game.reset()
        n_moves = 1
        temperature = 1
        while True:
            board = self.game.state

            self.mcts = MCTS(self.game, self.model, self.num_simulations,restrict_movement = self.restrict_movement)
            root = self.mcts.run(self.model, board,turn= self.game.turn,waiting=self.game.waiting)
            if n_moves == 10:
                temperature = 0

            action_probs = [0 for _ in range(self.game.size**2)]
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((encode(self.game.state,self.game.turn,self.game.waiting),self.game.turn, action_probs))
            action = root.select_action(temperature=temperature)
            xy = index_to_xy(self.game.size, action)
            self.game.move(xy)
            n_moves += 1
            reward = reward_if_terminal(self.game.state,xy)

            if not reward is None:
                ret = []
                for hist_state,hist_current_player ,hist_action_probs in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != self.game.turn))))

                return ret

    def learn(self):
        for i in range(self.num_iters):

            print("{}/{}".format(i, self.num_iters))

            train_examples = []
            old_model = deepcopy(self.model)
            for eps in range(self.num_episodes):
                print(f"episode: {eps}")
                iteration_train_examples = self.exceute_episode()
                train_examples.extend(iteration_train_examples)

            shuffle(train_examples)
            self.train(train_examples)
            is_better = self.model_eval(old_model,self.model,2)
            if not is_better:
                print("not changing")
                self.model = old_model
            self.save_checkpoint(folder=".")

    def train(self, examples):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        pi_losses = []
        v_losses = []

        for epoch in range(self.num_epochs):
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / 64):
                sample_ids = np.random.randint(len(examples), size=256)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.stack(boards).view(-1,2,8,8)
                target_pis = torch.tensor(pis,dtype=torch.float32)
                target_vs = torch.tensor(vs,dtype=torch.float64)


                # compute output
                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))
            print("Examples:")
           # print(out_pi[0].detach())
           # print(target_pis[0])

    def loss_pi(self, targets, outputs):
        fn  = torch.nn.KLDivLoss(reduction="batchmean")
        loss = fn(outputs.log(),targets)
        return loss

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return loss

    def save_checkpoint(self,folder):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, self.file)
        torch.save(self.model.state_dict(), filepath)

    def model_eval(self,old_model,new_model,n_matches):
        old_player = AplhaPlayer(self.game.size,old_model,"old",self.num_simulations,)
        new_player = AplhaPlayer(self.game.size,new_model,"new",self.num_simulations,)

        starter = old_player
        waiting = new_player
        who_won = []
        self.game.reset()
        for match in range(n_matches):
            result = play_game(self.game, starter, waiting)
            if result == X:
                who_won.append(starter.name)
            elif result == O:
                who_won.append(waiting.name)
            else:
                who_won.append(0)
            starter,waiting = waiting,starter
            # starter,waiting = waiting,starter
        if who_won.count(new_player.name)>=who_won.count(old_player.name):
            return True

