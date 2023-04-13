import numpy as np
import torch
from MCTS import MCTS
from piskvorky import index_to_xy,play_game
from utils import reward_if_terminal,encode
from variables import X,O
from bot.Players.Alpha_Zero.Alpha_player import AlphaPlayer
from copy import deepcopy
from torch.utils.data import Dataset,DataLoader
import concurrent.futures

class StateRewardProbsDataset(Dataset):

    def __init__(self, states,values_target, probs_target):
        self.states = states
        self.values_target = values_target
        self.probs_target = probs_target
        if len(states) != len(probs_target):
            raise ValueError

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.values_target[index],self.probs_target[index]

class Trainer:

    def __init__(self, game,name,model, num_iters = 500,num_simulations = 500,num_epochs = 3,num_iters_per_example = 20, num_episodes = 50, restrict_movement = False):
        self.game = game
        self.num_simulations = num_simulations
        self.num_episodes = num_episodes
        self.num_iters = num_iters
        self.num_epochs = num_epochs
        self.num_iters_per_example = num_iters_per_example

        self.model = model
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.restrict_movement = restrict_movement
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

            train_examples = [list(x) for x in zip(*train_examples)]

            self.train(train_examples)
            is_better = self.model_eval(old_model,self.model,2)
            if not is_better:
                print("not changing")
                self.model = old_model
            self.model.save()

    def train(self, examples):

        dataset = StateRewardProbsDataset(examples[0], examples[1], examples[2])
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


        for epoch in range(self.num_epochs):

            for batch in dataloader:
                X, y_action, y_value = batch
                y_value = y_value.to(torch.float32)
                X = X.view((-1, 2, 8, 8))
                y_hat_actions, y_hat_value = self.model(X)
                loss_actions = self.loss_pi(y_hat_actions, y_action)
                loss_values = self.loss_v(y_hat_value, y_value)
                total_loss = loss_actions + loss_values
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

           # print(out_pi[0].detach())
           # print(target_pis[0])

    def loss_pi(self, outputs,targets):
        fn  = torch.nn.KLDivLoss(reduction="batchmean")
        loss = fn(outputs.log(),targets)
        return loss

    def loss_v(self,outputs,targets):
        loss = torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
        return loss

    def model_eval(self,old_model,new_model,n_matches):
        old_player = AlphaPlayer(self.game.size,old_model,"old",self.num_simulations,restrict_movement=True)
        new_player = AlphaPlayer(self.game.size,new_model,"new",self.num_simulations,restrict_movement=True)

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
        print(who_won.count(new_player.name),who_won.count(old_player.name))
        if who_won.count(new_player.name)>=who_won.count(old_player.name):
            return True
