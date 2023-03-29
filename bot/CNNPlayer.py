import random
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from bot.mmplayer import minimax, listofpossiblemoves
from copy import deepcopy, copy
from torch import Tensor


def heuristic(game,move):
    return 0

class CNNetwork(torch.nn.Module):

    def __init__(self, size):
        self.size = size
        super(CNNetwork, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    )
        self.block2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    )

        self.linear1 = nn.Linear(4096, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, self.size ** 2)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        q_values = self.linear3(x)
        return q_values

    def probs(self, x):
        self.eval()
        with torch.no_grad():
            q_values = self.forward(x)
            return self.softmax(q_values)[0], q_values[0]


class Match():

    def __init__(self, board_log, move_log, rewards_log, final_reward):
        self.last_seen = None
        self.encoded_board_log = board_log
        self.move_log = move_log
        self.rewards_log = rewards_log
        self.final_reward = final_reward
        self.q_values_log = []
        self.next_max_log = []

    def generate_values(self, model):

        self.q_values_log = []
        self.next_max_log = []
        for e, state in enumerate(self.encoded_board_log):
            probs, q_values = model.probs(state)
            q_values = np.copy(q_values)
            self.q_values_log.append(q_values)
            if e != 0:
                self.next_max_log.append(q_values[self.move_log[e]])
        self.next_max_log.append(self.final_reward)


class CNNMemory():

    def __init__(self, size: int):
        self.size = size
        self.games_won=[]
        self.games_drawn=[]
        self.games_lost=[]

    def add_match(self, match: Match):

        if match.final_reward > 0:
            self.add_match_to_list(self.games_won,match)
        elif match.final_reward < 0:
            self.add_match_to_list(self.games_lost,match)
        else:
            self.add_match_to_list(self.games_drawn,match)

    def add_match_to_list(self,games:list,match:Match):
        if len(games) >= self.size:
            games.pop(0)
        games.append(match)

    def __len__(self):
        return len(self.games_won)+len(self.games_lost) +len(self.games_drawn)

    def get_random_matches(self, n: int) -> list:
        if n > len(self):
            n = len(self)
        output = []
        size = n//3
        output.extend(self.sample(self.games_won, k=size))
        output.extend(self.sample(self.games_lost, k=size))
        output.extend(self.sample(self.games_drawn, k=size))
        return output

    def sample(self,games:list,k:int):
        if k > len(games):
            output = games
        else:
            output = random.sample(games, k=k)
        return output

class StateTargetValuesDataset(Dataset):

    def __init__(self, states, targets):
        self.states = states
        self.targets = targets
        if len(states) != len(targets):
            raise ValueError

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.targets[index]


class CNNPLayer():

    def __init__(self, size, name,memory_size, to_train=False, load=False, block_training = False, pretraining = False):
        self.last_seen = None
        self.side = None
        self.size = size
        self.to_train = to_train
        self.name = "CNN " + name + " " + str(size)
        self.file = "NNs\\" + self.name.replace(" ","-") + ".nn"
        self.EMPTY = 0
        self.block_training = block_training
        # model things
        self.model = CNNetwork(size=self.size)
        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=0.00025)
        self.loss_fn = nn.MSELoss()
        if load:
            if type(load) == type(""):
                self.model.load_state_dict(torch.load(load))
            else:
                self.model.load_state_dict(torch.load(self.file))
            self.model.eval()

        # reinforcemnt learning params
        self.reward_discount = 0.99
        self.win_value = 1.0
        self.draw_value = 0.0
        self.loss_value = -1.0
        # exploitation vs exploration
        self.random_move_increase=1.1# if player lost try to explore mroe
        self.random_move_prob = 0.9999
        self.random_move_decrease = 0.99997
        self.pretraining = pretraining
        if not self.to_train:
            self.random_move_prob = 0

        # logs
        self.curr_match_board_log = []
        self.curr_match_move_log = []
        self.curr_match_next_max_log = []  # the q value of the next move
        self.curr_match_values_log = []  # log of the q values generated for the respective baord
        self.curr_match_reward_log = []  # log of the rewards generated by the moves
        self.memory = CNNMemory(memory_size)
        print(self.memory.size)

    def encode(self, state):
        nparray = np.array([
            [(state == self.side).astype(int),
             (state == self.wait).astype(int)]
        ])
        output = torch.tensor(nparray, dtype=torch.float32)
        return output

    def new_game(self, side, other):
        print(self.random_move_prob)
        self.side = side
        self.wait = other
        self.curr_match_board_log = []  # not encoded history of board states
        self.curr_match_move_log = []
        self.curr_match_next_max_log = []  # the q value of the next move
        self.curr_match_values_log = []  # log of the q values generated for the respective baord
        self.curr_match_reward_log = []  # log of the rewards generated by the moves

    def calculate_targets(self, match):

        game_length = len(match.move_log)
        targets = []
        print(match.rewards_log)

        for i in range(game_length):
            target = np.copy(match.q_values_log[i])  # the q values of every move available on board
            old = target[match.move_log[i]]
            # change the value of the move made to the q value of the state it led to
            target[match.move_log[i]] = self.reward_discount * match.next_max_log[i] + match.rewards_log[i]
            #print(target[match.move_log[i]])
            targets.append(torch.tensor(target))
            print(old,target[match.move_log[i]])
        return targets

    def move(self, game, enemy_move):

        if self.to_train:
            self.curr_match_board_log.append(game.state.copy())
            # calculate punishemnt
            if not self.block_training:
                if self.curr_match_reward_log:
                    self.curr_match_reward_log[-1] -= self.reward(game, enemy_move)

        input = self.encode(game.state)
        probs, q_values = self.model.probs(input)
        print(q_values)
        q_values = np.copy(q_values)
        for index, value in enumerate(q_values):
            if not game.islegal(game.indextoxy(index)):
                probs[index] = -1
            elif probs[index] < 0:
                probs[index] = 0.0

        rand_n = np.random.rand(1)
        if self.to_train and (rand_n < self.random_move_prob or self.pretraining):
            move = random.choice(listofpossiblemoves(game))
        else:
            move = np.argmax(probs.numpy())
            move = game.indextoxy(move)

        game.move(move)


        # add reward
        if self.to_train:
            if len(self.curr_match_move_log) > 0:
                self.curr_match_next_max_log.append(q_values[np.argmax(probs.numpy())])
            self.curr_match_move_log.append(game.xytoindex(move))
            self.curr_match_values_log.append(q_values)

            if self.block_training:
                reward = self.block_training(game,move)
                self.curr_match_reward_log.append(reward)
            else:
                self.curr_match_reward_log.append(self.reward(game, move))


        return move

    def reward(self, game, move):

        odmena = 0
        points = [game.leftdiagpoints(move[0], move[1]), game.rightdiagpoints(move[0], move[1]),
                  game.rowpoints(move[0], move[1]), game.columnpoints(move[0], move[1])]
        points.sort(reverse=True)
        for point in points:
            if odmena == 0:
                odmena += point ** 2 / 10
            else:
                odmena += point ** 2 / 40

        return odmena

    def train(self, vysledek, epochs, n_recalls=0):
        if vysledek == self.side:
            reward = self.win_value
        elif vysledek == "0":
            reward = self.draw_value
        else:
            reward = self.loss_value

        if n_recalls < 0:
            n_recalls = 0

        self.curr_match_next_max_log.append(reward)
        self.curr_match_reward_log[-1] = 0

        encoded_board = [self.encode(x) for x in self.curr_match_board_log]
        this_match = Match(encoded_board, self.curr_match_move_log, self.curr_match_reward_log, reward)
        # if lost last game and now won, i want the nn to really remember how to win


        #self.train_on_matches([this_match], epochs)
        self.memory.add_match(deepcopy(this_match))

        if not self.pretraining:
            games_from_memory = self.memory.get_random_matches(n_recalls)

            self.train_on_matches(games_from_memory, epochs=15)

        if self.to_train and not self.pretraining:
            self.random_move_prob *= self.random_move_decrease

    def minimax_move(self, game):
        print("minimax move")
        move = minimax(game, 3,heuristic)
        return move

    def train_on_matches(self, matches: list, epochs):

        

        # trains on the game loaded in memory
        if not matches:
            return
        y = []
        X = []
        for match in matches:
            match.generate_values(self.model)
            targets = self.calculate_targets(match)
            for target in targets:
                y.append(target)
            # y = torch.cat([y, self.calculate_targets(match)])
            for board in match.encoded_board_log:
                X.append(board)
            # X = torch.cat([X, self.encode(board)])
        dataset = StateTargetValuesDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        paramdata = []
        for epoch in range(epochs):

            for batch in dataloader:
                # We run the training step with the recorded inputs and new Q value targets.
                X, y = batch
                X = X.view((-1, 2, 8, 8))
                y_hat = self.model(X)
                # y = y.view(-1, 1)

                loss = self.loss_fn(y_hat, y)

                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def save_model(self):
        torch.save(self.model.state_dict(), self.file )#.replace(" ","-"))

