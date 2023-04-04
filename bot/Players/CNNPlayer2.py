import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from bot.Players.Minimax_player import minimax, list_of_possible_moves
from copy import copy,deepcopy
from bot.Networks import CNNetwork_preset, CNNetwork_big
from bot.Players.Player_abstract_class import Player


def heuristic(game, move):
    return 0


class CNNCache:

    def __init__(self, game_size):
        self.states_log = []
        self.rewards_log = []
        self.values = None
        self.game_size = game_size

    def add_states(self, states: list, final_reward: float):
        for index_old, state in enumerate(self.states_log):
            for index_new, new_state in enumerate(states):
                #print(new_state, state)
                if torch.all(torch.eq(new_state, state) == True):
                    self.rewards_log[index_old].append(final_reward)
                    states.pop(index_new)

        for new_state in states:
            self.states_log.append(new_state)
            self.rewards_log.append([final_reward])

    def __len__(self):
        return len(self.states_log)

    def get_all_statevalues(self):
        # computes the value of the state as the average of all its observed final rewards
        values = [sum(x) / len(x) for x in self.rewards_log]
        return values

    @staticmethod
    def get_possible_states(state):
        output = []
        #print(state.size())
        for x, row in enumerate(state[0][0]):
            for y, space in enumerate(row):
                if space.item() <= 0.001:
                    if state[0,1, y, x] == 0:
                        new_state = deepcopy(state)
                        new_state[0,0, y, x] = 1
                        output.append((new_state, (y, x)))
        #print(output)
        return output

    def compute_state_target_matrix(self, state):
        # creates the target_matrix for the model to learn from
        target_matrix = np.empty((self.game_size, self.game_size))
        target_matrix[:] = np.nan

        next_states = self.get_possible_states(state)
        for index_old, old_state in enumerate(self.states_log):
            for next_state, move in next_states:
                if torch.all(torch.eq(next_state, old_state) == True):
                    approximation = self.values[index_old]
                    target_matrix[move[1], move[0]] = approximation

        return target_matrix

    def get_states_targets(self):
        targets = []
        self.values = self.get_all_statevalues()
        for state in self.states_log:
            targets.append(self.compute_state_target_matrix(state))

        return self.states_log, targets


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


class CNNPlayer_proximal(Player):

    def __init__(self, size, name, preset=False, to_train=False, load=False, pretraining=False, double_dqn=False,
                 random_move_prob=0.9999, random_move_decrease=0.9997, minimax_prob=0.2):
        # self.last_seen = None

        super().__init__(name, to_train)
        self.side = None
        self.size = size
        self.name = "CNN proximal " + name + " " + str(size)

        self.EMPTY = 0

        # model things
        if preset:
            self.model = CNNetwork_preset(size=self.size)
            self.file = ".\\NNs_preset\\" + self.name.replace(" ", "-") + ".nn"
        else:
            self.model = CNNetwork_big(size=self.size)
            self.file = ".\\NNs\\" + self.name.replace(" ", "-") + ".nn"

        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=0.00025)
        self.loss_fn = nn.MSELoss()
        if load:
            if isinstance(load, str):
                self.model.load_state_dict(torch.load(load))
            else:
                self.model.load_state_dict(torch.load(self.file))
            self.model.eval()

        self.old_network = None
        self.double_dqn = double_dqn

        # reinforcement learning params
        self.reward_discount = 0.99
        self.win_value = 1.0
        self.draw_value = 0.0
        self.loss_value = -1.0
        # exploitation vs exploration
        self.random_move_increase = 1.1  # if player lost try to explore more
        self.random_move_prob = random_move_prob
        self.random_move_decrease = random_move_decrease
        self.minimax_prob = minimax_prob
        self.pretraining = pretraining
        if not self.to_train:
            self.random_move_prob = 0

        # logs
        self.match_state_log = []
        self.cache = CNNCache(size)

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
        self.match_state_log = []

    def game_end(self, result):
        if result == self.side:
            reward = self.win_value
        elif result == self.wait:
            reward = self.loss_value
        else:
            reward = self.draw_value

        self.cache.add_states(self.match_state_log, reward)

    def train(self, epochs):

        X, y = self.cache.get_states_targets()
        dataset = StateTargetValuesDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(epochs):

            for batch in dataloader:
                # We run the training step with the recorded inputs and new Q value targets.
                X, y = batch
                X = X.view((-1, 2, 8, 8))
                y_hat = self.model(X)
                # y = y.view(-1, 1)

                loss = self.loss_fn(y_hat, y)
                #print(loss)
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

        if self.to_train and not self.pretraining:
            self.random_move_prob *= self.random_move_decrease

    def move(self, game, enemy_move):

        input = self.encode(game.state)

        if self.to_train:
            self.match_state_log.append(input)

        if self.double_dqn and self.old_network:
            probs, q_values = self.old_network.probs(input)
        else:
            probs, q_values = self.model.probs(input)
        print(q_values)
        q_values = np.copy(q_values)
        for index, value in enumerate(q_values):
            if not game.is_legal(game.index_to_xy(index)):
                probs[index] = -1
            elif probs[index] < 0:
                probs[index] = 0.0

        rand_n = np.random.rand(1)
        if self.to_train and (rand_n < self.random_move_prob or self.pretraining):
            move = random.choice(list_of_possible_moves(game))
            rand_n2 = np.random.rand(1)
            if rand_n2 < self.minimax_prob and not self.pretraining:
                move = self.minimax_move(game)
        else:
            print("real move")
            move = np.argmax(probs.numpy())
            move = game.index_to_xy(move)

        game.move(move)
        return move

    def minimax_move(self, game):
        print("minimax move")
        move = minimax(game, 3, heuristic)
        return move

    def save_model(self):
        torch.save(self.model.state_dict(), self.file)  # .replace(" ","-"))
