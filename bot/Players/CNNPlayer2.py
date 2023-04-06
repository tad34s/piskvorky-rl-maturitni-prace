import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from bot.Players.Minimax_player import minimax, list_of_possible_moves
from copy import copy,deepcopy
from bot.Networks import CNNetwork_preset, CNNetwork_big
from bot.Players.Player_abstract_class import Player
import time
from math import ceil

def heuristic(game, move):
    return 0

class GameState:

    def __init__(self,state,final_reward,depth):
        self.state = state
        self.final_rewards = [final_reward]
        self.next_states = []
        self.depth = depth

    def value(self):
        return sum(self.final_rewards)/len(self.final_rewards)


class CNNCache:

    def __init__(self, game_size):
        self.game_size = game_size
        self.states_log = self.generate_empty_log()

    def generate_empty_log(self):
        return [ [] for x in range(ceil((self.game_size**2)/2))]

    def add_states(self, states: list, final_reward: float):
        st = time.time()
        last_found_index = None

        for depth, new_state in enumerate(states):
            found = False
            for index_logged, state_logged in enumerate(self.states_log[depth]):
                if torch.all(torch.eq(new_state, state_logged.state)==True): # if state is already logged
                    state_logged.final_rewards.append(final_reward) # add this reward to cache
                    if depth != 0:# if its not the first state
                        self.states_log[depth-1][last_found_index].next_states.append(index_logged) # add its index to the next_state array of the state before
                    last_found_index = index_logged # remeber where we put the game state so we can add the to the next_state array next iteration
                    found = True

            if not found:
                new_game_state = GameState(deepcopy(new_state), final_reward,depth)
                self.states_log[depth].append(new_game_state)
                if not last_found_index is None:
                    self.states_log[depth-1][last_found_index].next_states.append(len(self.states_log[depth])-1)
                last_found_index = len(self.states_log[depth])-1

        et = time.time()
        elapsed_time = et - st
        print('Add_states: ', elapsed_time, 'seconds')
    def __len__(self):
        return len(self.states_log)

    def compute_state_target_matrix(self, state:GameState):
        # creates the target_matrix for the model to learn from
        target_matrix = torch.full((self.game_size,self.game_size),torch.nan,dtype=torch.float32)

        for index_next_state in state.next_states:
            next_state = self.states_log[state.depth+1][index_next_state]
            index = (torch.eq(state.state[0,0],next_state.state[0,0])==False).nonzero()
            target_matrix[index[0][0],index[0][1]] = next_state.value()
            print(target_matrix)

        return target_matrix.flatten()

    def get_states_targets(self):
        st = time.time()
        targets = []
        all_game_states = [x for depth in self.states_log for x in depth]
        for state in all_game_states:
            targets.append(self.compute_state_target_matrix(state))
        et = time.time()
        elapsed_time = et - st
        print('Get states: ', elapsed_time, 'seconds')
        states = [x.state for x in all_game_states]
        return states, targets

    def wipe(self):
        self.states_log = self.generate_empty_log()



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
        self.loss_fn = CustomMSE()
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
        if self.to_train:
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
                loss = my_loss(y_hat, y)
                print("loss",loss)
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    print(param.grad)
                    if param.grad is None:
                        continue
                    torch.nan_to_num_(param.grad)
                self.optim.step()

        if self.to_train and not self.pretraining:
            self.random_move_prob *= self.random_move_decrease
        self.cache.wipe()


    def move(self, game, enemy_move):

        input = self.encode(game.state)

        if self.to_train:
            self.match_state_log.append(input)

        if self.double_dqn and self.old_network:
            probs, q_values = self.old_network.probs(input)
        else:
            probs, q_values = self.model.probs(input)

        print("q_values\n",q_values)
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

class CustomMSE(nn.Module):
    def __init__(self):
        super(CustomMSE, self).__init__()

    def forward(self, output, target):
        return torch.nanmean((output - target)**2)

def my_loss(output, target):
    return torch.nanmean((output - target)**2)
