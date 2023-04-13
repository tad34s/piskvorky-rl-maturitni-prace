import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from bot.Players.Minimax_player import minimax, list_of_possible_moves
from copy import deepcopy
from bot.Player_abstract_class import Player
import time
from math import ceil
from piskvorky.functions import mask_invalid_moves

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


    def __len__(self):
        return len(self.states_log)

    def compute_state_target_matrix(self, state:GameState):
        # creates the target_matrix for the model to learn from
        matrix = torch.full([self.game_size**2],torch.nan)
        mask = torch.ones(self.game_size**2)
        for index_next_state in state.next_states:
            next_state = self.states_log[state.depth+1][index_next_state]
            index = (torch.eq(state.state[0,0].flatten(),next_state.state[0,0].flatten())==False).nonzero()
            mask[index] = 100
            matrix[index] = next_state.value()
        return matrix, mask

    def get_states_targets(self):
        st = time.time()
        targets = []
        all_game_states = [x for depth in self.states_log for x in depth]
        for state in all_game_states:
            target_matrix, mask = self.compute_state_target_matrix(state)
            targets.append(target_matrix)

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

    def __init__(self, size, name,model, to_train=False, pretraining=False, double_dqn=False,
                 random_move_prob=0.9999, random_move_decrease=0.9997, minimax_prob=0.2, restrict_movement= False):
        # self.last_seen = None

        super().__init__(name, to_train)
        self.side = None
        self.size = size
        self.name = "CNN proximal " + name + " " + str(size)
        self.restrict_movement = restrict_movement
        self.EMPTY = 0

        # model things
        self.model = model
        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=0.00025)
        self.loss_fn = CustomMSE()

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

    def new_game(self, side: int, other: int):
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
                loss = self.loss_fn(y_hat, y)
                print("loss",loss)
                # Backprop
                self.optim.zero_grad()
                loss.backward()
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


        q_values = np.copy(q_values)
        probs = np.copy(probs)
        mask = mask_invalid_moves(game.state,self.restrict_movement)
        probs *= mask
        rand_n = np.random.rand(1)
        if self.to_train and (rand_n < self.random_move_prob or self.pretraining):
            move = random.choice(list_of_possible_moves(game.state))
            rand_n2 = np.random.rand(1)
            if rand_n2 < self.minimax_prob and not self.pretraining:
                move = self.minimax_move(game)
        else:
            print("real move")
            move = np.argmax(probs)
            move = game.index_to_xy(move)

        game.move(move)
        return move

    def minimax_move(self, game):
        print("minimax move")
        move = minimax(game, 3, heuristic,restrict_movement=True)
        return move

    def save_model(self):
        torch.save(self.model.state_dict(), self.file)  # .replace(" ","-"))



class CustomMSE(nn.Module):
    def __init__(self):
        super(CustomMSE, self).__init__()

    def forward(self, output, target):
        indexes = torch.isfinite(target)

        output = output[torch.isfinite(target)]
        target = target[torch.isfinite(target)]

        cost_tensor = (output - target) ** 2
        #cost_tensor = torch.multiply(cost_tensor,masks)
        return torch.mean(cost_tensor)