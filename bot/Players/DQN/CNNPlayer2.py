import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from bot.Players.Minimax_player import minimax, list_of_possible_moves
from copy import deepcopy
from bot.Player_abstract_class import Player
from variables import EMPTY
from math import ceil
from piskvorky.functions import mask_invalid_moves


def heuristic(game, move):
    return 0


class GameState:
    """
    Individual state of the game
    """

    def __init__(self, state:torch.Tensor, final_reward:float, depth:int):
        self.state = state
        self.final_rewards = [final_reward]
        self.next_states = [] # indices of states that occurred after
        self.depth = depth

    def value(self)->float:
        """
        The average of all the rewards the state led to
        :return:
        """
        return sum(self.final_rewards) / len(self.final_rewards)


class CNNCache:
    """
    Cache that stores all the states the CNNPlayer will learn form.
    """

    def __init__(self, game_size:int):
        self.game_size = game_size
        self.states_log = self.generate_empty_log()

    def generate_empty_log(self)->list:
        """
        Generates list of empty list with the length of maximum number of moves a player can make.
        :return:
        """


        return [[] for x in range(ceil((self.game_size ** 2) / 2))]

    def add_states(self, states: list, final_reward: float)->None:
        """
        Adds states to the states_log
        Will put the nth state  into the nth list, this will sort the states, by the game length they occurred in.
        We do this, so we do not have to loop through the whole state_log when looking for matching states.
        :param states:
        :param final_reward:
        :return:
        """


        last_found_index = None

        for depth, new_state in enumerate(states):
            found = False
            for index_logged, state_logged in enumerate(self.states_log[depth]):
                if torch.all(torch.eq(new_state, state_logged.state) == True):  # if state is already logged
                    state_logged.final_rewards.append(final_reward)  # add his reward to cache
                    if depth != 0:  # if it is not the first state
                        # add its index to the next_state array of the state before
                        self.states_log[depth - 1][last_found_index].next_states.append(index_logged)

                    # remember where we put the game state, so we can add them to the next_state array next iteration
                    last_found_index = index_logged
                    found = True

            if not found:
                new_game_state = GameState(deepcopy(new_state), final_reward, depth)
                self.states_log[depth].append(new_game_state)
                if not last_found_index is None:
                    self.states_log[depth - 1][last_found_index].next_states.append(len(self.states_log[depth]) - 1)
                last_found_index = len(self.states_log[depth]) - 1

    def __len__(self)->int:
        return len(self.states_log)

    def compute_state_target_matrix(self, state: GameState)->torch.Tensor:
        """
        Creates the target_matrix for the model to learn from.
        See the documentation for further explanation.
        :param state:
        :return:
        """
        matrix = torch.full([self.game_size ** 2], torch.nan)
        for index_next_state in state.next_states:
            next_state = self.states_log[state.depth + 1][index_next_state]
            index = (torch.eq(state.state[0, 0].flatten(), next_state.state[0, 0].flatten()) == False).nonzero()
            matrix[index] = next_state.value()
        return matrix

    def get_states_targets(self)->tuple:
        """
        Return states and their target matrices
        :return: states,targets
        """
        targets = []
        all_game_states = [x for depth in self.states_log for x in depth]
        for state in all_game_states:
            target_matrix= self.compute_state_target_matrix(state)
            targets.append(target_matrix)

        states = [x.state for x in all_game_states]
        return states, targets

    def wipe(self):
        self.states_log = self.generate_empty_log()


class StateTargetValuesDataset(Dataset):

    def __init__(self, states:list, targets:list):
        self.states = states
        self.targets = targets
        if len(states) != len(targets):
            raise ValueError

    def __len__(self)->int:
        return len(self.states)

    def __getitem__(self, index:int):
        return self.states[index], self.targets[index]


class CNNPlayer_proximal(Player):
    """
    Player that uses Machine Learning to approximate the chance of him winning.
    """
    def __init__(self, size:int, name:str, model, to_train:bool=False,
                 random_move_prob:float=0.9999, random_move_decrease:float=0.9997, minimax_prob:float=0.2, restrict_movement:bool=False):
        """
        :param size: game size
        :param name:
        :param model:
        :param to_train: whether we want to train it or not
        :param random_move_prob: chance to make a random move
        :param random_move_decrease: the amount it decreases after each training
        :param minimax_prob: chance to make a move based on the minimax algorithm instead of a random move
        :param restrict_movement:  search only spaces near symbols
        """
        super().__init__(name, to_train)
        self.size = size
        self.name = "CNN proximal " + name + " " + str(size)
        self.restrict_movement = restrict_movement
        self.EMPTY = EMPTY

        # model things
        self.model = model
        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=0.00025)
        self.loss_fn = CustomMSE()

        self.old_network = None

        # reinforcement learning params
        self.reward_discount = 0.99
        self.win_value = 1.0
        self.draw_value = 0.0
        self.loss_value = -1.0
        # exploitation vs exploration
        self.random_move_prob = random_move_prob
        self.random_move_decrease = random_move_decrease
        self.minimax_prob = minimax_prob
        if not self.to_train:
            self.random_move_prob = 0

        # logs
        self.match_state_log = []
        self.cache = CNNCache(size)

    def encode(self, state:torch.Tensor)->torch.Tensor:
        """
        Encode game for model
        :param state:
        :return:
        """
        nparray = np.array([
            [(state == self.side).astype(int),
             (state == self.opponent).astype(int)]
        ])
        output = torch.tensor(nparray, dtype=torch.float32)
        return output

    def new_game(self, side, other):
        self.side = side
        self.opponent = other
        self.match_state_log = []

    def game_end(self, result):
        if result == self.side:
            reward = self.win_value
        elif result == self.opponent:
            reward = self.loss_value
        else:
            reward = self.draw_value
        if self.to_train:
            self.cache.add_states(self.match_state_log, reward)

    def train(self, epochs:int)->None:

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
                print("loss", loss)
                # Backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
        if self.to_train:
            self.random_move_prob *= self.random_move_decrease

        self.cache.wipe()

    def move(self, game, enemy_move):

        input = self.encode(game.state)
        if self.to_train:
            self.match_state_log.append(input)

        probs, q_values = self.model.probs(input)

        probs = np.copy(probs)
        mask = mask_invalid_moves(game.state, self.restrict_movement)
        probs *= mask
        rand_n = np.random.rand(1)
        if self.to_train and rand_n < self.random_move_prob:
            # we make a random move
            move = random.choice(list_of_possible_moves(game.state))
            rand_n2 = np.random.rand(1)
            if rand_n2 < self.minimax_prob:
                # we make minimax move
                move = self.minimax_move(game)
        else:
            print("real move")
            # we make the move by the model
            move = np.argmax(probs)
            move = game.index_to_xy(move)

        game.move(move)
        return move

    def minimax_move(self, game)->tuple:
        print("minimax move")
        move = minimax(game, 3, heuristic, restrict_movement=True)
        return move


class CustomMSE(nn.Module):
    """
    A custom loss what filters all the nans
    """

    def __init__(self):
        super(CustomMSE, self).__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        output = output[torch.isfinite(target)]
        target = target[torch.isfinite(target)]

        cost_tensor = (output - target) ** 2
        return torch.mean(cost_tensor)
