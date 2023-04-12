import torch.optim
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from bot.Player_abstract_class import Player
from bot.alpha_gomoku.model import AlphaCNNetwork_preset
from bot.alpha_gomoku.MCTS import MCTS
from piskvorky import index_to_xy
from copy import copy
import time
class StateData():

    def __init__(self,state,action_probs):
        self.state = state
        self.action_probs = torch.tensor(action_probs / np.sum(action_probs),dtype=torch.float32)
        self.final_reward = None
class AlphaMemory():

    def __init__(self):
        self.states_log = []

    def add_states(self,states:list):
        for state in states:
            self.states_log.append(state)

    def wipe(self):
        self.states_log = []

    def get_training_examples(self):
        states = []
        action_probs = []
        rewards = []
        for data in self.states_log:
            states.append(data.state)
            action_probs.append(data.action_probs)
            rewards.append(data.final_reward)

        return states,action_probs,rewards




class StateActionprobsRewardDataset(Dataset):
    def __init__(self, states, action_probs,reward):
        self.states = states

        self.action_probs = action_probs
        self.reward = reward
        if len(states) != len(action_probs):
            raise ValueError

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.action_probs[index], self.reward[index]

class AplhaPlayer(Player):

    def __init__(self,size,model,name,num_simulations,to_train=False, restrict_movement = False,):
        super().__init__(name, to_train)
        self.size = size
        self.name = "Alpha Zero " + name + " " + str(size)
        self.restrict_movement = restrict_movement
        self.num_simulations = num_simulations
        self.model = model

        self.optim = torch.optim.Adam(self.model.parameters(),lr=0.1)
        self.loss_actions = self.loss_pi
        self.loss_value = torch.nn.MSELoss()

        self.memory = AlphaMemory()
        self.current_match = []

    def encode(self, state):

        nparray = np.array([
            [(state == self.side).astype(int),
             (state == self.other).astype(int)]
        ])
        output = torch.tensor(nparray, dtype=torch.float32)
        return output

    def new_game(self,side,other) ->None:
        self.side = side
        self.other = other
        self.current_match = []
    def move(self,game,enemy_move) ->tuple:
        st = time.time()
        mcts = MCTS(game, self.model, self.num_simulations,restrict_movement=self.restrict_movement)
        action_probs = [0 for _ in range(game.size ** 2)]
        root = mcts.run(self.model,game.state,turn=game.turn,waiting = game.waiting)
        for k, v in root.children.items():
            action_probs[k] = v.visit_count

        self.current_match.append(StateData(self.encode(game.state), action_probs))
        action = root.select_action(temperature=0)
        move = index_to_xy(self.size, action)
        game.move(move)
        et = time.time()
        elapsed_time = et - st
        print('One move: ', elapsed_time, 'seconds')
        return move

    def game_end(self,result) ->None:
        if result == self.side:
            reward = 1.0
        elif result == self.other:
            reward = -1.0
        else:
            reward = 0

        for state in self.current_match:
            state.final_reward = torch.tensor([reward],dtype=torch.float32)
        self.memory.add_states(copy(self.current_match))

    def train(self,epochs):

        X, action_probs, rewards = self.memory.get_training_examples()
        dataset = StateActionprobsRewardDataset(X,action_probs,rewards)
        dataloader = DataLoader(dataset, batch_size=64, shuffle= True)
        for epoch in range(epochs):
            for batch in dataloader:
                X, y_action, y_value = batch
                y_value = y_value.to(torch.float32)
                X = X.view((-1, 2, 8, 8))
                y_hat_actions, y_hat_value = self.model(X)
                loss_actions = self.loss_actions(y_hat_actions,y_action)
                loss_values = self.loss_value(y_hat_value,y_value)
                total_loss = loss_actions + loss_values
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

        self.memory.wipe()


