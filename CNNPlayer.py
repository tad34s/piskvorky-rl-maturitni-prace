import torch
import torch.nn as nn
import numpy as np
import math


class CNNetwork(torch.nn.Module):

    def __init__(self, size):
        self.size = size
        super(CNNetwork, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=2, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32),
                                    nn.MaxPool2d(kernel_size=2, stride=2)
                                    )

        self.linear1 = nn.Linear(32 * (math.ceil(self.size / 2) - 1) ** 2, 350)
        self.linear2 = nn.Linear(350, 225)
        self.linear3 = nn.Linear(225, self.size ** 2)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.block1(x)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        q_values = self.linear3(x)
        return q_values

    def probs(self, x):
        self.eval()
        with torch.no_grad():
            q_values = self.forward(x)
            return self.softmax(q_values)[0], q_values[0]


class CNNPLayer():

    def __init__(self, size, name, to_train=False, load=False):
        self.side = None
        self.size = size
        self.to_train = to_train
        self.name = "CNN " + name
        self.EMPTY = 0
        # model things
        self.model = CNNetwork(size=self.size)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.loss_fn = nn.MSELoss()
        if load:
            self.model.load_state_dict(torch.load(load))
            self.model.eval()

        # reinforcemnt learning params
        self.reward_discount = 0.95
        self.win_value = 1.0
        self.draw_value = 0.0
        self.loss_value = -1.0
        # exploitation vs exploration
        self.random_move_prob = 0.0
        self.random_move_decrease = 0.95
        # logs
        self.board_position_log = []
        self.action_log = []
        self.next_max_log = []
        self.values_log = []

    def encode(self, state):
        nparray = np.array([
            [(state == self.side).astype(int),
             (state == self.wait).astype(int),
             (state == self.EMPTY).astype(int)]
        ])
        output = torch.tensor(nparray, dtype=torch.float32)

        return output

    def newgame(self, side, other):
        self.side = side
        self.wait = other
        self.board_position_log = []
        self.action_log = []
        self.next_max_log = []
        self.values_log = []

    def calculate_targets(self):
        # pridat intermitten rewardy, minus kdyz nepritel connectne, plus kdyz ja connectnu
        game_length = len(self.action_log)
        targets = np.empty((len(self.action_log), len(self.values_log[0])), dtype=np.float32)

        for i in range(game_length):
            target = np.copy(self.values_log[i])

            target[self.action_log[i]] = self.reward_discount * self.next_max_log[i] #+ reward(self.board_position_log[i])
            targets[i] = target

        return torch.tensor(targets)

    def move(self, game):
        self.board_position_log.append(game.state.copy())
        input = self.encode(game.state)
        probs, q_values = self.model.probs(input)

        q_values = np.copy(q_values)
        for index, value in enumerate(q_values):
            if not game.islegal(game.indextoxy(index)):
                probs[index] = -1
            elif probs[index] < 0:
                probs[index] = 0.0

        if self.to_train and (np.random.rand(1) < self.random_move_prob):
            move = game.random_move()
        else:
            move = np.argmax(probs.numpy())
            move = game.indextoxy(move)

        if len(self.action_log) > 0:
            self.next_max_log.append(q_values[np.argmax(probs.numpy())])

        self.action_log.append(game.xytoindex(move))
        self.values_log.append(q_values)

        game.move(move)

        return move

    def train(self, vysledek, epochs):
        if vysledek == self.side:
            reward = self.win_value
        elif vysledek == "0":
            reward = self.draw_value
        else:
            reward = self.loss_value

        self.next_max_log.append(reward)

        y = self.calculate_targets()

        # We convert the input states we have recorded to feature vectors to feed into the training.

        X = torch.tensor([])
        for x in self.board_position_log:
            X = torch.cat([X, self.encode(x)])
        # X = torch.empty(size=(len(self.board_position_log),3,5,5))
        # for e, board in enumerate(self.board_position_log):

        # X[e] = self.encode(board)

        for epoch in range(epochs):
            # We run the training step with the recorded inputs and new Q value targets.
            y_hat = self.model(X)
            # y = y.view(-1, 1)

            loss = self.loss_fn(y_hat, y)

            # Backprop
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()


        self.random_move_prob *= self.random_move_decrease

    def save_model(self):
        torch.save(self.model.state_dict(), self.name)
