import numpy as np
import torch
import torch.nn as nn


class CNNetwork_preset(torch.nn.Module):

    def __init__(self, size, name, load=False):
        self.size = size
        super(CNNetwork_preset, self).__init__()

        self.build_graph()
        self.name = "CNN " + name + " " + str(size)

        self.file = "..\\NNs_preset\\" + self.name.replace(" ", "-") + ".nn"

        if load:
            if isinstance(load, str):
                self.load_state_dict(torch.load(load))
            else:
                self.load_state_dict(torch.load(self.file))
            self.eval()

    def build_graph(self):
        self.conv_activation = nn.ReLU()
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(-1)

        self.conv_5 = nn.Conv2d(2, 8, kernel_size=5, stride=1, padding=2)
        self.conv_5.weight = torch.nn.Parameter(self.create_weights(5))
        self.conv_5.bias = torch.nn.Parameter(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))

        self.conv_friendly = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                                           self.conv_activation)
        self.conv_enemy = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                                        self.conv_activation)

        self.conv_layer = nn.Sequential(nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
                                        self.conv_activation)
        self.conv_layer2 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                                         self.conv_activation)
        self.linear = nn.Linear(16 * (self.size ** 2), self.size ** 2)

    def forward(self, x):
        self.eval()
        with torch.no_grad():
            x = self.conv_5(x)

        friendly, enemy = torch.split(x, 4, 1)
        friendly = self.conv_friendly(friendly)
        enemy = self.conv_enemy(enemy)

        x = torch.cat((friendly, enemy), dim=1)
        x = self.conv_layer(x)
        x = self.conv_layer2(x)
        x = x.flatten(start_dim=1)
        x = self.activation(x)
        q_values = self.linear(x)
        return q_values

    def probs(self, x):
        self.eval()
        with torch.no_grad():
            q_values = self.forward(x)
            return self.softmax(q_values)[0], q_values[0]

    def create_weights(self, kernel_size):
        # creates 8 kernel weights first 4 is 2 diagonals, row and column the second 4 are exact same but checks the enemy
        if not 2 < kernel_size < 6:
            raise Exception
        left_diagonal = [[[1 if x == y else 0 for x in range(kernel_size)] for y in range(kernel_size)],
                         np.zeros((kernel_size, kernel_size))]
        right_diagonal = [[[1 if x == y else 0 for x in range(kernel_size)] for y in reversed(range(kernel_size))],
                          np.zeros((kernel_size, kernel_size))]
        column = [[[1 if y == 2 else 0 for x in range(kernel_size)] for y in reversed(range(kernel_size))],
                  np.zeros((kernel_size, kernel_size))]
        row = [[[1 if x == 2 else 0 for x in range(kernel_size)] for y in reversed(range(kernel_size))],
               np.zeros((kernel_size, kernel_size))]
        friendly_kernels = [left_diagonal, right_diagonal, column, row]
        enemy_kernels = [reversed(left_diagonal), reversed(right_diagonal), reversed(column), reversed(row)]
        return torch.tensor([friendly_kernels, enemy_kernels], dtype=torch.float32).view([8, 2, 5, 5])

    def save(self):
        torch.save(self.state_dict(), self.file)


class CNNetwork_big(torch.nn.Module):

    def __init__(self, size, name, load=False):
        super(CNNetwork_big, self).__init__()

        self.size = size
        self.build_graph()
        self.name = "CNN big " + name + " " + str(size)
        self.file = ".\\NNs\\" + self.name.replace(" ", "-") + ".nn"

        if load:
            if isinstance(load, str):
                self.load_state_dict(torch.load(load))
            else:
                self.load_state_dict(torch.load(self.file))
            self.eval()

    def build_graph(self):

        self.block1 = nn.Sequential(nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    )
        self.block2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    )

        self.linear1 = nn.Linear(64 * (self.size ** 2), 16 * (self.size ** 2))
        self.linear2 = nn.Linear(16 * (self.size ** 2), 8 * (self.size ** 2))
        self.linear3 = nn.Linear(8 * (self.size ** 2), self.size ** 2)
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

    def save(self):
        torch.save(self.state_dict(), self.file)
