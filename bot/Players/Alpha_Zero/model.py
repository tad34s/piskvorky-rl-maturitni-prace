import numpy as np
import torch
import torch.nn as nn


class AlphaCNNetwork_preset(torch.nn.Module):

    """
    Special Neural Network for Alpha Zero, has two heads: value head and action head. Similarly to CNN_preset uses
    preset kernel weights in the first convolutional layer.
    """
    def __init__(self, size:int, name:str, load=False):
        self.size = size
        super(AlphaCNNetwork_preset, self).__init__()
        self.name = "Alpha Zero " + name + " " + str(size)
        self.build_graph()
        # default file
        self.file = "NNs_preset\\" + self.name.replace(" ", "-") + ".nn"

        if load:
            if isinstance(load, str):
                # if specified location load the location
                self.load_state_dict(torch.load(load))
            else:
                # if simply True go for the default location
                self.load_state_dict(torch.load(self.file))
            self.eval()

    def build_graph(self)->None:
        self.conv_activation = nn.ReLU()

        self.conv_5 = nn.Conv2d(2, 8, kernel_size=5, stride=1, padding=2)
        self.conv_5.weight = torch.nn.Parameter(self.create_weights(5))
        self.conv_5.bias = torch.nn.Parameter(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32))

        self.conv_friendly = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                                           self.conv_activation)
        self.conv_enemy = nn.Sequential(nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                            self.conv_activation)

        self.conv_layer = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                        self.conv_activation)
        self.conv_layer2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                                         self.conv_activation)

        self.conv_action = nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1)

        self.value_head = nn.Sequential(nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
                                        nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
                                        nn.MaxPool2d(8, 1),
                                        nn.Tanh()
                                        )
        self.action_head = nn.Sequential(nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
                                         nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
                                         )
        self.action_activation = nn.Softmax(-1)

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

        value = self.value_head(x)
        value = value.view(-1, 1)

        action_probs = self.action_head(x)
        action_probs = action_probs.flatten(start_dim=1)
        action_probs = self.action_activation(action_probs)

        return action_probs, value

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            probs, value = self.forward(x)
            return probs.numpy()[0], value.numpy()[0]

    def create_weights(self, kernel_size: int) -> torch.Tensor:
        """
        Creates weights to be preset for the first convolutional layer.
        Generates 8 kernel weights first 4 is 2 diagonals, row and column the second 4 are exact same but checks the enemy.
        :param kernel_size:
        :return:
        """
        #
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
