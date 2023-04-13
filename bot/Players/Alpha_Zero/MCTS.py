import numpy as np
import math
from copy import copy
from variables import VELIKOST
from piskvorky import index_to_xy, mask_invalid_moves
from bot.Players.Alpha_Zero.utils import reward_if_terminal,encode


def ucb_score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior, turn,waiting):
        self.visit_count = 0
        self.turn = turn
        self.waiting = waiting
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None


    def expanded(self):
        return len(self.children) > 0

    def take_action(self,action):
        x,y = index_to_xy(VELIKOST,action)
        next_state = copy(self.state)
        next_state[y,x] = self.turn
        return next_state

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, state, turn,waiting, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.turn = turn
        self.waiting = waiting
        self.state = state

        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob,turn=waiting,waiting=turn)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())


class MCTS:

    def __init__(self, game, model, num_simulations,restrict_movement = False):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.restrict_movement = restrict_movement
        self.total_mask_time = 0


    def run(self, model, state, turn,waiting):
        root = Node(0, turn,waiting)


        # EXPAND root
        action_probs, value = model.predict(encode(state,turn,waiting))
        # mask invalid moves
        mask = mask_invalid_moves(state,self.restrict_movement)
        action_probs *= mask

        action_probs /= np.sum(action_probs)
        root.expand(state, turn,waiting, action_probs)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]

            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state = parent.take_action(action)
            # Get the board from the perspective of the other player
            # The value of the new state from the perspective of the other player
            value = reward_if_terminal(next_state,index_to_xy(VELIKOST,action))
            if value is None:
                # If the game has not ended:
                # EXPAND
                action_probs, value = model.predict(encode(next_state,parent.waiting,parent.turn))
                mask = mask_invalid_moves(next_state, self.restrict_movement)
                action_probs *= mask
                action_probs /= np.sum(action_probs)
                node.expand(next_state,parent.waiting,parent.turn, action_probs)

            self.backpropagate(search_path, value, parent.waiting)

        return root

    def backpropagate(self, search_path, value, turn):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.turn == turn else -value
            node.visit_count += 1

