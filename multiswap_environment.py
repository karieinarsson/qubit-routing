'''
This is a OpenAI gym environment made for qubit routing
'''

import math
from typing import List, Tuple
import copy
import pygame
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from stable_baselines3.common.env_checker import check_env
import torch as th

# types
Matrix = List[List[int]]

TimestepLayer = List[List[int]]
FlattenedTimeStepLayer = List[int]
State = List[TimestepLayer]
FlattenedState = List[FlattenedTimeStepLayer]

PermutationMatrix = List[List[int]]

# pygame constants
PG_WIDTH = 100
PG_HEIGHT = 100
X_START = PG_WIDTH*0.6
Y_START = PG_HEIGHT*0.6
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
PURPLE = (255, 0, 255)
YELLOW = (255, 255, 0)
BROWN = (165, 42, 42)
PINK = (255, 20, 147)
GREY = (50.2, 50.2, 50.2)
PURPLE = (50.2, 0, 50.2)
LIME = (191, 255, 0)


def main():
    '''
    Main function to run tests
    '''
    edge_index = th.tensor(
            [[0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 6, 7],
             [1, 3, 2, 4, 5, 4, 6, 5, 7, 8, 7, 8]], 
            dtype=th.long
        )
    SwapEnvironment(5, edge_index, 9)


class SwapEnvironment(Env):
    '''
    Our environment
    '''

    def __init__(self,
                 depth: int,
                 edge_index: List[List[int]],
                 n_qubits: int,
                 timeout: int = 200
                 ) -> None:
        """
        :param: depth: depth of code
        :param: edge_index: list of links between qubits in architecture
        :param: n_qubits: number of qubits in architecture
        :param: timeout: (optional) max steps per episode before fail

        :return: returns nothing
        """
        self.depth = depth
        self.n_qubits = n_qubits
        self.edge_index = edge_index
        self.arch = self.__edge_index_arch(self.edge_index, self.n_qubits)

        self.max_swaps = np.floor(self.n_qubits/2)
        self.timeout = timeout
        
        # array of possible actions
        # self.possible_actions = self.__get_possible_actions()
        
        # Number of actions we can take
        self.action_space = Discrete(len(self.edge_index.t()))
        self.observation_space = Box(low=0, high=self.max_swaps,
                                     shape=(1, depth, n_qubits, n_qubits, ),
                                     dtype=np.float64)

        self.state = None
        self.code = None

        # reset environment
        self.reset()

        # pygame screen initialization
        self.screen = None
        self.isopen = True

    def step(self, action: int) -> Tuple[List[int], int, bool, 'info']:
        """
        :param: action: the int reprensentation of an action

        :return: returns state after action, reward of the action,
            bool that indicates if the episode if done and info
        """
        self.state = self.state.reshape((self.depth, self.rows*self.cols))
        self.max_episode_steps -= 1
        swap_matrix = self.possible_actions[action]
        self.state = np.matmul(self.state, swap_matrix)
        self.code = np.matmul(self.code, swap_matrix)
        # Rewards
        reward = self.reward_func(self.state, action)

        if reward >= -1:
            # remove the executed slice and add a new slice from code at the end
            self.state[0], self.code = self.code[:1], self.code[1:]
            self.state = np.roll(self.state, -1, axis=0)
            self.max_layers -= 1

        done = self.max_episode_steps <= 0 or self.max_layers <= 0

        info = {}
        self.state = self.state.reshape(
            (1, self.depth, self.rows, self.cols, ))
        return self.state, reward, done, info

    def render(self, mode="human", render_list=None) -> bool:
        pass

    def reset(self,
              code: State = None
              ) -> State:
        """
        :param: code: (optional) code to reset the state to

        :return: returns the state
        """
        self.max_layers = self.depth
        if code is None:
            self.code = self.processing(self.__make_code().reshape(
                (self.depth, self.rows * self.cols)), preprocessing=True)
        else:
            self.code = self.processing(code.reshape(
                (self.depth, self.rows * self.cols)), preprocessing=True)

        self.code = np.pad(self.code, ((0, self.depth), (0, 0)))
        self.state, self.code = self.code[:self.depth], self.code[self.depth:]
        self.state = self.state.reshape(
            (1, self.depth, self.rows, self.cols, ))

        self.max_episode_steps = self.timeout
        return self.state

    def __edge_index_arch(self, edge_index, n_qubits):
        arch = th.zeros((n_qubits, n_qubits))
        for x, y in edge_index.t():
            arch[x][y], arch[y][x] = 1, 1
        return arch

# reward function

    def reward_func(self, state: FlattenedState, action: int) -> int:
        """
        :param: state: A flattened state of gates
        :param: action: Action

        :return: The immediate reward
        """
        if self.is_executable_state(state):
            parallell_actions = self.prune_action_space(state)
            if action in parallell_actions:
                return 0
            return -1
        return -2

    def is_executable_state(self,
                            state: FlattenedState
                            ) -> bool:
        """
        :param: state: A flattened state of gates

        :return: Bool which is True if all gates are executable in the first timestep layer
        """
        connectivity_matrix = self.__timestep_layer_to_connectivity_matrix(
            state[0])
        if (connectivity_matrix & self.architecture == connectivity_matrix).all():
            return True
        return False

# render functions

    def action_render(self, action_matrix: PermutationMatrix) -> List[Tuple[int, int]]:
        """
        Input:
            - action_matrix: PermutationMatrix corresponding to an action

        Output: List of tuples of ints describing between what qubits SWAP-gates are placed
        """
        action_matrix = action_matrix.tolist()
        action_tuples = []
        used_nodes = []
        for i, _ in enumerate(action_matrix):
            if i not in used_nodes:
                idx = action_matrix[i].index(1)
                used_nodes.append(idx)
                if idx != i:
                    action_tuples.append(tuple((i, idx)))
        return action_tuples


if __name__ == '__main__':
    main()
