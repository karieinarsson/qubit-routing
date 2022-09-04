'''
This is a OpenAI gym environment made for qubit routing
'''

from typing import List, Tuple
import torch as th
import math
import copy
from torch import Tensor as torch
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import networkx as nx
from numpy.linalg import matrix_power

# types
Matrix = List[List[int]]


def main():
    '''
    Main function to run tests
    '''
    edge_index = th.tensor(
        [[0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8],
         [1, 3, 0, 2, 4, 1, 5, 0, 4, 6, 1, 3, 5, 7, 2, 4, 8, 3, 7, 4, 6, 8, 7, 8]],
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
        # vars
        self.depth = depth
        self.n_qubits = n_qubits
        self.max_swaps = np.floor(self.n_qubits/2)
        self.timeout = timeout

        # links
        self.edge_index = edge_index

        # adj matrix and dist matrix
        self.adj = self.__coo_to_adj(self.edge_index, self.n_qubits)
        #self.dist_mtx = self.__get_distance_matrix(self.adj)
    
        #array of possible actions
        self.actions = self._get_actions()

        # Number of actions we can take
        self.action_space = Discrete(len(self.actions))

        self.state = th.zeros((self.depth, self.n_qubits, 1), dtype=th.float)
        for i in range(self.depth):
            self.state[i] = th.tensor([[2], [1], [0], [3], [4], [6], [5], [7], [8]], dtype=th.float)
    
        #self.state = th.tensor([[2], [1], [0], [3], [4], [6], [5], [7], [8]], dtype=th.float)

        print(x)
        print(self.adj)

        assert False
        self.observation_space = Box(low=0, high=self.max_swaps,
                                     shape=(depth, n_qubits, 1, ),
                                     dtype=np.float64)
        
        self.layers = 0
        self.code_length = None

        # reset environment
        self.reset()

    def step(self, action: int) -> Tuple[List[int], int, bool, 'info']:
        """
        :param: action: the int reprensentation of an action

        :return: returns state after action, reward of the action,
            bool that indicates if the episode if done and info
        """
        action_matrix = self.actions[action]
        self.state = np.transpose(self.state)
        self.state = np.matmul(self.state, action_matrix)
        self.code = np.matmul(self.code, action_matrix)
    
        self.max_episode_steps -= 1

        "reward_func needs to be rewritten"
        reward = self.reward_func()

        if reward >= -1:
            # remove the executed slice and add a new slice from code at the end
            self.state[0], self.code = self.code[:1], self.code[1:]
            self.state = np.roll(self.state, -1, axis=0)
            self.code_length -= 1


        done = self.code_length <= 0 or self.max_episode_steps <= 0 
        
        info = {}

        self.state = np.transpose(self.state)

        return self.state, reward, done, info

    def render(self, mode="human", render_list=None) -> bool:
        pass

    def reset(self):
        """
        :return: returns the state
        """
        self.max_layers = self.depth

        self.state = self._make_state()
        
        self.code_length = len(self.state)

        self.max_episode_steps = self.timeout

        return self.state

# Init helper functions

    def __gate_to_permutation_matrix(self,
                                     permutaion_matrix,
                                     gate: List[int]
                                     ):
        """
        :param: permutaion_matrix: permutation matrix
        :param: gate: a list of the start qubit and target qubit of the gate
        :return: returns a permutaion matrix with the gate link included
        """
        permutaion_matrix = copy.deepcopy(permutaion_matrix)
        q_0, q_1 = gate

        permutaion_matrix[q_0][q_0] = 0
        permutaion_matrix[q_1][q_1] = 0
        permutaion_matrix[q_0][q_1] = 1
        permutaion_matrix[q_1][q_0] = 1

        return permutaion_matrix

    def _get_actions(self,
                               iterations: int = None,
                               architecture: Matrix = None,
                               permutaion_matrix = None
                               ):
        """
        :param: iterations: (used for recursion) The current iteration of the recurtion
        :param: architecture: (used for recurision) a modified architecture that removes links from and to used qubits
        :param: permutaion_matrix: (used for recursion) a permutaion matrix that could include swaps
        :return: List of permutation matrices corresponding to all possible actions for the current architecture
        """
        
        if iterations is None:
            iterations = math.floor(self.n_qubits/2)

        if architecture is None:
            architecture = self.adj

        if permutaion_matrix is None:
            permutaion_matrix = np.identity(self.n_qubits, dtype=int)

        possible_actions = [permutaion_matrix]

        for row in range(self.n_qubits):
            for col in range(self.n_qubits):
                if architecture[row][col] == 0:
                    continue
                modified_architecture = copy.deepcopy(architecture)
                modified_architecture[row] = 0
                modified_architecture[col] = 0
                modified_architecture[:, col] = 0
                modified_architecture[:, row] = 0
                modified_permutaion_matrix = self.__gate_to_permutation_matrix(
                    permutaion_matrix, [row, col])
                for action in self._get_actions(iterations-1, modified_architecture, modified_permutaion_matrix):
                    possible_actions.append(action)

        possible_actions = [tuple(action) for action in possible_actions]
        possible_actions = np.unique(possible_actions, axis=0)

        return possible_actions

    def __coo_to_adj(self, edge_index, n_qubits):
        """"""
        assert edge_index.shape[0] == 2, "Not COO notation"
        adj = th.zeros((n_qubits, n_qubits))
        for q_0, q_1 in edge_index.t():
            adj[q_0][q_1] = 1
        return adj

    def __get_distance_matrix(self, adj):
        """The distance matrix indicates how many swaps are needed to go from
        qubit[i] to qubit[j].
        Args:
            adj (np.asarray): The adjacency matrix of the hardware coupling graph.
        Returns:
            distance_matrix (np.asarray): The distance matrix
        """
        assert nx.is_connected(nx.from_numpy_matrix(adj)), "graph is not connected"
        assert adj.shape[0] == adj.shape[1]
        distance_matrix = adj > 0  # set all values to 1
        i = 2
        while np.count_nonzero(
            distance_matrix == 0
        ):  # until all entries are occupied
            mat = matrix_power(distance_matrix, i)
            mat = mat > 0  # set all values to 1
            mask = (
                distance_matrix == 0
            )  # do not overwrite already occupied entries!
            mat = i * np.multiply(mat, mask)
            distance_matrix = mat + distance_matrix
            i += 1
        return torch.tensor(distance_matrix)

# Code generation

    def _make_state_slice(self):
        """
        :return: Flattened timestep layer of random gates
        """
        max_gates = math.floor(self.n_qubits/2)
        state_slice = np.zeros(self.n_qubits)
        for i in range(1, np.random.choice(range(2, max_gates+2))):
            state_slice[i-1] = i
            state_slice[i-1+max_gates] = i
        np.random.shuffle(state_slice)
        return state_slice

    def _make_state(self):
        """
        :return: State composed of random timestep layers with random gates
        """
        #state = np.zeros((self.max_layers, self.rows, self.cols))
        state = []
        for i in range(self.max_layers):
            state.append(self._make_state_slice())
        return state
# reward function

    def reward_func(self, state, action: int) -> int:
        """
        :param: state: A flattened state of gates
        :param: action: Action

        :return: The immediate reward
        """
        return 1

    def is_executable_state(self, state) -> bool:
        """
        :param: state: A flattened state of gates

        :return: Bool which is True if all gates are executable in the first timestep layer
        """
        return True


if __name__ == '__main__':
    main()
