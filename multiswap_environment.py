'''
This is a OpenAI gym environment made for qubit routing
'''

from typing import List, Tuple
import torch as th
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

        # array of possible actions
        self.actions = self.__get_actions()

        # Number of actions we can take
        self.action_space = Discrete(len(self.actions))

        self.state = th.zeros((self.depth, self.n_qubits, 1), dtype=th.float)
        for i in range(self.depth):
            self.state[i] = th.tensor([[2], [1], [0], [3], [4], [6], [5], [7], [8]], dtype=th.float)
    
        #self.state = th.tensor([[2], [1], [0], [3], [4], [6], [5], [7], [8]], dtype=th.float)

        self.observation_space = Box(low=0, high=self.max_swaps,
                                     shape=(depth, n_qubits, 1, ),
                                     dtype=np.float64)

        # reset environment
        self.reset()

    def step(self, action: int) -> Tuple[List[int], int, bool, 'info']:
        """
        :param: action: the int reprensentation of an action

        :return: returns state after action, reward of the action,
            bool that indicates if the episode if done and info
        """
        return self.state, 1, False, {} 

    def render(self, mode="human", render_list=None) -> bool:
        pass

    def reset(self):
        """
        :return: returns the state
        """
        return self.state

# Init helper functions

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

    def __get_actions(self):
        """"""
        return np.array([np.identity(self.n_qubits)])

# Pruning

    def pruning(self, obs):
        return range(len(self.actions))

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
