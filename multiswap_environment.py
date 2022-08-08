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
    env = SwapEnvironment(1, 2, 2)
    env.tester()


class SwapEnvironment(Env):
    '''
    Our environment
    '''

    def __init__(self,
                 depth: int,
                 rows: int,
                 cols: int,
                 timeout: int = 200,
                 n_qubits: int = None,
                 links: List[Tuple[int]] = None
                 ) -> None:
        """
        :param: depth: depth of code
        :param: rows: number of rows in architecture
        :param: cols: number of cols in architecture
        :param: timeout: (optional) max steps per episode before fail
        :param: n_qubits: (optional) number of qubits in architecture
        :param: links: (optional) list of links between qubits in architecture

        :return: returns nothing
        """
        self.depth = depth
        self.rows = rows
        self.cols = cols

        self.n_qubits = self.rows * self.cols if n_qubits is None else n_qubits
        self.links = self.__gen_links() if links is None else links
        self.architecture = self.__get_adjencency_matrix(self.links)

        self.max_swaps_per_time_step = np.floor(self.rows * self.cols/2)
        self.timeout = timeout
        # array of possible actions
        self.possible_actions = self.__get_possible_actions()
        # Number of actions we can take
        self.action_space = Discrete(len(self.possible_actions))
        self.observation_space = Box(low=0, high=np.floor(self.rows * self.cols / 2),
                                     shape=(1, depth, rows, cols, ), dtype=np.uint8)

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
        if render_list is None:
            return
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (PG_WIDTH*self.cols, PG_HEIGHT*self.rows))

        num_font = pygame.font.SysFont(None, 25)
        img0 = num_font.render('0', True, RED)
        img1 = num_font.render('1', True, RED)
        img2 = num_font.render('2', True, RED)
        img3 = num_font.render('3', True, RED)
        img4 = num_font.render('4', True, RED)
        img5 = num_font.render('5', True, RED)
        img6 = num_font.render('6', True, RED)
        img7 = num_font.render('7', True, RED)
        img8 = num_font.render('8', True, RED)
        img9 = num_font.render('9', True, RED)
        s_img = num_font.render('S', True, BLACK)

        dict = {
            0: BLACK,
            1: GREEN,
            2: BLUE,
            3: PURPLE,
            4: YELLOW,
            5: BROWN,
            6: PINK,
            7: GREY,
            8: PURPLE,
            9: LIME
        }

        num_dict = {
            0: img0,
            1: img1,
            2: img2,
            3: img3,
            4: img4,
            5: img5,
            6: img6,
            7: img7,
            8: img8,
            9: img9
        }

        num_matrix = []
        tmp = 0
        for _ in range(self.rows):
            tmpm = []
            for _ in range(self.cols):
                tmpm.append(tmp)
                tmp += 1
            num_matrix.append(tmpm)

        surface = pygame.Surface(self.screen.get_size())

        pygame.draw.rect(surface, WHITE, surface.get_rect())

        # row / col %

        for j in range(1, self.cols+1):
            for i in range(1, self.rows+1):
                pygame.draw.circle(
                    surface, BLACK, ((X_START*j), (Y_START*i)), 20)
                if j < self.rows:
                    pygame.draw.line(
                        surface, BLACK, ((X_START*j), (Y_START*i)), ((X_START*(j+1)), ((Y_START*i))), 4)
                if i < self.cols:
                    pygame.draw.line(
                        surface, BLACK, ((X_START*j), (Y_START*i)), ((X_START*j), ((Y_START*(i+1)))), 4)
                pygame.draw.circle(surface, dict.get(
                    render_list[0][i-1][j-1]), ((X_START*j), (Y_START*i)), 15)
                surface.blit(num_dict.get(
                    num_matrix[i-1][j-1]), ((X_START*j)-5, (Y_START*i)-5))

        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

        index = 0
        running = True

        while running:
            ev = pygame.event.get()

            for event in ev:

                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if index % 2 == 0:
                        pygame.draw.rect(surface, WHITE, surface.get_rect())
                        for j in range(1, self.cols+1):
                            for i in range(1, self.rows+1):
                                pygame.draw.circle(
                                    surface, BLACK, ((X_START*j), (Y_START*i)), 20)
                                if j < self.rows:
                                    pygame.draw.line(
                                        surface, BLACK, ((X_START*j), (Y_START*i)), ((X_START*(j+1)), ((Y_START*i))), 4)
                                if i < self.cols:
                                    pygame.draw.line(
                                        surface, BLACK, ((X_START*j), (Y_START*i)), ((X_START*j), ((Y_START*(i+1)))), 4)
                                    surface.blit(num_dict.get(
                                        num_matrix[i-1][j-1]), ((X_START*j)-5, (Y_START*i)-5))

                    if event.key == pygame.K_n:
                        # next one
                        if index == len(render_list)-1:
                            print("At last obs")
                        else:
                            index += 1

                            if type(render_list[index]) is list:
                                pygame.draw.rect(
                                    surface, WHITE, surface.get_rect())
                                for j in range(1, self.cols+1):
                                    for i in range(1, self.rows+1):
                                        pygame.draw.circle(
                                            surface, BLACK, ((X_START*j), (Y_START*i)), 20)
                                        if j < self.rows:
                                            pygame.draw.line(
                                                surface, BLACK, ((X_START*j), (Y_START*i)), ((X_START*(j+1)), ((Y_START*i))), 4)
                                        if i < self.cols:
                                            pygame.draw.line(
                                                surface, BLACK, ((X_START*j), (Y_START*i)), ((X_START*j), ((Y_START*(i+1)))), 4)
                                        pygame.draw.circle(surface, dict.get(
                                            render_list[index][i-1][j-1]), ((X_START*j), (Y_START*i)), 15)
                                        surface.blit(num_dict.get(
                                            num_matrix[i-1][j-1]), ((X_START*j)-5, (Y_START*i)-5))
                            else:
                                swap_matrix = self.possible_actions[render_list[index]]

                                for j in range(1, self.cols+1):
                                    for i in range(1, self.rows+1):
                                        pygame.draw.circle(surface, dict.get(
                                            render_list[index-1][i-1][j-1]), ((X_START*j), (Y_START*i)), 15)
                                        surface.blit(num_dict.get(
                                            num_matrix[i-1][j-1]), ((X_START*j)-5, (Y_START*i)-5))
                                tuple_list = self.action_render(swap_matrix)

                                num_matrix = np.matmul(swap_matrix, np.asarray(num_matrix).reshape(
                                    self.rows*self.cols)).reshape((self.rows, self.cols)).tolist()
                                for i in range(len(num_matrix)):
                                    num_matrix[i] = list(
                                        map(int, num_matrix[i]))

                                for t in tuple_list:
                                    r0 = t[0]//self.cols
                                    c0 = t[0] % self.cols
                                    r1 = t[1]//self.cols
                                    c1 = t[1] % self.cols
                                    x0 = X_START*(c0+1)
                                    y0 = Y_START*(r0+1)
                                    x1 = X_START*(c1+1)
                                    y1 = Y_START*(r1+1)
                                    x = x1+((x0-x1)/2)
                                    y = y1+((y0-y1)/2)
                                    pygame.draw.rect(surface, CYAN, pygame.Rect(
                                        (x-10, y-10), (20, 20)))
                                    surface.blit(s_img, (x-6, y-8))

                            self.screen.blit(surface, (0, 0))
                            pygame.display.flip()
                    if event.key == pygame.K_b:
                        # back one
                        if index == 0:
                            print("At first obs")
                        else:
                            index -= 1

                            if type(render_list[index]) is list:
                                pygame.draw.rect(
                                    surface, WHITE, surface.get_rect())

                                for j in range(1, self.cols+1):
                                    for i in range(1, self.rows+1):
                                        pygame.draw.circle(
                                            surface, BLACK, ((X_START*j), (Y_START*i)), 20)
                                        if j < self.rows:
                                            pygame.draw.line(
                                                surface, BLACK, ((X_START*j), (Y_START*i)), ((X_START*(j+1)), ((Y_START*i))), 4)
                                        if i < self.cols:
                                            pygame.draw.line(
                                                surface, BLACK, ((X_START*j), (Y_START*i)), ((X_START*j), ((Y_START*(i+1)))), 4)
                                        pygame.draw.circle(surface, dict.get(
                                            render_list[index][i-1][j-1]), ((X_START*j), (Y_START*i)), 15)
                                        surface.blit(num_dict.get(
                                            num_matrix[i-1][j-1]), ((X_START*j)-5, (Y_START*i)-5))

                            else:
                                swap_matrix = self.possible_actions[render_list[index]]

                                for j in range(1, self.cols+1):
                                    for i in range(1, self.rows+1):
                                        pygame.draw.circle(surface, dict.get(
                                            render_list[index-1][i-1][j-1]), ((X_START*j), (Y_START*i)), 15)
                                        surface.blit(num_dict.get(
                                            num_matrix[i-1][j-1]), ((X_START*j)-5, (Y_START*i)-5))
                                tuple_list = self.action_render(swap_matrix)

                                num_matrix = np.matmul(np.asarray(num_matrix).reshape(
                                    self.rows*self.cols), swap_matrix.T).reshape((self.rows, self.cols)).tolist()
                                for i in range(len(num_matrix)):
                                    num_matrix[i] = list(
                                        map(int, num_matrix[i]))

                                for t in tuple_list:
                                    r0 = t[0]//self.cols
                                    c0 = t[0] % self.cols
                                    r1 = t[1]//self.cols
                                    c1 = t[1] % self.cols
                                    x0 = X_START*(c0+1)
                                    y0 = Y_START*(r0+1)
                                    x1 = X_START*(c1+1)
                                    y1 = Y_START*(r1+1)
                                    x = x1+((x0-x1)/2)
                                    y = y1+((y0-y1)/2)
                                    pygame.draw.rect(surface, CYAN, pygame.Rect(
                                        (x-10, y-10), (20, 20)))
                                    surface.blit(s_img, (x-6, y-8))

                            self.screen.blit(surface, (0, 0))
                            pygame.display.flip()

        self.screen.blit(surface, (0, 0))

        pygame.event.pump()
        pygame.display.flip()
        return self.isopen

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

# Tester functions

    def tester(self) -> None:
        """
        Tests functions on the class and prints True if it worked and False if not

        :return: returns nothing
        """
        env = swap_environment(10, 2, 2)
        print("Test1:", (np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0], [
              0, 0, 0, 0]]) == env.__timestep_layer_to_connectivity_matrix(np.array([1, 2, 2, 1]))).all())
        print("Test2:", (np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [
              0, 0, 0, 0]]) == env.__timestep_layer_to_connectivity_matrix(np.array([0, 1, 1, 0]))).all())
        print("Test3:", (np.array([[0, 1, 1, 0], [0, 0, 0, 1], [
              0, 0, 0, 1], [0, 0, 0, 0]]) == env.architecture).all())
        print("Test4:", env.is_executable_state(np.array([[2, 1, 2, 1]])))
        print("Test5:", not env.is_executable_state(np.array([[0, 1, 1, 0]])))
        print("Test6:", len(env.possible_actions) == 7)
        env2 = swap_environment(1, 3, 3)
        print("Test7:", len(env2.possible_actions) == 131)
        mtx = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0]])
        par = env2.prune_action_space(mtx)
        success = True
        print("Test8: ", end="")
        for a in par:
            if not (np.matmul(mtx, env2.possible_actions[a]) == mtx).all():
                print("\n\tFalse")
                print("Action:", a)
                print(env.possible_actions[a])
                print("From:", mtx, "->", np.matmul(mtx,
                      env.possible_actions[a]), "not", mtx, "->", mtx, "\n")
                success = False
        if success:
            print(True)
        success = True
        mtx = np.array([[1, 0, 1, 0, 2, 2, 0, 0, 0]])
        non_par = env2.prune_action_space(mtx)
        print("Test9: ", end="")
        for a in non_par:
            if (np.matmul(mtx, env2.possible_actions[a]) == mtx).all():
                print("\n\tFalse")
                print("Action:", a)
                print(env.possible_actions[a])
                success = True
        if success:
            print(True)
        check_env(env, warn=False)
        print("Environment check:", True)

# architecture subfunctions

    def __gen_links(self) -> List[List[int]]:
        """
        Generates links for a square like architecture with links up, down, left and right

        :return: returns list of links
        """
        quantum_arch = np.arange(self.rows*self.cols)
        links = []
        for node in quantum_arch:
            if node + 1 < quantum_arch.size and node % self.cols != self.cols-1:
                links.append((node, node+1))
            if node + self.cols < quantum_arch.size:
                links.append((node, node+self.cols))
        return np.array(links)

    def __get_adjencency_matrix(self,
                                links: List[List[int]]
                                ) -> Matrix:
        """
        :param: links: list of links

        :return: returns an adjencency matrix
        """
        adjencency_matrix = np.zeros((self.n_qubits, self.n_qubits), dtype=int)
        for q_0, q_1 in links:
            adjencency_matrix[q_0][q_1] = 1
        return adjencency_matrix

    def __timestep_layer_to_connectivity_matrix(self,
                                                timestep_layer: FlattenedTimeStepLayer
                                                ) -> Matrix:
        """
        :param: timestep_layer: flattned timestep layer

        :return: returns connectivity matrix for the gates in the timestep layer
        """
        connectivity_matrix = np.zeros(
            (self.n_qubits, self.n_qubits), dtype=int)
        for gate in np.arange(1, int(np.max(timestep_layer))+1):
            q_0, q_1 = np.where(timestep_layer == gate)[0]
            connectivity_matrix[q_0][q_1] = 1
        return connectivity_matrix

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

    def __gate_to_permutation_matrix(self,
                                     permutaion_matrix: PermutationMatrix,
                                     gate: List[int]
                                     ) -> PermutationMatrix:
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

    def __get_possible_actions(self,
                               iterations: int = None,
                               architecture: Matrix = None,
                               permutaion_matrix: PermutationMatrix = None
                               ) -> List[PermutationMatrix]:
        """
        :param: iterations: (used for recursion) The current iteration of the recurtion
        :param: architecture: (used for recurision) a modified architecture that removes links from and to used qubits
        :param: permutaion_matrix: (used for recursion) a permutaion matrix that could include swaps

        :return: List of permutation matrices corresponding to all possible actions for the current architecture
        """
        if iterations is None:
            iterations = self.max_swaps_per_time_step

        if architecture is None:
            architecture = self.architecture

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
                for action in self.__get_possible_actions(iterations-1, modified_architecture, modified_permutaion_matrix):
                    possible_actions.append(action)

        possible_actions = [tuple(action) for action in possible_actions]
        possible_actions = np.unique(possible_actions, axis=0)

        return possible_actions

# code generation

    def __make_state_slice(self) -> FlattenedTimeStepLayer:
        """
        :return: Flattened timestep layer of random gates
        """
        max_gates = math.floor(self.rows*self.cols/2)
        state_slice = np.zeros(self.rows*self.cols)
        for i in range(1, np.random.choice(range(2, max_gates+2))):
            state_slice[i-1] = i
            state_slice[i-1+max_gates] = i
        np.random.shuffle(state_slice)
        return state_slice

    def __make_code(self) -> State:
        """
        :return: State composed of random timestep layers with random gates
        """
        state = np.zeros((self.max_layers, self.rows, self.cols))
        for i, _ in enumerate(state):
            state[i] = self.__make_state_slice().reshape(
                (self.rows, self.cols))
        return state

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

# pruning

    def __get_parallell_actions(self, state: FlattenedState) -> List[int]:
        """
        :param: state: flattened state
        :return: Returns list of actions
        """
        inverse_identety = np.ones(
            (self.possible_actions.shape), dtype=int)-np.identity(self.n_qubits, dtype=int)
        action_connectivity = inverse_identety & self.possible_actions

        used = np.where(state[0] > 0)
        used_matrix = self.__get_used_matrix(used)
        parallell_map = np.sum(
            used_matrix & action_connectivity, axis=(1, 2)) == 0
        return np.where(parallell_map)[0]

    def __get_pruned_action_space(self, state: FlattenedState) -> List[int]:
        """
        :param: state: flattened state
        :return: Returns list of actions
        """
        inverse_identety = np.ones(
            (self.possible_actions.shape), dtype=int)-np.identity(self.n_qubits, dtype=int)
        action_connectivity = inverse_identety & self.possible_actions

        inverse_architecture = np.ones(
            (self.possible_actions.shape), dtype=int) - self.architecture
        pruned_filter = inverse_architecture & self.__timestep_layer_to_connectivity_matrix(
            state[0])

        not_linked_gates = np.squeeze(
            np.column_stack(np.where(pruned_filter[0] == 1)))
        pruned_select_matrix = self.__get_used_matrix(not_linked_gates)
        pruned_map = np.sum(pruned_select_matrix &
                            action_connectivity, axis=(1, 2)) != 0

        return np.where(pruned_map)[0]

    def prune_action_space(self, state: FlattenedState) -> List[int]:
        """
        :param: state: flattened state
        :return: Returns list of actions
        """
        if self.is_executable_state(state):
            return self.__get_parallell_actions(state)
        return self.__get_pruned_action_space(state)

    def __get_used_matrix(self, used: List[int]):
        """
        :param: used: list of used qubits
        :return: returns connectivity map with used qubits
        """
        used_matrix = np.zeros((self.n_qubits, self.n_qubits), dtype=int)
        for qubit in used:
            used_matrix[qubit] = 1
            used_matrix[:, qubit] = 1
        return used_matrix

# processing

    def processing(self, state: State, preprocessing: bool = True) -> State:
        """
        :param: state: A flattened state of gates
        :param: preprocessing: bool that tells if this is used as preprocessing or postprocessing

        :return: Flattened compressed state
        """
        gates = []
        for mtx in state:
            used = []
            for val in mtx:
                if val != 0 and val not in used:
                    used.append(val)
                    gates.append(
                        (val/abs(val), np.array([i for i, x in enumerate(mtx) if x == val])))

        return_state = []
        c_gate = 0
        swap_gate = 0
        layer = np.zeros(self.rows*self.cols)
        for val, pos in gates:
            if layer[pos[0]] != 0 or layer[pos[1]] != 0:
                return_state.append(layer)
                layer = np.zeros(self.rows*self.cols)
                c_gate = 0
                swap_gate = 0
            if val < 0:
                swap_gate -= 1
                layer[pos[0]] = swap_gate
                layer[pos[1]] = swap_gate
            else:
                c_gate += 1
                layer[pos[0]] = c_gate
                layer[pos[1]] = c_gate

        return_state.append(layer)

        return_state = np.array(return_state)

        if preprocessing:
            return_state = np.pad(
                return_state, ((0, self.depth-return_state.shape[0]), (0, 0)))
        return return_state

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
