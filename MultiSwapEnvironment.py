from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import math
from typing import List, Tuple
import copy
from stable_baselines3.common.env_checker import check_env
import pygame
import math
from itertools import compress

# types
Matrix = List[List[int]]

TimestepLayer = List[List[int]]
FlattenedTimeStepLayer = List[int]
State = List[TimestepLayer]
FlattenedState = List[FlattenedTimeStepLayer]

PermutationMatrix = List[List[int]]

#pygame constants
PG_WIDTH  = 100
PG_HEIGHT = 100
X_START   = PG_WIDTH*0.6
Y_START   = PG_HEIGHT*0.6
#Colors
WHITE   = (255,255,255)
BLACK   = (0,0,0)
BLUE    = (0,0,255)
GREEN   = (0,255,0)
RED     = (255,0,0)
CYAN    = (0,255,255)
PURPLE  = (255,0,255)
YELLOW  = (255,255,0)
BROWN   = (165,42,42)
PINK    = (255,20,147)
GREY    = (50.2,50.2,50.2)
PURPLE  = (50.2,0,50.2)
LIME    = (191,255,0)

def main():
    env = swap_environment(10,3,3)
    check_env(env)
    

#Our environment
class swap_environment(Env):
    def __init__(self, depth: int, rows: int, cols: int, 
            max_swaps_per_time_step: int = -1, timeout: int = 200) -> None:
        self.depth = depth
        self.rows = rows
        self.cols = cols
        if max_swaps_per_time_step < 0 or max_swaps_per_time_step > np.floor(self.rows * self.cols/2):
            self.max_swaps_per_time_step = np.floor(self.rows * self.cols/2)
        else:
            self.max_swaps_per_time_step = max_swaps_per_time_step
        self.timeout = timeout
        #array of possible actions
        self.possible_actions = self.get_possible_actions()
        #Number of actions we can take
        self.action_space = Discrete(len(self.possible_actions))
        self.observation_space = Box(low=0, high=np.floor(self.rows * self.cols / 2),
                                shape=(1, depth, rows, cols, ), dtype=np.uint8)
        
        #reset environment
        self.reset()

        #pygame screen initialization
        self.screen = None
        self.isopen = True
    
    def step(self, action: int) -> Tuple[List[int], int, bool, 'info']:
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
        if self.max_episode_steps <= 0 or self.max_layers <= 0:
            done = True
        else:
            done = False
        
        info = {}
        self.state = self.state.reshape((self.depth, self.rows, self.cols))
        return self.state, reward, done, info

    def render(self, mode = "human", render_list = None) -> bool: 
        if render_list is None:
            return 
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((PG_WIDTH*self.cols,PG_HEIGHT*self.rows))
       
        num_font = pygame.font.SysFont(None,25)
        img0 = num_font.render('0',True,RED)
        img1 = num_font.render('1',True,RED)
        img2 = num_font.render('2',True,RED)
        img3 = num_font.render('3',True,RED)
        img4 = num_font.render('4',True,RED)
        img5 = num_font.render('5',True,RED)
        img6 = num_font.render('6',True,RED)
        img7 = num_font.render('7',True,RED)
        img8 = num_font.render('8',True,RED)
        img9 = num_font.render('9',True,RED) 
        s_img = num_font.render('S',True,BLACK)
        
        dict={
            0:BLACK,
            1:GREEN,
            2:BLUE,
            3:PURPLE,
            4:YELLOW,
            5:BROWN,
            6:PINK,
            7:GREY,
            8:PURPLE,
            9:LIME
            }

        num_dict={
                0:img0,
                1:img1,
                2:img2,
                3:img3,            
                4:img4,            
                5:img5,
                6:img6,
                7:img7,
                8:img8,
                9:img9  
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

        pygame.draw.rect(surface,WHITE,surface.get_rect())
         
        #row / col %

        for j in range(1,self.cols+1):
            for i in range(1,self.rows+1):
                pygame.draw.circle(surface,BLACK,((X_START*j),(Y_START*i)),20)
                if j < self.rows:
                    pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*(j+1)),((Y_START*i))),4)
                if i < self.cols: 
                    pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*j),((Y_START*(i+1)))),4)
                pygame.draw.circle(surface,dict.get(render_list[0][i-1][j-1]),((X_START*j),(Y_START*i)),15)
                surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))

        self.screen.blit(surface,(0,0))
        pygame.display.flip()

        index = 0
        running = True

        while running:
            ev = pygame.event.get()

            for event in ev:

                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if index%2 == 0:                    
                        pygame.draw.rect(surface,WHITE,surface.get_rect())
                        for j in range(1,self.cols+1):
                            for i in range(1,self.rows+1):
                                pygame.draw.circle(surface,BLACK,((X_START*j),(Y_START*i)),20)
                                if j < self.rows:
                                    pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*(j+1)),((Y_START*i))),4)
                                if i < self.cols: 
                                    pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*j),((Y_START*(i+1)))),4)
                                    surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))

                    if event.key == pygame.K_n:
                        #next one
                        if index == len(render_list)-1:
                            print("At last obs")
                        else:
                            index += 1
                        

                            if type(render_list[index]) is list:
                                pygame.draw.rect(surface,WHITE,surface.get_rect())
                                for j in range(1,self.cols+1):
                                    for i in range(1,self.rows+1):
                                        pygame.draw.circle(surface,BLACK,((X_START*j),(Y_START*i)),20)
                                        if j < self.rows:
                                            pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*(j+1)),((Y_START*i))),4)
                                        if i < self.cols: 
                                            pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*j),((Y_START*(i+1)))),4)
                                        pygame.draw.circle(surface,dict.get(render_list[index][i-1][j-1]),((X_START*j),(Y_START*i)),15)
                                        surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))
                            else:
                                swap_matrix = self.possible_actions[render_list[index]]

                                for j in range(1,self.cols+1):
                                    for i in range(1,self.rows+1):
                                        pygame.draw.circle(surface,dict.get(render_list[index-1][i-1][j-1]),((X_START*j),(Y_START*i)),15)
                                        surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))
                                tuple_list = self.action_render(swap_matrix)
                                   
                                num_matrix = np.matmul(swap_matrix,np.asarray(num_matrix).reshape(self.rows*self.cols)).reshape((self.rows,self.cols)).tolist()
                                for i in range(len(num_matrix)): 
                                    num_matrix[i] = list(map(int,num_matrix[i]))

                                for t in tuple_list:
                                    r0 = t[0]//self.cols
                                    c0 = t[0]%self.cols
                                    r1 = t[1]//self.cols
                                    c1 = t[1]%self.cols
                                    x0 = X_START*(c0+1)
                                    y0 = Y_START*(r0+1)
                                    x1 = X_START*(c1+1)
                                    y1 = Y_START*(r1+1)
                                    x = x1+((x0-x1)/2)
                                    y = y1+((y0-y1)/2)
                                    pygame.draw.rect(surface,CYAN,pygame.Rect((x-10,y-10),(20,20)))
                                    surface.blit(s_img,(x-6,y-8))
                            

                            self.screen.blit(surface,(0,0))
                            pygame.display.flip()
                    if event.key == pygame.K_b:
                        #back one
                        if index == 0:
                            print("At first obs")
                        else:    
                            index -= 1
                        
                            if type(render_list[index]) is list:
                                pygame.draw.rect(surface,WHITE,surface.get_rect())
                            
                                for j in range(1,self.cols+1):
                                    for i in range(1,self.rows+1):
                                        pygame.draw.circle(surface,BLACK,((X_START*j),(Y_START*i)),20)
                                        if j < self.rows:
                                            pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*(j+1)),((Y_START*i))),4)
                                        if i < self.cols: 
                                            pygame.draw.line(surface,BLACK,((X_START*j),(Y_START*i)),((X_START*j),((Y_START*(i+1)))),4)
                                        pygame.draw.circle(surface,dict.get(render_list[index][i-1][j-1]),((X_START*j),(Y_START*i)),15)
                                        surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))
                                 
                            else: 
                                swap_matrix = self.possible_actions[render_list[index]]

                                for j in range(1,self.cols+1):
                                    for i in range(1,self.rows+1):
                                        pygame.draw.circle(surface,dict.get(render_list[index-1][i-1][j-1]),((X_START*j),(Y_START*i)),15)
                                        surface.blit(num_dict.get(num_matrix[i-1][j-1]),((X_START*j)-5,(Y_START*i)-5))
                                tuple_list = self.action_render(swap_matrix)
                            
                                num_matrix = np.matmul(np.asarray(num_matrix).reshape(self.rows*self.cols),swap_matrix.T).reshape((self.rows,self.cols)).tolist()
                                for i in range(len(num_matrix)): 
                                    num_matrix[i] = list(map(int,num_matrix[i]))

                                for t in tuple_list:
                                    r0 = t[0]//self.cols
                                    c0 = t[0]%self.cols
                                    r1 = t[1]//self.cols
                                    c1 = t[1]%self.cols
                                    x0 = X_START*(c0+1)
                                    y0 = Y_START*(r0+1)
                                    x1 = X_START*(c1+1)
                                    y1 = Y_START*(r1+1)
                                    x = x1+((x0-x1)/2)
                                    y = y1+((y0-y1)/2)
                                    pygame.draw.rect(surface,CYAN,pygame.Rect((x-10,y-10),(20,20)))
                                    surface.blit(s_img,(x-6,y-8))
                           
                            self.screen.blit(surface,(0,0))
                            pygame.display.flip()

        self.screen.blit(surface,(0,0))

        pygame.event.pump()
        pygame.display.flip()
        return self.isopen

    def reset(self, code: State = None) -> State:
        self.max_layers = self.depth
        if code is None:
            self.code = self.processing(self.make_code().reshape((self.depth, self.rows * self.cols)), preprocessing = True)
        else:
            self.code = self.processing(code.reshape((self.depth, self.rows * self.cols)), preprocessing = True)

        self.code = np.pad(self.code, ((0,self.depth),(0,0)))
        self.state, self.code = self.code[:self.depth], self.code[self.depth:]
        self.state = self.state.reshape((self.depth, self.rows, self.cols))

        self.max_episode_steps = self.timeout
        return self.state

    def is_executable_state(self, state: State) -> bool:
        """
        Input: 
            - state: A flattened state of gates
        
        Output: Bool which is True if all gates are executable in the first timestep
        """
        for pos in range(self.rows * self.cols):
            gate = state[0][pos]
            if gate > 0:
                neighbors = [state[0][pos+i] if pos+i >= 0 and pos+i < self.rows*self.cols 
                        and not (pos%self.rows == 0 and i == -1) 
                        and not (pos%self.rows == self.rows-1 and i == 1) else 0 
                        for i in [1, -1, self.rows, -self.rows]]
                if not gate in neighbors:
                    return False
        return True
    
    def get_possible_actions(self, iterations: int = None, used: List[int] = None) -> List[PermutationMatrix]:
        """
        Input: 
            - iterations: The current iteration of the recurtion
            - used: What qubits have been used for gates

        Output: List of permutation matrices corresponding to all possible actions
                for the current size of quantum circuit
        """
        if used is None:
            used = []

        if iterations is None or iterations == -1:
            iterations = self.max_swaps_per_time_step
        m = np.arange(self.rows*self.cols)
        possible_actions = []
        for pos in m:
            if not pos in used:
                neighbors = [m[pos+i] if pos+i >= 0 and pos+i < self.rows*self.cols 
                        and not m[pos+i] in used
                        and not (pos%self.rows == 0 and i == -1) 
                        and not (pos%self.rows == self.rows-1 and i == 1) else -1 
                        for i in [1, -1, self.cols, -self.cols]]
                for target in neighbors:
                    if target != -1:
                        a = [pos, target]
                        a.sort()
                        if not [a] in possible_actions:
                            used_tmp = used.copy()
                            possible_actions.append([a])
                            used_tmp.append(pos)
                            used_tmp.append(target)
                            if iterations >= 1: 
                                for action in self.get_possible_actions(iterations = iterations-1, used = used_tmp):
                                    action.append(a)
                                    action.sort()
                                    if not action in possible_actions:
                                        possible_actions.append(action)

        if iterations == self.max_swaps_per_time_step:
            return_possible_actions = np.zeros((len(possible_actions)+1, self.rows*self.cols, self.rows*self.cols))
            return_possible_actions[0] = np.identity(self.rows*self.cols)
            for idx, action in enumerate(possible_actions):
                m = np.identity(self.rows*self.cols)
                for swap in action:
                    pos1, pos2 = swap
                    m[pos1][pos1] = 0
                    m[pos2][pos2] = 0
                    m[pos1][pos2] = 1
                    m[pos2][pos1] = 1
                return_possible_actions[idx+1] = m
            return return_possible_actions
        
        return possible_actions

    def make_state_slice(self) -> FlattenedTimeStepLayer:
        """
        Output: Flattened timestep layer of random gates
        """
        max_gates = math.floor(self.rows*self.cols/2)
        state_slice = np.zeros(self.rows*self.cols)
        for i in range(1, np.random.choice(range(2, max_gates+2))):
            state_slice[i-1] = i
            state_slice[i-1+max_gates] = i
        np.random.shuffle(state_slice)
        return state_slice

    def make_code(self) -> State:
        """
        Output: State composed of random timestep layers with random gates
        """
        state = np.zeros((self.max_layers, self.rows, self.cols))
        for i in range(len(state)):
            state[i] = self.make_state_slice().reshape((self.rows, self.cols))
        return state

    def reward_func(self, state: FlattenedState, action: int) -> int:
        """
        Input:
            - state: A flattened state of gates
            - action: Action

        Output: The immediate reward
        """
        parallell_actions, _ = self.get_parallell_actions(state)
        if self.is_executable_state(state):
            if action in parallell_actions:
                return 0
            return -1
        return -2

    def get_parallell_actions(self, state: State) -> Tuple[List[int], List[int]]:
        """
        Input:
            - state: A flattened state of gates

        Output: List of actions that do not affect any gates in the first timestep
                of the state
        """
        used_matrix = np.zeros(self.possible_actions.shape)
        used = np.where(state[0]>0)[0]
        for i in used:
            used_matrix[:,i,i] = 1
        
        tmp = np.sum(np.bitwise_and(used_matrix.astype(int), self.possible_actions.astype(int)), axis=(1,2)) == len(used)

        parallell_actions = np.array([i for i, v in enumerate(tmp) if v])
        non_parallell_actions = np.array([i for i, v in enumerate(tmp) if not v])

        return parallell_actions, non_parallell_actions
    
    def processing(self, state: State, preprocessing: bool = True) -> State:
        """
        Input:
            - state: A flattened state of gates
            - preprocessing: bool that tells if this is used as preprocessing or postprocessing
        
        Output: Flattened compressed state
        """
        gates = []
        for idx, m in enumerate(state):
            used = []
            for v in m:
                if v != 0 and v not in used:
                    used.append(v)
                    gates.append((v/abs(v), np.array([i for i, x in enumerate(m) if x == v])))
        
        return_state = []
        c_gate = 0
        swap_gate = 0
        layer = np.zeros(self.rows*self.cols)
        for v, x in gates:
            if layer[x[0]] != 0 or layer[x[1]] != 0:
                return_state.append(layer)
                layer = np.zeros(self.rows*self.cols)
                c_gate = 0
                swap_gate = 0
            if v < 0:
                swap_gate -= 1
                layer[x[0]] = swap_gate
                layer[x[1]] = swap_gate
            else:
                c_gate += 1
                layer[x[0]] = c_gate
                layer[x[1]] = c_gate
            
        return_state.append(layer)

        return_state = np.array(return_state)

        if preprocessing:
            return_state = np.pad(return_state, ((0,self.depth-return_state.shape[0]),(0,0)))
        return return_state

    def action_render(self, action_matrix: PermutationMatrix) -> List[Tuple[int, int]]:
        """
        Input:
            - action_matrix: PermutationMatrix corresponding to an action

        Output: List of tuples of ints describing between what qubits SWAP-gates are placed
        """
        action_matrix = action_matrix.tolist()
        action_tuples = [] 
        used_nodes = [] 
        for i in range(len(action_matrix)): 
            if i not in used_nodes: 
                idx = action_matrix[i].index(1) 
                used_nodes.append(idx)
                if idx != i:
                    action_tuples.append(tuple((i,idx)))
        return action_tuples

if __name__ == '__main__':
    main()
