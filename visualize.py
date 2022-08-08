from typing import List, Tuple
import numpy as np
from gym.envs.registration import register
from multiswap_environment import SwapEnvironment
from stable_baselines3.common.env_util import make_vec_env
from dqn.dqn import DQN

Matrix = List[List[int]]

DEPTH_OF_CODE = 10
ROWS = 2
COLS = 2
MAX_SWAPS_PER_TIME_STEP = -1

N_EVAL_EPISODES = 1000

MODEL_DIR = "models/"

modelName = f"model-{DEPTH_OF_CODE}-{ROWS}-{COLS}"

register(
    id="multiswap_environment-v0",
    entry_point="multiswap_environment:SwapEnvironment",
    max_episode_steps=200,
)

venv = make_vec_env("multiswap_environment-v0", n_envs=1, env_kwargs={
                    "depth": DEPTH_OF_CODE, "rows": ROWS, "cols": COLS})
env = venv.envs[0]

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = DQN.load(MODEL_DIR + modelName, env=venv)


render_list = []
a = []


def pm_to_state_rep(pm: Matrix):
    action_matrix = pm.tolist()
    action_tuples = []
    used_nodes = []
    for index, value in enumerate(action_matrix):
        if index not in used_nodes:
            idx = action_matrix[index].index(1)
            used_nodes.append(idx)
            if idx != index:
                action_tuples.append(tuple((index, idx)))
    idx = -1
    return_a = np.zeros(ROWS*COLS)
    for q_0, q_1 in action_tuples:
        return_a[q_0] = idx
        return_a[q_1] = idx
        idx -= 1
    return list(return_a.reshape(ROWS*COLS))


def squash(swap_matrix, preprocessing=False):
    gates = []
    for index, value_list in enumerate(swap_matrix):
        used = []
        for value in value_list:
            if value != 0 and value not in used:
                used.append(value)
                gates.append(
                    (value/abs(value), np.array([i for i, x in enumerate(value_list) if x == value])))

    return_state = []
    i = 0
    c_gate = 0
    swap_gate = 0
    layer = np.zeros(ROWS*COLS)
    for v, x in gates:
        if layer[x[0]] != 0 or layer[x[1]] != 0:

            return_state.append(layer)
            layer = np.zeros(ROWS*COLS)
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
        return_state = np.pad(
            return_state, ((0, DEPTH_OF_CODE-return_state.shape[0]), (0, 0)))
    return return_state


env.reset()

env.state = squash(env.state.reshape(
    (DEPTH_OF_CODE, ROWS*COLS)), True).reshape(env.state.shape)

obs = env.state

#done = False
# render_list.append(env.state[0][0].tolist())
# a.append(env.state[0][0].reshape(rows*cols).tolist())
#length = 0
#reward = 0
# while not done:
#    action, _states = model.predict(obs[None][None], venv,  deterministic=True)
#    if action != 0:
#        render_list.append(action[0])
# only add first obs since it removes the first one to step
#    obs,r,done,_ = env.step(action[0])
#    length += 1
#    reward += r
#    render_list.append(obs[0].tolist())
#    if action != 0:
#        a.append(pm_to_state_rep(env.possible_actions[action[0]]))
#    a.append(obs[0].reshape(rows*cols).tolist())

# print(reward)
print(np.array(a))
print(squash(np.array(a)))

env.render(mode="human", render_list=[[[1, 1], [2, 2]], 2, [
           [0, 1], [0, 1]], 3, [[1, 0], [0, 1]]])


#rewards = np.zeros(n_eval_episodes)
#current_reward, episode = 0, 0
#env = swap_environment(depth_of_code, rows, cols, max_swaps_per_time_step)
# while episode < n_eval_episodes:
#    action = env.action_space.sample()
#    _, reward, done, _ = env.step(action)
#    current_reward += reward
#    if done:
#        rewards[episode] = current_reward
#        current_reward = 0
#        episode += 1
#        env.reset()
#
#print(f"Mean reward random: {np.mean(rewards)} +/- {np.std(rewards)}")
#
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
#print(f"Mean reward model: {mean_reward} +/- {std_reward}")
