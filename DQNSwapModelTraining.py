## Update: Logs rollout, time and train to plot them in tensorboard using tensorboard --logdir '.\logdir\DQNModel(StateToValue)_[#nr]\' command.

from MultiSwapEnvironment import swap_environment

import numpy as np
import os
import matplotlib.pyplot as plt

from gym.envs.registration import register

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import EvalCallback
from dqn.evaluation import evaluate_policy

from dqn.policies import CustomCnnPolicy
from dqn.swap_vec_env import SwapVecEnv
from dqn.dqn import CustomDQN as DQN

#env variables
depth = 10
rows = 2
cols = 2
max_swaps_per_time_step = -1

#model variables (previously 2e4)
learning_starts = int(5e4)
verbose = 1
exploration_fraction = 0.5
exploration_initial_eps = 1
exploration_final_eps = 0.1
batch_size = 512
learning_rate = 0.001
target_update_interval = int(1e4)
tau = 0.5
gamma = 0.5
train_freq = 4

#training variables (previously 1e5)
total_timesteps = int(2e5)
log_interval = 4

#evaluation
n_eval_episodes = 200


register(
    id="MultiSwapEnvironment-v0",
    entry_point="MultiSwapEnvironment:swap_environment",
    max_episode_steps=200,
)

venv = make_vec_env("MultiSwapEnvironment-v0", n_envs = 20, env_kwargs = {"depth": depth, "rows": rows, "cols": cols})
env = swap_environment(depth, rows, cols)

eval_callback = EvalCallback(
        env, 
        best_model_save_path='./logdir/logs/',
        log_path='./logdir/logs/', 
        eval_freq=500,
        deterministic=True,
        verbose = 0,
        render=False, 
        n_eval_episodes = 10
        )

# Defining agent name
model_dir = "models/"
model_name = f"DQNModel({depth},{rows},{cols})"
logdir="logdir/"

# Intantiate the agent
model = DQN(CustomCnnPolicy, 
            venv, 
            verbose = verbose,
            train_freq = train_freq,
            gamma = gamma,
            tau = tau,
            target_update_interval = target_update_interval,
            learning_starts = learning_starts, 
            exploration_fraction = exploration_fraction, 
            exploration_final_eps = exploration_final_eps, 
            exploration_initial_eps = exploration_initial_eps,
            batch_size = batch_size,
            optimize_memory_usage = True,
            learning_rate = learning_rate,
            tensorboard_log=logdir
        )



# Train the agent
model.learn(total_timesteps = total_timesteps, 
            log_interval = log_interval, 
            tb_log_name=model_name, 
            callback=eval_callback)

# Save the agent
model.save(model_dir + model_name)

print("training done")

rewards = np.zeros(n_eval_episodes)
current_reward, episode = 0, 0
while episode < n_eval_episodes:
    action = env.action_space.sample()
    _, reward, done, _ = env.step(action)
    current_reward += reward
    if done:
        rewards[episode] = current_reward
        current_reward = 0
        episode += 1
        env.reset()

print(f"Mean reward random: {np.mean(rewards)}")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)

print(f"Mean reward model: {mean_reward}")

write_text=open(f"logdir/{model_name}_{len(os.listdir(logdir))}/{model_name}"+".txt","w+")
write_text.write(
    f"Model: {model_name} \r\n"+
    "\r\n"+
    
    "Environment Variables \r\n"+
    f"depth = {depth}\r\n"+
    f"rows = {rows}\r\n"+
    f"cols = {cols}\r\n"+
    f"max_swaps_per_time_step = {max_swaps_per_time_step}\r\n"+
    "\r\n"+
    
    "Model Variables \r\n"+
    f"learning_starts = {learning_starts}\r\n"+
    f"verbose = {verbose}\r\n"+
    f"exploration_fraction = {exploration_fraction}\r\n"+
    f"exploration_initial_eps = {exploration_initial_eps}\r\n"+
    f"exploration_final_eps = {exploration_final_eps}\r\n"+
    f"batch_size = {batch_size}\r\n"+
    f"learning_rate = {learning_rate}\r\n"+
    f"target_update_interval = {target_update_interval}\r\n"+
    f"tau = {tau}\r\n"+
    f"gamma = {gamma}\r\n"+
    f"train_freq = {train_freq}\r\n"+
    "\r\n"+
    
    "Training Variables \r\n"+
    f"total_timesteps = {total_timesteps}\r\n"+
    f"log_interval = {log_interval}\r\n"+
    "\r\n"+
    
    "Evaluation \r\n"+
    f"n_eval_episodes = {n_eval_episodes}\r\n"+
    "\r\n"+
    
    "Results \r\n"+
    f"np.mean(rewards) = {np.mean(rewards)}\r\n"+
    f"mean_reward = {mean_reward}\r\n")

write_text.close()
