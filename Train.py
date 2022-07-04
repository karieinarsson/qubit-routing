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
from dqn.dqn import CustomDQN as DQN

def main():
    Training(total_timesteps = int(3e5), rows = 3, cols = 2)
    

def Training(
    #env variables
    depth = 10,
    rows = 3,
    cols = 3,
    max_swaps_per_time_step = -1,
    n_envs = 20,

    #model variables (previously 2e4)
    learning_starts = int(5e4),
    verbose = 1,
    exploration_fraction = 0.5,
    exploration_initial_eps = 1,
    exploration_final_eps = 0.1,
    batch_size = 512,
    learning_rate = 0.001,
    target_update_interval = int(1e4),
    tau = 0.5,
    gamma = 0.5,
    train_freq = 4,

    #training variables (previously 1e5)
    total_timesteps = int(7e4),
    log_interval = 4,

    #evaluation
    eval_freq = 500,
    n_eval_episodes = 10
):

    register(
        id="MultiSwapEnvironment-v0",
        entry_point="MultiSwapEnvironment:swap_environment",
        max_episode_steps=200,
    )

    venv = make_vec_env("MultiSwapEnvironment-v0", n_envs = n_envs, env_kwargs = {"depth": depth, "rows": rows, "cols": cols})
    env = swap_environment(depth, rows, cols)

    eval_callback = EvalCallback(
            env, 
            best_model_save_path='./logdir/logs/',
            log_path='./logdir/logs/', 
            eval_freq=eval_freq,
            deterministic=True,
            verbose = verbose,
            render=False, 
            n_eval_episodes = n_eval_episodes
            )

    # Defining agent name
    model_dir = "models/"
    model_name = f"model-{depth}-{rows}-{cols}"
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

    print("Training done!")

if __name__ == "__main__":
    main()
