'''
Training function
'''
from gym.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from dqn.policies import CnnPolicy as CustomCnnPolicy
from dqn.dqn import DQN
from multiswap_environment import SwapEnvironment
from stable_baselines3.common.vec_env import SubprocVecEnv

def main():
    '''
    Runs one training run
    '''
    train(
        total_timesteps = int(15e6),
        rows = 4,
        cols = 3,
        verbose = 0,
        exploration_fraction = 0.2
    )

def train(
    #env variables
    depth = 10,
    rows = 3,
    cols = 3,
    n_envs = 24,

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
    '''
    Trains a DDQN model to compile quantum code
    '''

    register(
        id="multiswap_environment-v0",
        entry_point="multiswap_environment:SwapEnvironment",
        max_episode_steps=200,
    )

    venv = make_vec_env(
            "multiswap_environment-v0",
            n_envs = n_envs,
            env_kwargs = {"depth": depth, "rows": rows, "cols": cols},
            vec_env_cls = SubprocVecEnv,
            vec_env_kwargs = {"start_method": "fork"}
            )
    env = SwapEnvironment(depth, rows, cols)

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
                tensorboard_log = logdir,
                gradient_steps = -1
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
