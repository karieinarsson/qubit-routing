from typing import Any, Dict, List, Optional, Type, Union, Tuple

import gym
import torch as th
from torch import nn
from gym.spaces import Discrete
import numpy as np

from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import register_policy

from dqn.torch_layers import CustomCNN

class CustomCnnPolicy(DQNPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(CustomCnnPolicy, self).__init__(
            observation_space,
            Discrete(1),
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def predict(
        self,
        observations: Union[np.ndarray, Dict[str, np.ndarray]],
        env: VecEnv,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        self.set_training_mode(False)
        
        actions = np.zeros(env.num_envs, dtype = int)
        possible_actions = env.envs[0].possible_actions

        for idx, obs in enumerate(observations):
            x, d, r, c = obs.shape
            obs = obs.reshape((d, r*c))
            action_set = env.envs[0].pruning(obs)
                
            with th.no_grad(): 
                action = th.Tensor(np.arrar([possible_actions[i] for i in action_set]))
                tensor_obs = th.Tensor(obs).reshape((d,r*c,))
                tensor_obs = th.matmul(tensor_obs, action)
                value = self._predict(tensor_obs.reshape((len(action),x,d,r,c)), deterministic=deterministic)

            for i, o in enumerate(np.array(tensor_obs)):
                value[i] += env.envs[0].reward_func(o, action_set[i])
            
            actions[idx] = action_set[np.argmax(value)]
            
        return actions, state
