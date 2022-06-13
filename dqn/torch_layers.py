from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import gym
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor 

class CustomCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 1):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels, depth_of_code, rows, cols = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, 32, kernel_size=(depth_of_code,2,2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()).float()[None]).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )
        

    def forward(self, observations: th.Tensor) -> th.Tensor:
        returnTensor = self.linear(self.cnn(observations))
        return returnTensor
