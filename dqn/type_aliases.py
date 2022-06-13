"""Common aliases for type hints"""

from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common import callbacks
from stable_baselines3.common import vec_env

GymEnv = Union[gym.Env, vec_env.VecEnv]
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]
TensorDict = Dict[Union[str, int], th.Tensor]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[None, Callable, List[callbacks.BaseCallback], callbacks.BaseCallback]

# A schedule takes the remaining progress as input
# and ouputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]

class CustomReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    next_observations: th.Tensor
    rewards: th.Tensor


class CustomDictReplayBufferSamples(CustomReplayBufferSamples):
    observations: TensorDict
    V_next_observations: TensorDict
    rewards: th.Tensor
