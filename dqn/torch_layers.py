"""GNN"""
import gym
import torch as th
from torch import nn, flatten
from torch.nn import Linear
from torch.nn.functional import relu
from torch_geometric.nn import GCNConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GCNN(BaseFeaturesExtractor):
    def __init__(self, obs_space: gym.spaces.Box, feature_dim = 1):
        super(GCNN, self).__init__(obs_space, feature_dim)
        depth, n_qubits, features = obs_space.shape
        n_flatten = depth * n_qubits * features * 20
        self.gcn1 = GCNConv(1, 20)

        # Compute shape by doing one forward pass
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
        )

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = relu(x)
        x = flatten(x, start_dim = 1)
        return self.linear(x)
