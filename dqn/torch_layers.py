"""GNN"""
import gym
import torch
from torch.nn import Linear
from torch.nn.functional import relu
from torch_geometric.nn import GCNConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GCNN(BaseFeaturesExtractor):
    def __init__(self, obs_space: gym.spaces.Box, feature_dim = 1):
        super(GCNN, self).__init__(obs_space, feature_dim)
        shape = obs_space.shape

        self.gcn1 = GCNConv(9, 64)
        self.gcn2 = GCNConv(64, 64)
        self.dense1 = Linear(64, feature_dim)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = relu(x)
        x = self.gcn2(x, edge_index)
        x = relu(x)
        return self.dense1(x)
