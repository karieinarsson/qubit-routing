"""GNN"""
import gym
import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, obs_space: gym.spaces.Box, out_channels = 1):
        super().__init__()
        in_channels = obs_space[0]

        self.gcn1 = GCNConv(in_channels, 64)
        self.gcn2 = GCNConv(64, 64)
        self.dense1 = Linear(64, out_channels)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = ReLU(x)
        x = self.gcn2(x, edge_index)
        x = ReLU(x)
        return self.dense1(x)
