import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import SumPooling
from .ggin_layer import GGINLayer

class GGIN(nn.Module):
    def __init__(self, in_feats, h_feats, num_layers, num_classes):
        super(GGIN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GGINLayer(in_feats, h_feats))
        for _ in range(num_layers - 1):
            self.layers.append(GGINLayer(h_feats, h_feats))
        self.pool = SumPooling()
        self.fc1 = nn.Linear(h_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, num_classes)

    def forward(self, g, inputs, g_initial, g_lead):
        h = inputs
        h_lead = g_lead.ndata['feat']
        h_initial = self.pool(g_initial, g_initial.ndata['feat'])
        h_global = self.pool(g, h)

        for layer in self.layers:
            h = layer(g, h, h_global, h_initial, h_lead)
            h = torch.relu(h)
            h_global = self.pool(g, h)

        h = torch.relu(self.fc1(h_global))
        h = self.fc2(h)
        return h
