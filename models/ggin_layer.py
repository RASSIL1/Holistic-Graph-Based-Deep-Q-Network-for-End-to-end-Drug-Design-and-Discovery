import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import SumPooling

class GGINLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GGINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(out_feats, out_feats),
            nn.ReLU(),
            nn.Linear(out_feats, out_feats)
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.pool = SumPooling()
        self.attention = nn.Linear(out_feats, 1)

    def forward(self, g, h, h_global, h_initial, h_lead):
        with g.local_scope():
            h_in = h
            g.ndata['h'] = h
            g.update_all(dgl.function.u_add_v('h', 'h', 'm'), dgl.function.sum('m', 'neigh'))
            h = (1 + self.eps) * h_in + g.ndata['neigh']
            lead_attention = torch.sigmoid(self.attention(h_lead)).unsqueeze(1).expand(-1, h.size(1), -1)
            h_sum = h + h_global + h_initial + lead_attention * h_lead
            h = self.mlp(h_sum)

            return h
