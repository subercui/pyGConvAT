import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer, Conv1dGroup, BrainDecodeConv


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.conv_encoder = Conv1dGroup(out_channels=nheads)
        enc_nfeat = self.conv_encoder.calc_out_features(in_features=nfeat)
        self.attentions = [GraphAttentionLayer(enc_nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # important add to graph

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.out_layer1 = nn.Linear(in_features=nhid * nheads, out_features=10)
        self.out_layer2 = nn.Linear(in_features=10, out_features=nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        enc_out = self.conv_encoder(x)
        x = torch.cat([att(enc_out, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_layer1(x))
        x = self.out_layer2(x)
        out = F.log_softmax(x, dim=-1)
        return out


class CNNBaseline(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(CNNBaseline, self).__init__()
        self.dropout = dropout

        self.conv_encoder = Conv1dGroup(out_channels=nheads)
        enc_nfeat = self.conv_encoder.calc_out_features(in_features=nfeat)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.out_layer1 = nn.Linear(in_features=enc_nfeat, out_features=10)
        self.out_layer2 = nn.Linear(in_features=10, out_features=nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        enc_out = self.conv_encoder(x)
        x = F.dropout(enc_out, self.dropout, training=self.training)
        x = F.elu(self.out_layer1(x))
        x = self.out_layer2(x)
        out = F.log_softmax(x, dim=-1)
        return out


class BrainDecode(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(BrainDecode, self).__init__()
        self.dropout = dropout

        self.conv_encoder = BrainDecodeConv(out_channels=nheads)
        enc_nfeat = self.conv_encoder.calc_out_features(in_features=nfeat)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.across_chan = nn.Linear(in_features=18*enc_nfeat, out_features=enc_nfeat)
        self.out_layer1 = nn.Linear(in_features=2*enc_nfeat, out_features=10)
        self.out_layer2 = nn.Linear(in_features=10, out_features=nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        enc_out = self.conv_encoder(x)
        x = F.dropout(enc_out, self.dropout, training=self.training)
        x_chan = F.elu(self.across_chan(x.view(x.size(0),1,-1))).repeat(1,18,1)  # feature across all channels
        x = torch.cat([x, x_chan.view(x.size())], dim=-1)
        x = F.elu(self.out_layer1(x))
        x = self.out_layer2(x)
        out = F.log_softmax(x, dim=-1)
        return out

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

