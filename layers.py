import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, Wb=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # important - Parameter() add vector to back prop
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(3*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.IF_Wb = Wb
        if self.IF_Wb:
            self.encoder = nn.Linear(in_features=in_features, out_features=out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # h = torch.mm(input, self.W)  # nodes * features
        B = input.size(0)
        h = input.matmul(self.W)  # batch * nodes * features
        N = h.size()[1]  # nodes

        H_self = h.repeat(1, 1, N).view(B, N * N, -1)  # (N, nodes*nodes, features)
        H_neibor = h.repeat(1, N, 1)
        H_corr = H_self * H_neibor
        a_input = torch.cat([H_self, H_neibor, H_corr], dim=2).view(B, N, -1, 3 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # attention coefficient, batch * N * N #TODO: need more layers, add h.*h

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)  # N * N, attention[0][1] sums to 1.
        if self.IF_Wb:
            h_ = self.encoder(input)  # encode with a different W
            h_prime = torch.matmul(attention, h_)  # N * features
        else:
            h_prime = torch.matmul(attention, h)  # N * features

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def forward2(self, input, adj):
        # h = torch.mm(input, self.W)  # nodes * features
        B = input.size(0)
        h = input.matmul(self.W)  # batch * nodes * features
        N = h.size()[1]  # nodes

        H_self = h.repeat(1, 1, N).view(B, N * N, -1)  # (N, nodes*nodes, features)
        H_neibor = h.repeat(1, N, 1)
        H_corr = H_self * H_neibor
        a_input = torch.cat([H_self, H_neibor, H_corr], dim=2).view(B, N, -1, 3 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # attention coefficient, batch * N * N #TODO: need more layers, add h.*h

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)  # (batch, N, N)
        return attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class BrainDecodeConv(nn.Module):
    """similar implementation of the braindecode network but for this spike detection task"""
    def __init__(self, out_channels=8):
        super(BrainDecodeConv, self).__init__()
        c_out = out_channels
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=c_out, kernel_size=10, stride=2, dilation=1)
        self.c_out = c_out

    def calc_out_features(self, in_features):
        out1_feaures = int(np.floor((in_features-10)/2 + 1))
        out2_features = int(np.floor((out1_feaures-3)/3 + 1))
        out_features = self.c_out * (out2_features)
        return out_features

    def forward(self, x):
        """
                x: (b_s, len, embsize)
                """
        assert x.dim() == 3  # (batch, channels, features)
        batch, channels = x.size(0), x.size(1)
        x = x.view(batch * channels, 1, x.size(2))  # (batch*channels, 1, features)

        # conv
        out1 = F.relu(self.conv1(x))  # (batch*chanels, c_out, features1)
        out2 = F.max_pool1d(out1, kernel_size=3)

        # concatenate conv outputs
        out = out2.view(batch * channels, -1)  # (batch*chanels, features)
        out = out.view(batch, channels, out.size(1))

        return out


class Conv1dGroup(nn.Module):
    """
    A group of 3 layer 1d conv layers
    """
    def __init__(self, out_channels=8):
        super(Conv1dGroup, self).__init__()
        c_out = out_channels
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=c_out, kernel_size=10, stride=2, dilation=1)
        # out feature dim: (N, c_out, (nfeat-10)/2 +1)
        self.conv2 = nn.Conv1d(in_channels=c_out, out_channels=c_out, kernel_size=5, stride=1, dilation=1)
        self.conv3 = nn.Conv1d(in_channels=c_out, out_channels=c_out, kernel_size=5, stride=1, dilation=2)
        self.c_out = c_out

    def calc_out_features(self, in_features):
        out1_feaures = round((in_features-10)/2 + 1)
        out2_features = round((out1_feaures-5)/1 + 1)
        out3_features = round((out2_features-2*(5-1)-1)/1 + 1)
        out_features = self.c_out * (out2_features+out3_features)
        return out_features

    def forward(self, x):
        """
        x: (b_s, len, embsize)
        """
        assert x.dim() == 3  # (batch, channels, features)
        batch, channels = x.size(0), x.size(1)
        x = x.view(batch*channels, 1, x.size(2))  # (batch*channels, 1, features)

        # conv
        out1 = F.relu(self.conv1(x))  # (batch*chanels, c_out, features1)
        out2 = F.relu(self.conv2(out1))  # (batch*chanels, c_out, features2)
        out3 = F.relu(self.conv3(out2))  # (batch*chanels, c_out, features3)

        # concatenate conv outputs
        out = torch.cat([
            out2.view(batch*channels, -1),
            out3.view(batch*channels, -1)
        ], dim=-1)  # (batch*chanels, features)
        out = out.view(batch, channels, out.size(1))

        return out


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1)).cuda())
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
