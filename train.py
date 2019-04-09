from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, load_dataset
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=64, help='Number of samples in a mini batch.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')  # this is important
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def train(epoch):
    batch = args.batch
    output_epoch = []
    loss_epoch = []
    t = time.time()
    model.train()
    for idx in range(0, len(x_train), batch):
        features = Variable(torch.FloatTensor(x_train[idx:idx+batch]))
        adj = Variable(torch.FloatTensor(np.ones([features.shape[0], features.shape[1], features.shape[1]])))
        labels = Variable(torch.LongTensor(y_train[idx:idx+batch]))
        optimizer.zero_grad()
        # forward
        output = model(features, adj)
        output_epoch.append(output.data.numpy())
        output = output.view(-1, 2)
        labels = labels.view(-1)

        # backward
        loss_train = F.nll_loss(output, labels)
        loss_epoch.append(loss_train.data[0])
        acc_train = accuracy(output, labels)
        loss_train.backward()
        optimizer.step()
        if int(idx/batch) % 10 ==0:  # print log per 10 batches
            print(
                'Batch {:d}'.format(int(idx/batch)+1),
                'loss_train: {:.4f}'.format(loss_train.data[0]),
                'acc_train: {:.4f}'.format(acc_train.data[0]),
            )

    # test
    features = Variable(torch.FloatTensor(x_test))
    adj = Variable(torch.FloatTensor(np.ones([features.shape[0], features.shape[1], features.shape[1]])))
    labels = Variable(torch.LongTensor(y_test))

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    output = output.view(-1, 2)
    labels = labels.view(-1)
    loss_val = F.nll_loss(output, labels)
    acc_val = accuracy(output, labels)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(np.mean(np.array(loss_epoch))),
          '-- loss_val: {:.4f}'.format(loss_val.data[0]),
          'acc_val: {:.4f}'.format(acc_val.data[0]),
          'time: {:.4f}s'.format(time.time() - t))

    return np.mean(np.array(loss_epoch))


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))

if __name__=='__main__':
    # Load data
    X, Y = load_dataset(args.seed)
    split_rate = 0.8
    x_train = X[0:int(split_rate * X.shape[0])]
    x_test = X[int(split_rate * X.shape[0]):]

    y_train = Y[0:int(split_rate * Y.shape[0])]
    y_test = Y[int(split_rate * Y.shape[0]):]

    features = Variable(torch.FloatTensor(x_train[0]))
    adj = Variable(torch.FloatTensor(np.ones([features.shape[0],features.shape[0]])))
    labels = Variable(torch.LongTensor(y_train[0]))


    # Model and optimizer
    if args.sparse:
        model = SpGAT(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=int(y_train.max()) + 1,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)
    else:
        model = GAT(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=int(y_train.max()) + 1,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        # idx_train = idx_train.cuda()
        # idx_val = idx_val.cuda()
        # idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_values.append(train(epoch))

        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    compute_test()
