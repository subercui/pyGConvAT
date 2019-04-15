from sklearn import manifold, svm, datasets
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from models import GAT
import pickle as pkl
import os

from torch.autograd import Variable
from utils import load_data, accuracy, statics, load_dataset

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=64, help='Number of samples in a mini batch.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=4, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')  # this is important
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=200, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


tsne = manifold.TSNE

def TF():
    """true positive"""
    return

def run_EEG_record():
    """test on a record trial"""
    return

def attention_graph():
    """draw the graph map based on the attention index"""
    return

def compute_test():
    features = Variable(torch.FloatTensor(x_test))
    adj = Variable(torch.FloatTensor(np.ones([features.shape[0], features.shape[1], features.shape[1]])))
    labels = Variable(torch.LongTensor(y_test))

    model.eval()
    output = model(features, adj)
    output = output.view(-1, 2)
    labels = labels.view(-1)
    loss_test = F.nll_loss(output, labels)
    acc_test = accuracy(output, labels)
    tp, tn, fp, fn = statics(output, labels)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          #"accuracy= {:.4f}".format(acc_test.data[0]),
          "accuracy= {:.4f}".format((tp+tn)/(tp+fp+tn+fn)),
          "sensitivity= {:.4f}".format(tp/(tp+fn)),
          "specificity= {:.4f}".format(tn/(tn+fp)),
          )

def svm_run_and_test(load=True):
    if load & os.path.isfile('svm_model.pkl'):
        with open('svm_model.pkl', 'rb') as f: clf = pkl.load(f)
        print('model loaded')
    else:
        features = x_train
        features = features.reshape(x_train.shape[0]*x_train.shape[1], -1, 12)
        features = np.mean(features, axis=-1, keepdims=False)
        across_chan = np.mean(
            features.reshape(x_train.shape[0],x_train.shape[1],-1),axis=1,keepdims=False
        ).repeat(x_train.shape[1],axis=0) # across channel features
        features = np.concatenate([features, across_chan],axis=-1)
        labels = y_train.reshape(-1)

        clf = svm.SVC(gamma='scale', class_weight={0:1,1:2})
        clf.fit(features, labels)
        print('svm train finished')
        with open('svm_model.pkl', 'wb') as f:
            pkl.dump(clf,f)
            print('model saved')

    #test
    features = x_test
    features = features.reshape(x_test.shape[0] * x_test.shape[1], -1, 12)
    features = np.mean(features, axis=-1, keepdims=False)
    across_chan = np.mean(
        features.reshape(x_test.shape[0], x_test.shape[1], -1), axis=1, keepdims=False
    ).repeat(x_test.shape[1], axis=0)  # across channel features
    features = np.concatenate([features, across_chan], axis=-1)
    labels = y_test.reshape(-1)
    output = clf.predict(features)
    preds = output
    correct = np.equal(preds, labels)
    tp = correct * preds
    tn = correct * (1 - preds)
    fp = (1 - correct) * preds
    fn = (1 - correct) * (1 - preds)
    tp, tn, fp, fn = tp.sum(), tn.sum(), fp.sum(), fn.sum()
    print("Test set results:",
          "accuracy= {:.4f}".format((tp + tn) / (tp + fp + tn + fn)),
          "sensitivity= {:.4f}".format(tp / (tp + fn)),
          "specificity= {:.4f}".format(tn / (tn + fp)),
          )
    return clf

if __name__ == '__main__':
    # Load data
    X, Y = load_dataset(args.seed)
    split_rate = 0.8
    x_train = X[0:int(split_rate * X.shape[0])]
    x_test = X[int(split_rate * X.shape[0]):]

    y_train = Y[0:int(split_rate * Y.shape[0])]
    y_test = Y[int(split_rate * Y.shape[0]):]

    features = Variable(torch.FloatTensor(x_train[0]))
    adj = Variable(torch.FloatTensor(np.ones([features.shape[0], features.shape[0]])))
    labels = Variable(torch.LongTensor(y_train[0]))

    # Restore best model
    model = GAT(nfeat=120,
                nhid=4,
                nclass=int(y_train.max()) + 1,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
    model.load_state_dict(torch.load('{}/{}.pkl'.format('output(Apr 12 20.08.47 2019)', '649')))
    print('model loaded')

    # Testing
    mode = 'svm'  # define which test to run: GADN, svm, run_record, cut_channel_GADN, cut_channel_svm
    if mode == 'GADN':
        compute_test()
    elif mode == 'svm':
        clf = svm_run_and_test()