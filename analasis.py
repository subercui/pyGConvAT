from sklearn import manifold, svm, datasets
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from models import GAT, CNNBaseline
import pickle as pkl
import os
import mne

from torch.autograd import Variable
from utils import load_data, accuracy, statics, load_dataset, raw_eeg_pick, normalization

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

def run_EEG_record(model, file='EEG_data/DA0570A6_1-1+.edf'):
    """test on a record trial"""
    raw = mne.io.read_raw_edf(file, preload=True)
    raw = raw_eeg_pick(raw)  # now only EEG channels

    x = get_data(raw)  # (batch, chans, features)

    # run model
    features = Variable(torch.FloatTensor(x))
    adj = Variable(torch.FloatTensor(np.ones([features.shape[0], features.shape[1], features.shape[1]])))

    model.eval()
    output, attentions = model.forward2(features, adj)
    output = output.view(-1, 2)
    preds = output.max(1)[1].view(attentions.size(0),attentions.size(1)).data.numpy()
    attentions = attentions.data.numpy()

    # select attentions
    idx_1 = (preds.sum(axis=-1) > 0)
    attentions = attentions[idx_1,:,:,:]  # (66, 18, 18, 4)
    attentions = np.mean(attentions, axis=0)  # (18, 18, 4)
    attentions = np.concatenate([attentions, attentions.mean(axis=-1, keepdims=True)], axis=-1)

    # draw attention graph
    #TODO: need to draw every channels' graph, and a combined one
    for i in range(5):  # 5 graphs
        distance = to_symetric(attentions[:,:,i])
        # # MDS
        # mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, #random_state=seed,
        #                    dissimilarity="precomputed", n_jobs=1)
        # pos = mds.fit_transform(distance)
        # attention_graph(pos, title='MDS, Attention Head {}'.format(i))
        # t-sne
        tsne = manifold.TSNE(n_components=2, perplexity=5,  # random_state=seed,
                           metric="precomputed")
        pos = tsne.fit_transform(distance)
        attention_graph(pos, title='t-sne, Attention Head {}'.format(i))
    return


def to_symetric(mat):
    mat = 1/2 * (mat + mat.transpose())
    diag = np.eye(mat.shape[0])
    mat = mat * (1 - diag)
    return mat

def get_data(raw):
    data = raw.set_eeg_reference(ref_channels=['A1', 'A2']).notch_filter(
        freqs=50).filter(3, 70).resample(sfreq=200).drop_channels(['A1', 'A2']).get_data().clip(-0.0001, 0.0001)[None,
           :, :]
    ch_names = raw.info["ch_names"]
    sfreq = raw.info["sfreq"]

    # cut window
    x = []
    win_size = 0.6
    win_slide = 0.1
    for i in range(0, data.shape[2] - int(win_size * sfreq), int(win_slide*sfreq)):
        start_point = i
        end_point = i + int(win_size * sfreq)
        x.append(normalization(data[:, :, start_point: end_point]))
    x = np.concatenate(x, 0).astype(np.float32)

    return x

def attention_graph(X, title=None):
    """draw the graph map based on the attention index"""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
         'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Pz']

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color='red' if y[i] in ['T3', 'T5', 'F3', 'C3'] else 'blue',
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()

    return

def compute_test(x_test, y_test):
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
    return tp/(tp+fn)


def plot_chan_exp():
    GADN0 = np.array(
        [0.7318485523385301, 0.7483296213808464, 0.7701559020044544, 0.7790645879732739, 0.7946547884187083,
         0.8089086859688196, 0.8267260579064588, 0.8485523385300668, 0.866815144766147, 0.8948775055679288])
    GADN_Conv0 = np.array(
        [0.6552338530066815, 0.6552338530066815, 0.6552338530066815, 0.6552338530066815, 0.6552338530066815,
         0.6552338530066815, 0.6552338530066815, 0.6552338530066815, 0.6552338530066815, 0.6552338530066815])
    GADN1 = np.array(
        [0.7318485523385301, 0.7548693586698337, 0.7597736625514403, 0.7700626067159931, 0.7790332705586943,
         0.7945682451253482, 0.8006379585326954, 0.8061485909479078, 0.7831207065750736, 0.7900101936799184])
    GADN_Conv1 = np.array(
        [0.6552338530066815, 0.6717339667458432, 0.6712962962962963, 0.6881047239612976, 0.6949152542372882,
         0.6942896935933147, 0.7073365231259968, 0.6994022203245089, 0.6712463199214916, 0.6768603465851172])

    plt.figure()
    plt.axes(xlabel='channel delete num', ylabel='sensitivity')
    plt.plot(GADN0,label='GADN')
    plt.plot(GADN_Conv0, label='GADN_OnlyConv')
    plt.title('Delete background channels')
    plt.xticks(range(10))
    plt.legend()

    plt.figure()
    plt.axes(xlabel='channel delete num', ylabel='sensitivity')
    plt.plot(GADN1, label='GADN')
    plt.plot(GADN_Conv1, label='GADN_OnlyConv')
    plt.title('Delete spike channels')
    plt.xticks(range(10))
    plt.legend()
    plt.show()
    return

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

def cut_chan(num, x_test, y_test, cut_class=0):
    x_test_cut = np.asarray(x_test)
    y_test_cut = np.asarray(y_test)
    for i in range(num):
        idx_1 = y_test[:, i] == cut_class
        y_test_cut[idx_1, i] = 0
        x_test_cut[idx_1, i, :] = 0.0
    return x_test_cut, y_test_cut


def cut_chan_test(mode='GADN'):
    sensits = []
    for i in range(10):
        x_test_cut, y_test_cut = cut_chan(i, x_test, y_test)
        sensits.append(compute_test(x_test_cut,y_test_cut))
    print(sensits)

if __name__ == '__main__':
    # Load data
    X, Y = load_dataset(args.seed)
    split_rate = 0.8
    x_train = X[0:int(split_rate * X.shape[0])]
    x_test = X[int(split_rate * X.shape[0]):]

    y_train = Y[0:int(split_rate * Y.shape[0])]
    y_test = Y[int(split_rate * Y.shape[0]):]

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
    mode = 'GADN'  # define which test to run: GADN, svm, run_record, cut_channel_GADN, cut_channel_svm
    if mode == 'GADN':
        compute_test(x_test, y_test)
    elif mode == 'cut_channel_GADN':
        cut_chan_test(mode='GADN')
    elif mode == 'svm':
        clf = svm_run_and_test()
    elif mode == 'run_record':
        run_EEG_record(model)

    plot_chan_exp()