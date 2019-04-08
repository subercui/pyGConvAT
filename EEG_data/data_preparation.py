# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:29:16 2018

@author: think
"""
from __future__ import division
import mne
import matplotlib.pyplot as plt
import numpy as np
import run_waveform_py2

def plot_data(x, index, style = 'merge'):
    if style == 'merge':

        for i in range(x.shape[1]):
            plt.plot(x[index, i, :])

    elif style == 'separate':

        plt.figure(figsize = [3,14])
        for i in range(x.shape[1]):
            plt.subplot(x.shape[1], 1, i+1)
            plt.plot(x[index, i, :])
            plt.axis([0, x.shape[2], -1, 2])

def raw_eeg_pick(raw):
    ch_names = raw.info["ch_names"]
    drop_ch_list = []
    marker_list = ['T1', 'T2', 'STI', 'EMG', 'ECG', 'X', 'DC', 'Pulse', 'Wave', 'Mark', 'Sp', 'SP', 'EtCO', 'E', 'Cz']
    # For some trials, Cz is problematic#
    for ch_name in ch_names:
        for marker in marker_list:
            if marker in ch_name:
                drop_ch_list.append(ch_name)
                break
    raw.drop_channels(drop_ch_list)
    return raw

def minus_av(data):
    data_av = np.mean(data, axis = 1)
    data_minus_av = data - data_av
    return data_minus_av

def normalization(data):
    # subercui: should not minus mean again here
    # for i in range(data.shape[1]):
    #     mean = np.mean(data[0,i,:])
    #     data[0,i,:] = data[0,i,:] - mean
    normalizer = 0.0001
    data = data / normalizer
    data = data / np.std(data)
    return data

def loc_discharge(raw, annotations):
    locs = []
    if annotations:
        data = raw.get_data()
        data_annotation = data[44,:]
        for annotation in annotations:
            locs = locs + list(np.argwhere(data_annotation == annotation).reshape(-1))
    return locs  # locs is a list


def read_label(fname):
    locs = []
    labels = {}
    with open(fname,'r') as f:
        lines = f.readlines()
        for line in lines:
            entries = line.strip().split()
            loc = int(entries[0])
            labels[loc] = entries[1:]  # e.g. {130: 'F8', 'T4', 'T5'}
            locs.append(loc)
    return locs, labels


def translate(ch_names, label_list):
    label_array = np.zeros(len(ch_names))
    for label in label_list:
        idx = ch_names.index(label)
        label_array[idx] = 1
    return label_array


def get_data(raw, locs, labels):
    x_pos = []
    y_pos = []
    x_neg = []
    y_neg = []

    _, wave_dict_all, _, _, _, _, _ = run_waveform_py2.wave_analyze(raw)

    #raw.filter(3, 70.0, method='fir')
    # data = raw[:][0][None, :, :].clip(-0.0001, 0.0001)  # (1, channels, times)
    # data = minus_av(data)
    ori_sfreq = raw.info["sfreq"]
    data = raw.set_eeg_reference(ref_channels=['A1','A2']).notch_filter(
        freqs=50).filter(3, 70).resample(sfreq=200).drop_channels(['A1','A2']).get_data().clip(-0.0001, 0.0001)[None, :, :]
    ch_names = raw.info["ch_names"]
    sfreq = raw.info["sfreq"]

    # data_waveform = np.zeros(data.shape)
    # for index, ch_name in enumerate(ch_names):
    #     sharp_list = wave_dict_all[ch_name+'-AV']['sharp']
    #     for sharp_time in sharp_list:
    #         sharp_start, sharp_end = sharp_time
    #         sharp_start_point = int(sharp_start * sfreq)
    #         sharp_end_point = int(sharp_end * sfreq)
    #         data_waveform[0, index, sharp_start_point:sharp_end_point] = 1

    data = data[:,:,:,None]
    # data_waveform = data_waveform[:,:,:,None]
    # data = np.concatenate([data, data_waveform], axis = 3)

    num_pos = 9
    start_bound = 0
    end_bound = data.shape[2]
    win_size = 0.6
    loc_dis = 0.2

    for loc in locs:
        rloc = loc / ori_sfreq * sfreq  # the relative loc after changing sfreq
        for start_point in np.linspace(int(rloc-0.45*sfreq), int(rloc-0.15*sfreq), num = num_pos):
            start_point = min(end_bound - int(win_size*sfreq), max(start_bound, int(start_point)))
            end_point = start_point + int(win_size*sfreq)

            x_pos.append(normalization(data[:, :, start_point : end_point, :]))
            y_pos.append(translate(ch_names, labels[loc]))

    PNRATIO = 1
    for i in range(data.shape[2] - int(win_size*sfreq)):
        s = np.random.randint(data.shape[2] - int(win_size*sfreq))
        ACCEPT = True
        for loc in locs:
            rloc = loc / ori_sfreq * sfreq  # the relative loc after changing sfreq
            if (rloc > (s - loc_dis*sfreq)) and (rloc < (s + (win_size+loc_dis)*sfreq)):
                ACCEPT = False
        if ACCEPT:
            start_point = s
            end_point = s + int(win_size*sfreq)
            x_neg.append(normalization(data[:, :, start_point : end_point, :]))
            y_neg.append(np.zeros(len(ch_names)))
            if len(x_neg) >= PNRATIO * len(x_pos): break  # control the ratio of pos to neg samples

    x_pos = np.concatenate(x_pos, 0).astype(np.float32)
    y_pos = np.stack(y_pos, 0).astype(np.int64)

    x_neg = np.concatenate(x_neg, 0).astype(np.float32)
    y_neg = np.stack(y_neg, 0).astype(np.int64)

    assert len(x_pos) == len(y_pos)
    assert len(x_neg) == len(y_neg)
    return x_pos, y_pos, x_neg, y_neg

file_list = ["DA0570A0_1-1+.edf",
             "DA0570A4_1-1+.edf", "DA0570A5_1-1+.edf", "DA0570A6_1-1+.edf", "DA0570A7_1-1+.edf",
             "DA05709X_1-1+.edf", "DA05709Z_1-1+.edf", "DA10104A_1-1+.edf",
             "DA10104D_1-1+.edf", "DA10104E_1-1+.edf",
             "DA10104F_1-1+.edf", "DA10104G_1-1+.edf", "DA10104H_1-1+.edf", "DA10104I_1-1+.edf",
             "DA10104J_1-1+.edf", "DA101049_1-1+.edf"]

x_pos_list = []
y_pos_list = []
x_neg_list = []
y_neg_list = []
for file in file_list:
    raw = mne.io.read_raw_edf(file, preload=True)
    raw = raw_eeg_pick(raw)  # now only EEG channels
    locs, labels = read_label(fname=file[:-4]+'label.txt')  # locs is a list; labels is a dict.
    if locs:
        x_pos_once, y_pos_once, x_neg_once, y_neg_once = get_data(raw, locs, labels)
        print(file, x_pos_once.shape, x_neg_once.shape)
        x_pos_list.append(x_pos_once)
        y_pos_list.append(y_pos_once)
        x_neg_list.append(x_neg_once)
        y_neg_list.append(y_neg_once)
x_pos = np.concatenate(x_pos_list, axis = 0)
del x_pos_list
y_pos = np.concatenate(y_pos_list, axis = 0)
del y_pos_list
x_neg = np.concatenate(x_neg_list, axis = 0)
del x_neg_list
y_neg = np.concatenate(y_neg_list, axis = 0)
del y_neg_list

np.savez('dataset_aug.npz', x_pos, y_pos, x_neg, y_neg)
