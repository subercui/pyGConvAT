# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:29:16 2018

@author: think
"""
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
    for i in range(data.shape[1]):
        mean = np.mean(data[0,i,:])
        data[0,i,:] = data[0,i,:] - mean
    normalizer = 0.0001
    data = data / normalizer
    return data

def loc_discharge(raw, annotations):
    locs = []
    if annotations:
        data = raw.get_data()
        data_annotation = data[44,:]
        for annotation in annotations:
            locs = locs + list(np.argwhere(data_annotation == annotation).reshape(-1))  
    return locs
    
def get_data(raw, locs):
    x_pos = []
    x_neg = []
    
    _, wave_dict_all, _, _, _, _, _ = run_waveform_py2.wave_analyze(raw)
    
    raw.filter(3, 70.0, method='fir')
    data = raw[:][0][None, :, :].clip(-0.0001, 0.0001) 
    data = minus_av(data)
    ch_names = raw.info["ch_names"]
    sfreq = raw.info["sfreq"]

    data_waveform = np.zeros(data.shape)
    for index, ch_name in enumerate(ch_names):
        sharp_list = wave_dict_all[ch_name+'-AV']['sharp']
        for sharp_time in sharp_list:
            sharp_start, sharp_end = sharp_time
            sharp_start_point = int(sharp_start * sfreq)
            sharp_end_point = int(sharp_end * sfreq)
            data_waveform[0, index, sharp_start_point:sharp_end_point] = 1
    
    data = data[:,:,:,None]
    data_waveform = data_waveform[:,:,:,None]
    data = np.concatenate([data, data_waveform], axis = 3)
    
    num_pos = 11
    start_bound = 0
    end_bound = data.shape[2]
    
    for loc in locs:
        if loc > start_bound + 300 and loc < end_bound - 300:
            for start_point in np.linspace(loc-300, loc-100, num = num_pos):  
                start_point = int(start_point)
                end_point = start_point + 400
                
                x_pos.append(normalization(data[:, :, start_point : end_point, :]))
    
    rej_rate = 0.975
    for i in range(data.shape[2] - 400):
        ACCEPT = np.random.random() > rej_rate
        for loc in locs:
            if (loc > i - 100) and (loc < i + 500):
                ACCEPT = False
        if ACCEPT == True:
            start_point = i
            end_point = i + 400
            x_neg.append(normalization(data[:, :, start_point : end_point, :]))
            
    x_pos = np.concatenate(x_pos, 0).astype(np.float32)
    y_pos = np.ones(x_pos.shape[0]).astype(np.int64)
    
    x_neg = np.concatenate(x_neg, 0).astype(np.float32)
    y_neg = np.zeros(x_neg.shape[0]).astype(np.int64)
    return x_pos, y_pos, x_neg, y_neg

file_list = ["DA0570A0_1-1+.edf", "DA0570A1_1-1+.edf", "DA0570A2_1-1+.edf", "DA0570A3_1-1+.edf",
             "DA0570A4_1-1+.edf", "DA0570A5_1-1+.edf", "DA0570A6_1-1+.edf", "DA0570A7_1-1+.edf",
             "DA05709R_1-1+.edf", "DA05709X_1-1+.edf", "DA05709Z_1-1+.edf", "DA10104A_1-1+.edf",
             "DA10104B_1-1+.edf", "DA10104C_1-1+.edf", "DA10104D_1-1+.edf", "DA10104E_1-1+.edf",
             "DA10104F_1-1+.edf", "DA10104G_1-1+.edf", "DA10104H_1-1+.edf", "DA10104I_1-1+.edf",
             "DA10104J_1-1+.edf", "DA101049_1-1+.edf"]
             
             # DA10104A_1 is very noisy
             
annotation_list = [[4], [5], [3], [], [3], [3], [4], [3], [3], [4], [3], [3,4], [4],
                   [3,4], [3], [4], [3,4], [4], [4], [3], [3], [3,4]] # only s and S included

x_pos_list = []
y_pos_list = []       
x_neg_list = []
y_neg_list = []               
for file, annotations in zip(file_list, annotation_list):
    raw = mne.io.read_raw_edf(file, preload=True)
    locs = loc_discharge(raw, annotations)
    raw = raw_eeg_pick(raw)  # now only EEG channels
    if locs:
        x_pos_once, y_pos_once, x_neg_once, y_neg_once = get_data(raw, locs) 
        print(x_pos_once.shape)
        x_pos_list.append(x_pos_once)
        y_pos_list.append(y_pos_once)
        x_neg_list.append(x_neg_once)
        y_neg_list.append(y_neg_once)
x_pos = np.concatenate(x_pos_list, axis = 0)
y_pos = np.concatenate(y_pos_list, axis = 0)
x_neg = np.concatenate(x_neg_list, axis = 0)
y_neg = np.concatenate(y_neg_list, axis = 0)
    
np.savez('dataset1.npz', x_pos, y_pos, x_neg, y_neg)
