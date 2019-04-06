import mne
import matplotlib.pyplot as plt
import numpy as np


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
    return data

def loc_discharge(raw, annotations):
    locs = []
    if annotations:
        data = raw.get_data()
        data_annotation = data[44,:]
        for annotation in annotations:
            locs = locs + list(np.argwhere(data_annotation == annotation).reshape(-1))
    return locs

file_list = ["DA0570A0_1-1+.edf",
             "DA0570A4_1-1+.edf", "DA0570A5_1-1+.edf", "DA0570A6_1-1+.edf", "DA0570A7_1-1+.edf",
             "DA05709X_1-1+.edf", "DA05709Z_1-1+.edf", "DA10104A_1-1+.edf",
             "DA10104D_1-1+.edf", "DA10104E_1-1+.edf",
             "DA10104F_1-1+.edf", "DA10104G_1-1+.edf", "DA10104H_1-1+.edf", "DA10104I_1-1+.edf",
             "DA10104J_1-1+.edf", "DA101049_1-1+.edf"]

             # DA10104A_1 is OK; DA05709R_1-1+.edf SOZ is on C4 F8; DA0570A6 is suitable for graph analysis
             # Delete noisy data: "DA0570A1_1-1+.edf",[5], DA10104B_1-1+.edf [4],No label data: "DA0570A2_1-1+.edf",[3], "DA0570A3_1-1+.edf",[], loss label data: DA05709R_1-1+.edf [3], DA10104C_1-1+.edf,[3,4],

annotation_list = [[4], [3], [3], [4], [3], [4], [3], [3,4],
                   [3], [4], [3,4], [4], [4], [3], [3], [3,4]] # only s and S included

x_pos_list = []
y_pos_list = []
x_neg_list = []
y_neg_list = []
for file, annotations in zip(file_list, annotation_list):
    raw = mne.io.read_raw_edf(file, preload=True)
    locs = loc_discharge(raw, annotations)
    raw = raw_eeg_pick(raw)  # now only EEG channels
    fig = raw.copy().set_eeg_reference(ref_channels=['A1', 'A2']).pick_types(meg=False, eeg=True).notch_filter(
        freqs=50).filter(3, 70).resample(sfreq=200).plot()
    print(file)
    # np.savetxt(fname=file[:-4]+'label.txt', X=locs, fmt='%d')