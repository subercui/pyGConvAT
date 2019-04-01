# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal 
import mne
from mne.io.constants import FIFF
from matplotlib import pyplot as plt
from matplotlib import patches
import time
import csv
import sys

class wave_class(object):
    def __init__(self):
        self.spike_pn = []
        self.sharp_pn = []
        self.biphasic_pn = []
        self.spike_np = []
        self.sharp_np = []
        self.biphasic_np = []
        self.triphasic_npn = []
        self.triphasic_pnp = []
        self.polyphasic = []
        
def wave_analyze(raw):
    correct_kind(raw)
    picks_EEG = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
    raw.filter(1/(2*np.pi*0.3)/2, 70.0, picks=picks_EEG, method='fir')
    raw_pattern, picks_pattern = apply_pattern(raw, picks_EEG)
#    raw_pattern.plot()
    wave_all, wave_dict_all, discharge_time, peak_dics_slow, stds_all = waveform_analyze(raw_pattern, picks_pattern)
    return wave_all, wave_dict_all, discharge_time, raw_pattern, picks_pattern, peak_dics_slow, stds_all
    
def waveform_analyze(raw, picks):
    sfreq = raw.info['sfreq']
    channel_names = np.array(raw.info['ch_names'])[picks]
    raw_slow = raw.copy()
    raw_slow.filter(0.1, 15.0, picks=picks, method='fir')
    raw_fast = raw.copy()
    raw_fast.filter(14.0, 70.0, picks=picks, method='fir')
    
    data_slow = raw_slow.get_data(picks)
    data_slow *= 1e6
    data_fast = raw_fast.get_data(picks)
    data_fast *= 1e6
    
    peak_inds_slow = data_maxima_ind(data_slow)
    peak_inds_fast = data_maxima_ind(data_fast)
    
    peak_dics_slow, stds_all = peak_classify(data_slow, sfreq, peak_inds_slow, std_factor=0.6, smooth_count=1)
    wave_all = find_wave(peak_dics_slow, stds_all, data_fast, peak_inds_fast, sfreq)
    wave_dict_all = generate_dict(wave_all, channel_names)
    discharge_time=generate_times(wave_all)
    
#    large_inds_slow = generate_large_peak_ind(peak_dics_slow)
#    plot_data(raw_slow, picks, plot_names=['F8-AV'], gap=150, start=84, stop=94, peak_ind=peak_inds_slow, large_peak_ind=large_inds_slow, show_detail=True, line_colors=['k'])
    return wave_all, wave_dict_all, discharge_time, peak_dics_slow, stds_all
    
def correct_kind(raw):
    chs = raw.info['chs']
    for i in range(len(chs)):
        kind = chs[i]['kind']
        name = chs[i]['ch_name']
        if kind is FIFF.FIFFV_EEG_CH:
            if (u'STI' in name) or (u'sti' in name):
                chs[i]['kind'] = FIFF.FIFFV_STIM_CH
                continue
            elif (u'EOG' in name) or (u'eog' in name):
                chs[i]['kind'] = FIFF.FIFFV_EOG_CH
                continue
            elif (u'EMG' in name) or (u'emg' in name):
                chs[i]['kind'] = FIFF.FIFFV_EMG_CH
                continue
            elif (u'ECG' in name) or (u'ecg' in name):
                chs[i]['kind'] = FIFF.FIFFV_ECG_CH
                continue
            elif (u'MISC' in name) or (u'misc' in name):
                chs[i]['kind'] = FIFF.FIFFV_MISC_CH
                continue
            elif (u'DC' in name) or (u'dc' in name) or (name==u'POL E') or (name==u'EEG E') or (name==u'E') or (name==u'e'):
                chs[i]['kind'] = FIFF.FIFFV_SYST_CH
            elif (u'Event' in name) or (u'event' in name) or (u'Marker' in name) or (u'marker' in name):
                chs[i]['kind'] = FIFF.FIFFV_MISC_CH
                continue
    raw.info['chs'] = chs
    return

def apply_pattern(raw, picks_EEG):
    name_list_AV = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz','T1','T2','A1','A2']
    further_pick_to_AV = further_pick_by_name(raw, picks_EEG, name_list_AV)
    picks_AV = picks_EEG[further_pick_to_AV]
    name_list_move_AV = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz','A1','A2']
    further_pick_to_move_AV = further_pick_by_name(raw, picks_EEG, name_list_move_AV)
    picks_move_AV = picks_EEG[further_pick_to_move_AV]

    '''
    pick_C3 = picks_EEG[further_pick_by_name(raw, picks_EEG, ['C3'])]
    pick_C4 = picks_EEG[further_pick_by_name(raw, picks_EEG, ['C4'])]
    pick_T1 = picks_EEG[further_pick_by_name(raw, picks_EEG, ['T1'])]
    pick_T2 = picks_EEG[further_pick_by_name(raw, picks_EEG, ['T2'])]
    '''
    
    data_AV = np.mean(raw.get_data(picks_AV),axis=0)
    data = raw.get_data(picks_move_AV)
    data_mat_AV = np.dot(np.array([data_AV]).T, np.ones((1,data.shape[0]))).T
    raw_new = raw.copy()
    raw_new._data[picks_move_AV] = -(data-data_mat_AV)
    add_name_tail(raw_new, picks_move_AV, '-AV')
    
    '''
    raw_new._data[pick_T1] = -(raw._data[pick_T1]-raw._data[pick_C3])
    add_name_tail(raw_new, pick_T1, '-C3')
    raw_new._data[pick_T2] = -(raw._data[pick_T2]-raw._data[pick_C4])
    add_name_tail(raw_new, pick_T2, '-C4')
    '''
    picks_new = picks_AV
    return raw_new, picks_new
    
def add_name_tail(raw, picks, tail):
    for i in picks:
        raw.info['ch_names'][i] += tail
        raw.info['chs'][i]['ch_name'] += tail
    return

def further_pick_by_name(raw, picks, channel_names):
    further_picks = []
    all_names = np.array(raw.info['ch_names'])[picks]
    for j in range(len(all_names)):
        for this_name in channel_names:
            if '*' in this_name:
                this_name_key = this_name.split('*')[0]
                if (this_name_key in all_names[j]) and (all_names[j].split(this_name_key)[0]=='') and len(this_name_key)<len(all_names[j]):
                    further_picks.append(j)
                    break
            else:
                if this_name == all_names[j]:
                    further_picks.append(j)
                    break
    return np.array(further_picks)

def data_maxima_ind(data):
    n_picks = data.shape[0]
    maxima_inds = []
    for i in range(n_picks):
        maxima_P_ind = np.array(signal.argrelmax(data[i,:]))[0]
        maxima_N_ind = np.array(signal.argrelmin(data[i,:]))[0]
        maxima_ind = sorted(maxima_P_ind.tolist() + maxima_N_ind.tolist())
        if len(maxima_ind) == 0:
            maxima_ind.append(0)
        maxima_inds.append(np.array(maxima_ind))
    return maxima_inds

def peak_classify(data, freq, peak_inds, std_factor=1., smooth_count=1):
    n_channel = int(data.shape[0])
    peak_dics_all = []
    stds_all = []
    
    for i in range(n_channel):
        inds = np.array(peak_inds[i])
        peaks = data[i,inds]
        inds = inds.tolist()
        peaks = peaks.tolist()
        
        for count in range(smooth_count):
            inds_to_pop = []
            dones = [j<-1 for j in range(len(peaks))]
            for ii in range(len(peaks)):
                if dones[ii]:
                    pass
                elif ii<len(peaks)-3:
                    p0 = peaks[ii]
                    p1 = peaks[ii+1]
                    p2 = peaks[ii+2]
                    p3 = peaks[ii+3]
                    p1_line = p0 + float(inds[ii+1]-inds[ii])*(p3-p0)/float(inds[ii+3]-inds[ii])
                    p2_line = p0 + float(inds[ii+2]-inds[ii])*(p3-p0)/float(inds[ii+3]-inds[ii])
                    if max(abs(p1-p1_line),abs(p2-p2_line))<0.15*abs(p3-p0) and min(p1,p2)>min(p0,p3) and max(p1,p2)<max(p0,p3):
                        inds_to_pop += [ii+1,ii+2]
                        dones[ii+1:ii+3] = [True,True]
            for ii in range(len(inds_to_pop)):
                inds.pop(inds_to_pop[ii]-ii)
                peaks.pop(inds_to_pop[ii]-ii)
        
        inds = np.array(inds)
        peaks = np.array(peaks)
        means, stds = mean_std_around(data[i,:], inds, freq)
        peak_dics = []
        for ii in range(len(peaks)):
            mean = means[ii]
            std = stds[ii]
            peak_dic = {}
            peak_dic['type'] = 3
            peak_dic['done'] = False
            peak_dic['ind'] = inds[ii]
            peak_dic['time'] = inds[ii]/float(freq)
            peak_dic['peak'] = peaks[ii]
            if peak_dic['peak'] > mean+std_factor*std:
                peak_dic['position'] = 1
            elif peak_dic['peak'] < mean-std_factor*std:
                peak_dic['position'] = 2
            else:
                peak_dic['position'] = 0
            if ii > 1: #and max([abs(peak_dics[-2]['peak']),abs(peak_dics[-1]['peak']),abs(peak_dic['peak'])])>100:
                if peak_dics[-2]['peak'] > peak_dics[-1]['peak']:
                    peak_dics[-1]['type'] = which_v(peak_dics[-2],peak_dics[-1],peak_dic)
                else:
                    peak_dics[-1]['type'] = which_n(peak_dics[-2],peak_dics[-1],peak_dic)
            peak_dics.append(peak_dic)
        peak_dics_all.append(peak_dics)
        stds_all.append(stds)
    return peak_dics_all, stds_all

def generate_large_peak_ind(peak_dics_all):
    large_peak_inds = []
    for i in range(len(peak_dics_all)):
        peak_dics = peak_dics_all[i]
        this_large_peak_inds = []
        for j in range(len(peak_dics)):
            if peak_dics[j]['position'] > 0:
                this_large_peak_inds.append(peak_dics[j]['ind'])
        large_peak_inds.append(np.array(this_large_peak_inds))
    return large_peak_inds

def mean_std_around(data_array, inds, freq, mean_range=1, std_range=2):
    around_range_mean, around_range_std = mean_range*int(freq), std_range*int(freq)
    around_range = max([around_range_mean,around_range_std])
    inds_head = list(range(around_range))
    inds_head.reverse()
    data_head = data_array[inds_head]
    inds_tail = np.array(inds_head)+len(data_array)-around_range
    data_tail = data_array[inds_tail]
    data_extend = np.append(np.append(data_head,data_array),data_tail)
    datas_to_mean = []
    datas_to_std = []
    for ind in inds:
        data_to_mean = data_extend[(ind-around_range_mean+around_range):(ind+around_range_mean+around_range+1)]
        data_to_std = data_extend[(ind-around_range_std+around_range):(ind+around_range_std+around_range+1)]
        datas_to_mean.append(data_to_mean)
        datas_to_std.append(data_to_std)
    means = np.mean(datas_to_mean,axis=1)
    stds = np.std(datas_to_std,axis=1)    
    return means, stds

def which_v(left,v,right):
    vtype = 0#'v_unknown'
    if v['position'] == 2:
        vtype = -1#'v_x2x'
    elif v['position'] == 1:
        vtype = -2#'v_101'
    else:
        if left['position'] == 0:
            if right['position'] == 0:
                vtype = -3#'v_000'
            elif right['position'] == 1:
                vtype = -4#'v_001'
        elif left['position'] == 1:
            if right['position'] == 0:
                vtype = -5#'v_100'
            elif right['position'] == 1:
                vtype = -2#'v_101'
    return vtype

def which_n(left,n,right):
    ntype = 0
    if n['position'] == 1:
        ntype = 1#'n_x1x'
    elif n['position'] == 2:
        ntype = 2#'n_202'
    else:
        if left['position'] == 0:
            if right['position'] == 0:
                ntype = 3#'n_000'
            elif right['position'] == 2:
                ntype = 4#'n_002'
        elif left['position'] == 2:
            if right['position'] == 0:
                ntype = 5#'n_200'
            elif right['position'] == 2:
                ntype = 2#'n_202'
    return ntype

def find_wave(peak_dics_all, stds_all, data_spike, inds_spike, freq=256.):
    n_channel = len(peak_dics_all)
    spike_limit = 70e-3*freq
    slow_limit = 200e-3*freq
    wave_all  = []
    for i in range(n_channel):
        wave_inst = wave_class()
        peaks_spike = data_spike[i,inds_spike[i]]
        ind_spike = inds_spike[i]
        dones = [j<-1 for j in range(len(peaks_spike))]
        for ii in range(len(peaks_spike)):
            amp_threshold = 100.
            try:
                if dones[ii]:
                    pass
                elif abs(peaks_spike[ii+1]-peaks_spike[ii])>amp_threshold and ii>0 and ii<len(peaks_spike)-1:
                    wave_inst.spike_pn.append(np.array((ind_spike[ii-1]/freq,ind_spike[ii+2]/freq))+analysis_start_time)
                    dones[ii] = True
                    dones[ii+1] = True
            except:
                Exception
                    
        peak_dics = peak_dics_all[i]
        stds = stds_all[i]
        for ii in range(len(peak_dics)):
            amp_threshold = max([2*2*stds[ii],0.5*100.])
            try:
                if peak_dics[ii]['done']:
                    pass
                else:                    
                    if (peak_dics[ii]['type']==-4 or peak_dics[ii]['type']==-2 or peak_dics[ii]['type']==-1) \
                    and (peak_dics[ii+1]['type']==1) \
                    and (peak_dics[ii+2]['type']==-1) \
                    and (peak_dics[ii+3]['type']==5 or peak_dics[ii+3]['type']==2 or (peak_dics[ii+3]['type']==1 and (abs(peak_dics[ii+3]['peak']-peak_dics[ii+2]['peak'])<0.7*abs(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak']) or abs(peak_dics[ii+3]['peak']-peak_dics[ii]['peak'])<0.4*abs(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak']) ) )) \
                    and (peak_dics[ii+3]['ind']-peak_dics[ii]['ind']<60*slow_limit):
                        if peak_dics[ii+3]['ind']-peak_dics[ii]['ind']<spike_limit and abs(peak_dics[ii+1]['peak']-peak_dics[ii]['peak'])>=amp_threshold:
                            wave_inst.spike_pn.append(np.array((peak_dics[ii]['ind']/freq,peak_dics[ii+3]['ind']/freq))+analysis_start_time)
                            peak_dics[ii]['done'] = True
                            peak_dics[ii+1]['done'] = True
                            peak_dics[ii+2]['done'] = True
                            continue
                        elif (peak_dics[ii+1]['ind']-peak_dics[ii]['ind']<slow_limit) and (peak_slope(peak_dics[ii],peak_dics[ii+1])/peak_slope(peak_dics[ii+2],peak_dics[ii+3])>2 or peak_slope(peak_dics[ii],peak_dics[ii+1])/peak_slope(peak_dics[ii+1],peak_dics[ii+2])<-2) and (peak_dics[ii+1]['peak']-peak_dics[ii]['peak'])/(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak'])>0.6*0 and abs(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak'])>=0.75*amp_threshold:#float((peak_dics[ii+1]['ind']-peak_dics[ii]['ind']))/float((peak_dics[ii+3]['ind']-peak_dics[ii]['ind']))<0.28 and (peak_dics[ii+1]['peak']-peak_dics[ii]['peak'])/(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak'])>0.6:
                            wave_inst.sharp_pn.append(np.array((peak_dics[ii]['ind']/freq,peak_dics[ii+3]['ind']/freq))+analysis_start_time)
                            peak_dics[ii]['done'] = True
                            peak_dics[ii+1]['done'] = True
                            peak_dics[ii+2]['done'] = True
                            continue
                        elif (peak_dics[ii+1]['ind']-peak_dics[ii]['ind']<slow_limit) and abs(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak'])>=0.75*amp_threshold and peak_dics[ii+3]['ind']-peak_dics[ii]['ind']>slow_limit and abs(peak_dics[ii]['peak']-peak_dics[ii+3]['peak'])/abs(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak'])<0.3 and 0.5*(peak_dics[ii]['peak']+peak_dics[ii+3]['peak'])>min(peak_dics[ii+1]['peak'],peak_dics[ii+2]['peak'])+0.3*abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak']) and 0.5*(peak_dics[ii]['peak']+peak_dics[ii+3]['peak'])<min(peak_dics[ii+1]['peak'],peak_dics[ii+2]['peak'])+0.7*abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak']):
                            wave_inst.biphasic_pn.append(np.array((peak_dics[ii]['ind']/freq,peak_dics[ii+3]['ind']/freq))+analysis_start_time)
                            peak_dics[ii]['done'] = True
                            peak_dics[ii+1]['done'] = True
                            peak_dics[ii+2]['done'] = True
                            continue
                        
                    if (peak_dics[ii]['type']==4 or peak_dics[ii]['type']==2 or peak_dics[ii]['type']==1) and (peak_dics[ii+1]['type']==-1) and (peak_dics[ii+2]['type']==1 and abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])>75*2) and (peak_dics[ii+3]['type']==-1):
                        if (((peak_dics[ii+4]['type']==5) or peak_dics[ii+4]['type']==3 or (peak_dics[ii+4]['type']==2)) or (peak_dics[ii+4]['type']==1 and abs(peak_dics[ii+4]['peak']-peak_dics[ii+3]['peak'])<0.8*abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak']))) and abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])>=0.75*amp_threshold and abs(peak_dics[ii]['peak']-peak_dics[ii+1]['peak'])/abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])>0.25 and abs(peak_dics[ii]['peak']-peak_dics[ii+1]['peak'])/abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])<0.75 and abs(peak_dics[ii+3]['peak']-peak_dics[ii+4]['peak'])/abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])>0.25 and abs(peak_dics[ii+3]['peak']-peak_dics[ii+4]['peak'])/abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])<0.75 and peak_dics[ii+3]['ind']-peak_dics[ii]['ind']>slow_limit and peak_dics[ii+1]['ind']-peak_dics[ii]['ind']>0.15*(peak_dics[ii+4]['ind']-peak_dics[ii]['ind']):
                            wave_inst.triphasic_npn.append(np.array((peak_dics[ii]['ind']/freq,peak_dics[ii+4]['ind']/freq))+analysis_start_time)
                            peak_dics[ii]['done'] = True
                            peak_dics[ii+1]['done'] = True
                            peak_dics[ii+2]['done'] = True
                            peak_dics[ii+3]['done'] = True
                            continue
                        elif (peak_dics[ii+1]['ind']-peak_dics[ii]['ind']<slow_limit) and peak_dics[ii+4]['type']==1 and abs(peak_dics[ii+4]['peak']-peak_dics[ii+3]['peak'])/abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])>0.6*0 and abs(peak_dics[ii+4]['peak']-peak_dics[ii+3]['peak'])>=0.75*amp_threshold and peak_dics[ii+3]['ind']-peak_dics[ii]['ind']>slow_limit:
                            peak_dics[ii]['done'] = True
                            iii = 0
                            while ((peak_dics[ii+iii+1]['type']==-1) or (peak_dics[ii+iii+1]['type']==1)) and (iii<2 or abs(peak_dics[ii+iii+1]['peak']-peak_dics[ii+iii]['peak'])>0.6*0*abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])):
                                peak_dics[ii+iii+1]['done'] = True
                                iii = iii+1
                            wave_inst.polyphasic.append(np.array((peak_dics[ii]['ind']/freq,peak_dics[ii+iii+1]['ind']/freq))+analysis_start_time)
                            continue
                            
                    if (peak_dics[ii]['type']==4 or peak_dics[ii]['type']==2 or peak_dics[ii]['type']==1) and (peak_dics[ii+1]['type']==-1) and (peak_dics[ii+2]['type']==1) and (peak_dics[ii+3]['type']==-5 or peak_dics[ii+3]['type']==-2 or (peak_dics[ii+3]['type']==-1 and (abs(peak_dics[ii+3]['peak']-peak_dics[ii+2]['peak'])<0.7*abs(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak']) or abs(peak_dics[ii+3]['peak']-peak_dics[ii]['peak'])<0.4*abs(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak']) ) )) and (peak_dics[ii+3]['ind']-peak_dics[ii]['ind']<60*slow_limit):
                        if peak_dics[ii+3]['ind']-peak_dics[ii]['ind']<spike_limit and abs(peak_dics[ii+1]['peak']-peak_dics[ii]['peak'])>=amp_threshold:
                            wave_inst.spike_np.append(np.array((peak_dics[ii]['ind']/freq,peak_dics[ii+3]['ind']/freq))+analysis_start_time)
                            peak_dics[ii]['done'] = True
                            peak_dics[ii+1]['done'] = True
                            peak_dics[ii+2]['done'] = True
                            continue
                        elif (peak_dics[ii+1]['ind']-peak_dics[ii]['ind']<slow_limit) and (peak_slope(peak_dics[ii],peak_dics[ii+1])/peak_slope(peak_dics[ii+2],peak_dics[ii+3])>2 or peak_slope(peak_dics[ii],peak_dics[ii+1])/peak_slope(peak_dics[ii+1],peak_dics[ii+2])<-2) and (peak_dics[ii+1]['peak']-peak_dics[ii]['peak'])/(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak'])>0.6*0 and abs(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak'])>=0.75*amp_threshold:#float((peak_dics[ii+1]['ind']-peak_dics[ii]['ind']))/float((peak_dics[ii+3]['ind']-peak_dics[ii]['ind']))<0.28 and (peak_dics[ii+1]['peak']-peak_dics[ii]['peak'])/(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak'])>0.6:
                            wave_inst.sharp_np.append(np.array((peak_dics[ii]['ind']/freq,peak_dics[ii+3]['ind']/freq))+analysis_start_time)
                            peak_dics[ii]['done'] = True
                            peak_dics[ii+1]['done'] = True
                            peak_dics[ii+2]['done'] = True
                            continue
                        elif (peak_dics[ii+1]['ind']-peak_dics[ii]['ind']<slow_limit) and abs(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak'])>=0.75*amp_threshold and peak_dics[ii+3]['ind']-peak_dics[ii]['ind']>slow_limit and abs(peak_dics[ii]['peak']-peak_dics[ii+3]['peak'])/abs(peak_dics[ii+1]['peak']-peak_dics[ii+2]['peak'])<0.3 and 0.5*(peak_dics[ii]['peak']+peak_dics[ii+3]['peak'])>min(peak_dics[ii+1]['peak'],peak_dics[ii+2]['peak'])+0.3*abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak']) and 0.5*(peak_dics[ii]['peak']+peak_dics[ii+3]['peak'])<min(peak_dics[ii+1]['peak'],peak_dics[ii+2]['peak'])+0.7*abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak']):
                            wave_inst.biphasic_np.append(np.array((peak_dics[ii]['ind']/freq,peak_dics[ii+3]['ind']/freq))+analysis_start_time)
                            peak_dics[ii]['done'] = True
                            peak_dics[ii+1]['done'] = True
                            peak_dics[ii+2]['done'] = True
                            continue
                    
                    if (peak_dics[ii]['type']==-4 or peak_dics[ii]['type']==-2 or peak_dics[ii]['type']==-1) and (peak_dics[ii+1]['type']==1) and (peak_dics[ii+2]['type']==-1 and abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])>75*2) and (peak_dics[ii+3]['type']==1):
                        if (((peak_dics[ii+4]['type']==-5) or peak_dics[ii+4]['type']==-3 or (peak_dics[ii+4]['type']==-2)) or (peak_dics[ii+4]['type']==-1 and abs(peak_dics[ii+4]['peak']-peak_dics[ii+3]['peak'])<0.8*abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak']))) and abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])>=0.75*amp_threshold and abs(peak_dics[ii]['peak']-peak_dics[ii+1]['peak'])/abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])>0.25 and abs(peak_dics[ii]['peak']-peak_dics[ii+1]['peak'])/abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])<0.75 and abs(peak_dics[ii+3]['peak']-peak_dics[ii+4]['peak'])/abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])>0.25 and abs(peak_dics[ii+3]['peak']-peak_dics[ii+4]['peak'])/abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])<0.75 and peak_dics[ii+3]['ind']-peak_dics[ii]['ind']>slow_limit and peak_dics[ii+1]['ind']-peak_dics[ii]['ind']>0.15*(peak_dics[ii+4]['ind']-peak_dics[ii]['ind']):
                            wave_inst.triphasic_pnp.append(np.array((peak_dics[ii]['ind']/freq,peak_dics[ii+4]['ind']/freq))+analysis_start_time)
                            peak_dics[ii]['done'] = True
                            peak_dics[ii+1]['done'] = True
                            peak_dics[ii+2]['done'] = True
                            peak_dics[ii+3]['done'] = True
                            continue
                        elif (peak_dics[ii+1]['ind']-peak_dics[ii]['ind']<slow_limit) and peak_dics[ii+4]['type']==-1 and abs(peak_dics[ii+4]['peak']-peak_dics[ii+3]['peak'])/abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])>0.6*0 and abs(peak_dics[ii+4]['peak']-peak_dics[ii+3]['peak'])>=0.75*amp_threshold and peak_dics[ii+3]['ind']-peak_dics[ii]['ind']>slow_limit:
                            peak_dics[ii]['done'] = True
                            iii = 0
                            while ((peak_dics[ii+iii+1]['type']==1) or (peak_dics[ii+iii+1]['type']==-1)) and (iii<2 or abs(peak_dics[ii+iii+1]['peak']-peak_dics[ii+iii]['peak'])>0.6*0*abs(peak_dics[ii+2]['peak']-peak_dics[ii+1]['peak'])):
                                peak_dics[ii+iii+1]['done'] = True
                                iii = iii+1
                            wave_inst.polyphasic.append(np.array((peak_dics[ii]['ind']/freq,peak_dics[ii+iii+1]['ind']/freq))+analysis_start_time)
                            continue
            except:
                Exception
        wave_all.append(wave_inst)
    return wave_all

def peak_slope(peak_dic_1, peak_dic_2):
    x_1 = float(peak_dic_1['ind'])
    x_2 = float(peak_dic_2['ind'])
    y_1 = peak_dic_1['peak']
    y_2 = peak_dic_2['peak']
    slope = (y_2-y_1)/(x_2-x_1)
    return slope

def generate_dict(wave_all, channel_names):
    wave_dict_all = {}
    for i in range(len(wave_all)):
        key = channel_names[i]
        wave_inst = wave_all[i]
        wave_dict = {}
        wave_dict['spike'] = wave_inst.spike_pn + wave_inst.spike_np
        wave_dict['sharp'] = wave_inst.sharp_pn + wave_inst.sharp_np
        wave_dict['biphasic'] = wave_inst.biphasic_pn + wave_inst.biphasic_np
        wave_dict['triphasic'] = wave_inst.triphasic_npn + wave_inst.triphasic_pnp
        wave_dict['polyphasic'] = wave_inst.polyphasic
        wave_dict_all[key] = wave_dict
    return wave_dict_all

def generate_times(wave_all):
    discharge_time = []
    for i in range(len(wave_all)):
        this_wave = wave_all[i]
        this_time = this_wave.sharp_pn + this_wave.sharp_np# + this_wave.spike_pn + this_wave.spike_np
        discharge_time.append(this_time)
    return discharge_time
    
def plot_data(raw, picks, plot_names, gap=1000, start=0., stop=None, peak_ind=None, threshold=None, large_peak_ind=None, HFO_peak_ind=None, show_detail=False, HFO_time=None, line_colors=['b']):
    if raw._times[-1] < original_total_time:
        need_shift = True
    else:
        need_shift = False
    further_picks = further_pick_by_name(raw, picks, plot_names)    
    temp_raw = raw.copy()
    if need_shift:
        try:
            temp_raw.crop(start-analysis_start_time, stop-analysis_start_time)
        except Exception:
            print('\033[0;31mData (or part of data) at [%.2fs, %.2fs] has not been filtered.\033[0m'%(start,stop))
            return
        data, times = temp_raw[picks[further_picks]]
    else:
        temp_raw.crop(start, stop)
        data, times = temp_raw[picks[further_picks]]
    sfreq = raw.info['sfreq']
    names = np.array(raw.info['ch_names'])[picks[further_picks]]
    data = data*1e6
    n_plots = len(further_picks)
    if show_detail and (peak_ind is not None):
        mark_inds = []
        for i in range(n_plots):
            mark_ind = peak_ind[further_picks[i]]-(temp_raw.first_samp-raw.first_samp)
            mark_ind = mark_ind[np.where(mark_ind>=0)]
            mark_ind = mark_ind[np.where(mark_ind<data.shape[1])]
            mark_inds.append(mark_ind)
        if large_peak_ind is not None:
            large_mark_inds = []
            for i in range(n_plots):
                large_mark_ind = large_peak_ind[further_picks[i]]-(temp_raw.first_samp-raw.first_samp)
                large_mark_ind = large_mark_ind[np.where(large_mark_ind>=0)]
                large_mark_ind = large_mark_ind[np.where(large_mark_ind<data.shape[1])]
                large_mark_inds.append(large_mark_ind)
        if HFO_peak_ind is not None:
            HFO_mark_inds = []
            for i in range(n_plots):
                temp_HFO_ind = np.hstack(HFO_peak_ind[further_picks[i]]) if len(HFO_peak_ind[further_picks[i]])>0 else []
                HFO_mark_ind = peak_ind[further_picks[i]][temp_HFO_ind]-(temp_raw.first_samp-raw.first_samp)
                HFO_mark_ind = HFO_mark_ind[np.where(HFO_mark_ind>=0)]
                HFO_mark_ind = HFO_mark_ind[np.where(HFO_mark_ind<data.shape[1])]
                HFO_mark_inds.append(HFO_mark_ind)
    if (not show_detail) and (HFO_time is not None):
        HFO_mark_inds = []
        for i in range(n_plots):
            if need_shift:
                temp_HFO_ind = (sfreq*(np.hstack(HFO_time[further_picks[i]])-analysis_start_time)).astype(int) if len(HFO_time[further_picks[i]])>0 else []
            else:
                temp_HFO_ind = (sfreq*np.hstack(HFO_time[further_picks[i]])).astype(int) if len(HFO_time[further_picks[i]])>0 else []
            HFO_mark_ind = temp_HFO_ind - (temp_raw.first_samp-raw.first_samp)
            HFO_mark_ind = HFO_mark_ind[np.where(HFO_mark_ind>=0)]
            HFO_mark_ind = HFO_mark_ind[np.where(HFO_mark_ind<data.shape[1])]
            HFO_mark_inds.append(HFO_mark_ind)

    plt.figure(figsize=[14, 1.*n_plots])
    data_plot = plt.gca()
    if stop is None:
        stop = start+times[-1]
    data_plot.axis(xmin=start, xmax=stop, ymin=-0.5*gap, ymax=(n_plots-0.5)*gap)
    data_plot.set_yticks([]) 
    for i in range(n_plots):
        line_color = line_colors[i%len(line_colors)]
        data_plot.plot(times+start, data[i]+gap*(n_plots-1-i), color=line_color)
        data_plot.plot(times+start, 0*data[i]+gap*(n_plots-1-i-0.5), color='k', linewidth=0.2)
        data_plot.text(start, gap*(n_plots-1-i), names[i], horizontalalignment='right', verticalalignment='center', color=line_color)
        if (peak_ind is not None) and show_detail:
            data_plot.plot(times[mark_inds[i]]+start, data[i][mark_inds[i]]+gap*(n_plots-1-i),'c.')
        if (threshold is not None) and show_detail:
            data_plot.plot(times+start, 0*data[i]+gap*(n_plots-1-i)+threshold[further_picks[i]]*1e6, 'g--', linewidth=0.5)
            data_plot.plot(times+start, 0*data[i]+gap*(n_plots-1-i)-threshold[further_picks[i]]*1e6, 'g--', linewidth=0.5)
        if (large_peak_ind is not None) and show_detail:
            data_plot.plot(times[large_mark_inds[i]]+start, data[i][large_mark_inds[i]]+gap*(n_plots-1-i),'r.')
        if (HFO_peak_ind is not None) or (HFO_time is not None):
            for j in range(len(HFO_mark_inds[i])):
                data_plot.plot(np.ones(10)*times[HFO_mark_inds[i][j]]+start, data[i][HFO_mark_inds[i][j]]+np.linspace(gap*(n_plots-1.3-i),gap*(n_plots-0.7-i),10),'m', linewidth=2)
    data_plot.set_xlabel('time [s]')
    data_plot.text(start, -gap, 'spacing between channels: %.0f$\mu V$'%(gap), horizontalalignment='left', verticalalignment='center')
    plt.show()
    return data_plot

def key_for_names(pick_names, i):
    name = pick_names[i].split(' ')[-1].split('-')[0]
    tail = pick_names[i].split(' ')[-1].split('-')[-1] if len(pick_names[i].split(' ')[-1].split('-'))>1 else ""
    if "'" in name:
        alpha, number = name[:2], int(name[2:])
    else:
        alpha, number = name[:1], int(name[1:])
    key = alpha + chr(number) + tail
    return key

def get_hit_ratio(discharge_time, labels, record_start, tolerance=0.1):
    hit_sum = 0
    for i in np.arange(1,len(labels)):
        label_i = 60.*float(labels[i][0]) + float(labels[i][1]) - record_start
        hit = False
        for j in range(len(discharge_time)):
            this_discharge_time = discharge_time[j]
            for k in range(len(this_discharge_time)):
                if label_i >= this_discharge_time[k][0]-tolerance and label_i <= this_discharge_time[k][1]+tolerance:
                    hit = True
                    break
            if hit:
                hit_sum += 1
                break
    hit_ratio = float(hit_sum)/(len(labels)-1)
    return hit_ratio, hit_sum
    
if __name__ == "__main__":
    print("Welcome to the WAVE analysis system (beta)!")
    mne.utils.logger.setLevel(mne.utils.logging.CRITICAL)
    
    DATA_LOADED = False
    TIME_SPECIFIED = False
    START_TIME_SPECIFIED = False
    END_TIME_SPECIFIED = False
    data_path = input("\033[0;32mEnter the full path of the .edf file:\n\033[0m")
#    data_path = r'D:\Brain\source\case\wangshuqi_1\wangshuqi_1.edf'
    
    while not DATA_LOADED:
        try:
            if data_path == 'exit':
                sys.exit("WAVE analyis terminated.")
            print("\nLoading data...")
            raw = mne.io.read_raw_edf(data_path, preload=True)
        except Exception as e:
            data_path = input("\033[0;32mNot a valid .edf file, please re-enter the full path of the .edf file:\n\033[0m")
        else:
            DATA_LOADED = True
    
    raw_to_analyze = raw.copy()
    start_time_str = input("\033[0;32mSpecify the start time in seconds (from [0.00, %.2f]) for analysis:\n\033[0m"%raw._times[-1])
    while not START_TIME_SPECIFIED:
        try:
            if start_time_str == 'exit':
                sys.exit("WAVE analyis terminated.")
            start_time = float(start_time_str)
            raw_to_analyze.copy().crop(start_time, raw._times[-1])
        except Exception as e:
            start_time_str = input("\033[0;32mNot valid start time, please re-enter the start time in seconds (from [0.00, %.2f]) for analysis:\n\033[0m"%raw._times[-1])
        else:
            START_TIME_SPECIFIED = True
            
    end_time_str = input("\033[0;32mSpecify the end time in seconds (from [%.2f, %.2f]) for analysis:\n\033[0m"%(start_time,raw._times[-1]))
    while not END_TIME_SPECIFIED:
        try:
            if end_time_str == 'exit':
                sys.exit("WAVE analyis terminated.")
            end_time = float(end_time_str)
            raw_to_analyze.crop(start_time, end_time)
        except Exception as e:
            end_time_str = input("\033[0;32mNot valid end time, please re-enter the start time in seconds (from [%.2f, %.2f]) for analysis:\n\033[0m"%(start_time,raw._times[-1]))
        else:
            END_TIME_SPECIFIED = True
            
    global analysis_start_time, analysis_end_time, original_total_time
    analysis_start_time = start_time
    analysis_end_time = end_time
    original_total_time = raw._times[-1]
    
    print("\nStarting to analyze...")
    time_start_find = time.time()
#    wave_dict_all = find_waveform.waveform_analysis_direct(raw_to_analyze)
    wave_all, wave_dict_all, discharge_time, raw_pattern, picks_pattern, peak_dics_slow, stds_all = wave_analyze(raw_to_analyze)
    time_end_find = time.time()

    labels = []
    try:    
        with open(data_path[:-4]+'_labels.csv', 'rb') as csvfile:
            spamreader = csv.reader(csvfile,dialect='excel')
            for row in spamreader:
                labels.append(row)
    except:
        print ("\033[0;31mFailed to open " + data_path[:-4] + "_labels.csv.\033[0m")
    
    record_start = 8*60.+8
    hit_ratio, hit_sum = get_hit_ratio(discharge_time, labels, record_start)
    print("WAVE analysis finished in %.3f seconds for data at [%.2f, %.2f]s. Hit ratio is %.2f%% (%d out of %d)."%(time_end_find-time_start_find, analysis_start_time, analysis_end_time, 100*hit_ratio, hit_sum, len(labels)-1))
    
    print ("\033[0;36m\nNow the analyzed data can be plotted with the following code:\nplot_data(raw_pattern, picks_pattern, plot_names=np.array(raw_pattern.info['ch_names'])[picks_pattern], gap=100, start=80, stop=90, HFO_time=discharge_time, line_colors=['c','r'])\033[0m")
    
#D:\Brain\source\case\wangshuqi_1\wangshuqi_1.edf
#D:\Brain\source\case\wangshuqi_2\wangshuqi.edf
    
#i = 11
#ii = 1828
#peak_dics = peak_dics_slow[i]
#amp_threshold = max([2*2.3*stds_all[i][ii],0.5*100.])