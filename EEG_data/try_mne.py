import mne

fig = raw.copy().pick_types(meg=False, eeg=True).resample(sfreq=150).filter(3, 70).plot()

raw.info['ch_names']

fig = raw.copy().set_eeg_reference(ref_channels='average').pick_types(meg=False, eeg=True).resample(sfreq=150).filter(3, 70).plot()

# this is by far the best
fig = raw.copy().set_eeg_reference(ref_channels=['A1','A2']).pick_types(meg=False, eeg=True).resample(sfreq=250).filter(3, 70).plot()

fig = raw.copy().set_eeg_reference(
    ref_channels=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'A1', 'A2', 'T1', 'T2']
).pick_types(meg=False, eeg=True).resample(sfreq=150).filter(3, 70).plot()



raw.copy().get_data() - raw.copy().set_eeg_reference(ref_channels=['A1','A2']).get_data()