#!/usr/bin/env python3

import numpy as np
import h5py

def obj_to_string(array):
    return u''.join([chr(o[0]) for o in array])

def load_data(file):
    with h5py.File(file, 'r') as f:
        data = f['Data_eeg']
        # Column 1 = Rat ID
        rat_id = f[data[0][0]][:]
        rat_id = obj_to_string(rat_id)
        # Column 3 = EEG
        eeg = f[data[2][0]][:]
        # Column 4 = EEG time
        eeg_time = f[data[3][0]][:].reshape(-1)
        # Column 5 = Treadmill Speed
        speed = f[data[4][0]][:].reshape(-1)
        # Column 6 = Speed Time
        speed_time = f[data[5][0]][:].reshape(-1)
    return rat_id, eeg, eeg_time, speed, speed_time
