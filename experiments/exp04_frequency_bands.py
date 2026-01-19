#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
from pathlib import Path
import gc
import tensorflow as tf
import mne

from models.decoders import RNN
from utils.loaders import load_data
from utils.processors import Dataset
from utils.metrics import full_eval
from utils.random import reset_seed

if __name__ == "__main__":

    SEED = 123
    DATA_PATH = "data"
    OUT_PATH = "results"
    SEQ_SIZE = 20
    DATA_SPLITS = (0.8, 0.1, 0.1)
    MODE = "sequence"

    sessions = os.listdir(DATA_PATH)
    sessions = [s for s in sessions if s[0].isdigit()]
    # Inclusion criteria, see Methods in paper 
    # (32 channels and speed IQR > 10th percentile)
    exclude = np.load(os.path.join(DATA_PATH, "exclusions.npy")) 
    sessions = [s for s in sessions if s not in exclude]

    # Frequency bands
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
        # Cut-off at 45Hz during pre-processing
    }
     # Create MNE info structure
    sfreq = 100
    ch_names = [f'EEG{i+1}' for i in range(32)]
    ch_types = ['eeg'] * 32
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    corr_rows = {}
    r2_rows = {}

    # Start training loop
    for band_name, (l_freq, h_freq) in bands.items():
        corr_rows[band_name] = {}
        r2_rows[band_name] = {}

        for s in sessions:
            sess = os.path.join(DATA_PATH, s)
            rat_id, eeg, eeg_time, speed, speed_time = load_data(sess)

            # Build dataset
            data = Dataset(X=eeg, y=speed, Xt=eeg_time, yt=speed_time)

            # Extract frequency band
            raw = mne.io.RawArray(data.X.T, info)
            fband = raw.copy().filter(
                l_freq=l_freq, h_freq=h_freq, 
                l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                filter_length='auto', phase='zero').get_data().T
            data.X = np.copy(fband)

            # Run pipeline of pre-processed data
            data.split_dataset(splits=DATA_SPLITS).to_zscore().build(seq_size=SEQ_SIZE, mode=MODE)

            # Train decoder
            reset_seed(SEED)
            model = RNN()
            model.fit(
                data.X_train, data.y_train,
                X_val=data.X_val, y_val=data.y_val,
                verbose=0
            )
            # Evaluate model
            y_hat = model.predict(data.X_test, verbose=0)
            res = full_eval(y_hat, data.y_test)

            # Store results
            corr_rows[band_name][s] = float(res["correlation"])
            r2_rows[band_name][s] = float(res["r2_score"])

            # Clear up memory
            del rat_id, eeg, eeg_time, speed, speed_time, raw, fband, data, model, y_hat, res
            gc.collect()
            tf.keras.backend.clear_session()

    # Build dataframes
    corr_df = pd.DataFrame(corr_rows)
    r2_df   = pd.DataFrame(r2_rows)

    corr_df.index.name = "session_id"
    corr_df = corr_df.reset_index()
    r2_df.index.name = "session_id"
    r2_df = r2_df.reset_index()

    corr_df.to_csv(f"{OUT_PATH}/frequency_bands_correlation.csv", index=False)
    r2_df.to_csv(f"{OUT_PATH}/frequency_bands_r2_score.csv", index=False)
