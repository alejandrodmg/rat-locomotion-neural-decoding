#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
from pathlib import Path
import gc
import tensorflow as tf

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

    # Electrode mapping
    electrode_mapping = np.array([
        10, 11, 12, 13, 14, 15, 
        16, 17, 18, 19, 1, 20, 
        21, 22, 23, 24, 25, 26,
        27, 28, 29, 2, 30, 31, 
        32, 3, 4, 5, 6, 7, 8, 9
    ])

    # Region list
    region_list = {
        "Frontal": [8, 9, 6, 11],
        "SomatoMotor": [4, 13, 2, 15, 23, 24, 1, 16, 25, 26],
        "Motor": [7, 10, 5, 12, 3, 14],
        "Visual": [17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32],
    }
    
    channel_regions = []
    # Map each electrodes to their region
    electrode_to_region = {}
    for region, electrodes in region_list.items():
        for electrode in electrodes:
            electrode_to_region[electrode] = region
    # Map EEG channels to brain regions based on electrode mapping
    for electrode in electrode_mapping:
        region = electrode_to_region.get(electrode, 'Unknown')
        channel_regions.append(region)
    channel_regions = np.array(channel_regions)

    # Create pairs
    region_pairs = []
    for p in region_list.keys():
        region_pairs.append([p])
        for s in region_list.keys():
            if p != s:
                if [s, p] not in region_pairs:
                    region_pairs.append([p, s])

    corr_rows = {}
    r2_rows = {}

    # Start training loop
    for r in region_pairs:
        rname = ''.join(r)
        corr_rows[rname] = {}
        r2_rows[rname] = {}

        for s in sessions: 
            sess = os.path.join(DATA_PATH, s)
            rat_id, eeg, eeg_time, speed, speed_time = load_data(sess)
            # Select channels
            eeg = eeg[:, np.isin(channel_regions, r)]
            # Build dataset
            data = Dataset(X=eeg, y=speed, Xt=eeg_time, yt=speed_time)
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
            corr_rows[rname][s] = float(res["correlation"])
            r2_rows[rname][s] = float(res["r2_score"])

            # Clear up memory
            del data, model, y_hat, res, rat_id, eeg, eeg_time, speed, speed_time
            gc.collect()
            tf.keras.backend.clear_session()

    # Build dataframes
    corr_df = pd.DataFrame(corr_rows)
    r2_df   = pd.DataFrame(r2_rows)

    corr_df.index.name = "session_id"
    corr_df = corr_df.reset_index()
    r2_df.index.name = "session_id"
    r2_df = r2_df.reset_index()

    corr_df.to_csv(f"{OUT_PATH}/localized_correlation.csv", index=False)
    r2_df.to_csv(f"{OUT_PATH}/localized_r2_score.csv", index=False)
