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

def rows_to_df(rows_dict, sessions_order):
    eeg_map   = rows_dict.get("EEG_based", {})
    speed_map = rows_dict.get("Speed_based", {})

    all_sessions = set(eeg_map.keys()) | set(speed_map.keys())
    # Keep the original order of the sessions
    ordered = [s for s in sessions_order if s in all_sessions] + \
                [s for s in sorted(all_sessions) if s not in sessions_order]
    # Convert to dataframe
    df = pd.DataFrame({
        "session_id": ordered,
        "EEG_based":  [eeg_map.get(s, np.nan) for s in ordered],
        "Speed_based":[speed_map.get(s, np.nan) for s in ordered],
    })
    return df

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
    horizons = {
        'forward': [0, 10, 20, 50, 100],
        'reversed': [0, 10, 20, 50, 100]
    }

    corr_rows = {}
    r2_rows = {}

    # Start training loop
    for direction in horizons.keys():
        corr_rows[direction] = {}
        r2_rows[direction] = {}

        for h in horizons[direction]:
            corr_rows[direction][h] = {"EEG_based": {}, "Speed_based": {}}
            r2_rows[direction][h]   = {"EEG_based": {}, "Speed_based": {}}

            for s in sessions:
                sess = os.path.join(DATA_PATH, s)
                rat_id, eeg, eeg_time, speed, speed_time = load_data(sess)

                # EEG-based
                """ Regular EEG training in forward and reverse order while shifting the target """

                # Build dataset
                data = Dataset(X=eeg, y=speed, Xt=eeg_time, yt=speed_time)

                # Prepare data before splits
                # Direction
                if direction == "reversed":
                    data.X = data.X[::-1, :]
                    data.y = data.y[::-1]
                # Prediction horizon (shift target)
                if h > 0:
                    data.X = data.X[:-h, :]
                    data.y = data.y[h:]

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
                corr_rows[direction][h]["EEG_based"][s] = float(res["correlation"])
                r2_rows[direction][h]["EEG_based"][s] = float(res["r2_score"])

                # Clear up memory
                del data, model, y_hat, res
                gc.collect()
                tf.keras.backend.clear_session()

                # Speed-based
                """ Same processing but replacing the EEG X data with a 1-channel speed input """

                # Build dataset
                data = Dataset(X=eeg, y=speed, Xt=eeg_time, yt=speed_time)

                # Prepare data before splits
                # Replace EEG, with a 1-channel speed input
                data.X = np.expand_dims(data.y, axis=1)
                # Direction
                if direction == "reversed":
                    data.X = data.X[::-1, :]
                    data.y = data.y[::-1]
                # Prediction horizon (shift target)
                if h > 0:
                    data.X = data.X[:-h, :]
                    data.y = data.y[h:]

                # Do not train the model to predict the instantaneous speed
                # That is a trivial case because the prediction is in the input sequence
                # e.g. input = [1.2, 3.1, ..., 2.2], y = 2.2
                if h == 0:
                    # Store results
                    corr_rows[direction][h]["Speed_based"][s] = 1.0
                    r2_rows[direction][h]["Speed_based"][s] = 1.0
                    # Clear up memory
                    gc.collect()
                    del rat_id, eeg, eeg_time, speed, speed_time, data
                else:
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
                    corr_rows[direction][h]["Speed_based"][s] = float(res["correlation"])
                    r2_rows[direction][h]["Speed_based"][s] = float(res["r2_score"])

                    # Clear up memory
                    del rat_id, eeg, eeg_time, speed, speed_time, data, model, y_hat, res
                    gc.collect()
                    tf.keras.backend.clear_session()

    # Build dataframes, write one file per direction/horizon/metric
    for direction in horizons.keys():
        for h in horizons[direction]:
            # Correlation
            corr_df = rows_to_df(corr_rows[direction][h], sessions)
            corr_path = f"{OUT_PATH}/predictions_{direction}_{h}_correlation.csv"
            corr_df.to_csv(corr_path, index=False)

            # R2 score
            r2_df = rows_to_df(r2_rows[direction][h], sessions)
            r2_path = f"{OUT_PATH}/predictions_{direction}_{h}_r2_score.csv"
            r2_df.to_csv(r2_path, index=False)