#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import gc
import tensorflow as tf

from models.decoders import RNN, FNN
from utils.loaders import load_data
from utils.processors import Dataset
from utils.metrics import full_eval
from utils.random import reset_seed, shuffle_time

if __name__ == "__main__":

    SEED = 123
    DATA_PATH = "data"
    OUT_PATH = "results"
    SEQ_SIZE = 20
    DATA_SPLITS = (0.8, 0.1, 0.1)

    sessions = os.listdir(DATA_PATH)
    sessions = [s for s in sessions if s[0].isdigit()]

    exclude = np.load(os.path.join(DATA_PATH, "exclusions.npy"))
    sessions = [s for s in sessions if s not in exclude]

    registry = [
        ("FNN", "tabular", FNN),
        ("RNN", "sequence", RNN)
    ]

    model_names = []
    for name, _, _ in registry:
        model_names += [f"{name}_original", f"{name}_shuffled"]

    corr_rows = {}
    r2_rows = {}

    for s in sessions:
        sess = os.path.join(DATA_PATH, s)
        rat_id, eeg, eeg_time, speed, speed_time = load_data(sess)

        corr_rows[s] = {}
        r2_rows[s] = {}

        for name, mode, Decoder in registry:
            # Build dataset
            data = Dataset(X=eeg, y=speed, Xt=eeg_time, yt=speed_time)
            data.split_dataset(splits=DATA_SPLITS).to_zscore().build(
                seq_size=SEQ_SIZE,
                mode=mode
            )

            # Shuffle temporal order in training and validation data
            X_train_shuf = shuffle_time(data.X_train, mode, SEQ_SIZE)
            X_val_shuf = shuffle_time(data.X_val, mode, SEQ_SIZE)

            # Train decoder on shuffled temporal data
            reset_seed(SEED)
            model = Decoder()
            model.fit(
                X_train_shuf,
                data.y_train,
                X_val=X_val_shuf,
                y_val=data.y_val,
                verbose=1
            )

            # Evaluate on original test data
            y_hat = model.predict(data.X_test, verbose=0)
            res = full_eval(y_hat, data.y_test)

            corr_rows[s][f"{name}_original"] = float(res["correlation"])
            r2_rows[s][f"{name}_original"] = float(res["r2_score"])

            # Evaluate on shuffled-time test data
            X_test_shuf = shuffle_time(data.X_test, mode, SEQ_SIZE)
            y_hat_shuf = model.predict(X_test_shuf, verbose=0)
            res_shuf = full_eval(y_hat_shuf, data.y_test)

            corr_rows[s][f"{name}_shuffled"] = float(res_shuf["correlation"])
            r2_rows[s][f"{name}_shuffled"] = float(res_shuf["r2_score"])

            del data, model
            del X_train_shuf, X_test_shuf
            del y_hat, y_hat_shuf, res, res_shuf

            gc.collect()
            tf.keras.backend.clear_session()

        print(f"Done {s}")

        del eeg, eeg_time, speed, speed_time, rat_id
        gc.collect()

    # Build dataframes with decoding results
    df_corr = pd.DataFrame.from_dict(corr_rows, orient="index")[model_names]
    df_r2 = pd.DataFrame.from_dict(r2_rows, orient="index")[model_names]

    df_corr.insert(0, "session_id", df_corr.index)
    df_r2.insert(0, "session_id", df_r2.index)

    df_corr.to_csv(f"{OUT_PATH}/shuffled_correlation.csv", index=False)
    df_r2.to_csv(f"{OUT_PATH}/shuffled_r2_score.csv", index=False)
