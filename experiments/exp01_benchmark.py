#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
from pathlib import Path
import gc
import tensorflow as tf

from models.decoders import LinReg, RandForest, RNN, FNN, Transformer
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

    sessions = os.listdir(DATA_PATH)
    sessions = [s for s in sessions if s[0].isdigit()]
    # Inclusion criteria, see Methods in paper 
    # (32 channels and speed IQR > 10th percentile)
    exclude = np.load(os.path.join(DATA_PATH, "exclusions.npy")) 
    sessions = [s for s in sessions if s not in exclude]

    registry = [
        ("LinearRegression", "tabular", LinReg),
        ("RandomForest", "tabular", RandForest),
        ("FNN", "tabular", FNN),
        ("Transformer", "sequence", Transformer),
        ("RNN", "sequence", RNN)
    ]

    model_names = [name for name, _, _ in registry]
    corr_rows = {}
    r2_rows = {}

    # Start training loop
    for s in sessions:
        sess = os.path.join(DATA_PATH, s)
        rat_id, eeg, eeg_time, speed, speed_time = load_data(sess)

        corr_rows[s] = {}
        r2_rows[s] = {}

        for name, mode, Decoder in registry:
            # Build dataset
            data = Dataset(X=eeg, y=speed, Xt=eeg_time, yt=speed_time)
            data.split_dataset(splits=DATA_SPLITS).to_zscore().build(seq_size=SEQ_SIZE, mode=mode)

            # Train decoder
            reset_seed(SEED)
            model = Decoder()
            model.fit(
                data.X_train, data.y_train,
                X_val=data.X_val, y_val=data.y_val,
                verbose=0
            )

            # Evaluate model
            y_hat = model.predict(data.X_test, verbose=0)
            res = full_eval(y_hat, data.y_test)

            # Store results
            corr_rows[s][name] = float(res["correlation"])
            r2_rows[s][name] = float(res["r2_score"])

            # Clear up memory
            del data, model, y_hat, res
            gc.collect()
            tf.keras.backend.clear_session()

        print(f"Done {s}")
        # Clear up memory
        del eeg, eeg_time, speed, speed_time, rat_id
        gc.collect()

    # Build dataframes with decoding results
    df_corr = pd.DataFrame.from_dict(corr_rows, orient="index")[model_names]
    df_r2 = pd.DataFrame.from_dict(r2_rows, orient="index")[model_names]

    df_corr.insert(0, "session_id", df_corr.index)
    df_r2.insert(0, "session_id", df_r2.index)

    df_corr.to_csv(f"{OUT_PATH}/model_selection_correlation.csv", index=False)
    df_r2.to_csv(f"{OUT_PATH}/model_selection_r2_score.csv", index=False)