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

    rows = []

    for source in sessions:
        ss = os.path.join(DATA_PATH, source)
        source_rat_id, eeg, eeg_time, speed, speed_time = load_data(ss)

        # Build source dataset
        source_data = Dataset(X=eeg, y=speed, Xt=eeg_time, yt=speed_time)
        source_data.split_dataset(splits=DATA_SPLITS).to_zscore().build(seq_size=SEQ_SIZE, mode=MODE)
        ten_percent = int(source_data.timesteps * 0.1) # Get index of 10% of session

        # ---------- Scratch-10% ----------
        reset_seed(SEED)
        model = RNN()
        model.fit(
            source_data.X_train[:ten_percent], source_data.y_train[:ten_percent],
            X_val=source_data.X_val, y_val=source_data.y_val, verbose=0
        )
        y_hat = model.predict(source_data.X_test, verbose=0)
        res = full_eval(y_hat, source_data.y_test)
        rows.append({
            "source_id": source,
            "target_id": source,
            "type": "Scratch10SingleSession",
            "correlation": float(res["correlation"]),
            "r2_score": float(res["r2_score"])
        })
        print(rows)
        # Clear up memory
        del model, y_hat, res
        gc.collect()
        tf.keras.backend.clear_session()

        # ---------- Scratch-80% ----------
        reset_seed(SEED)
        model = RNN()
        model.fit(
            source_data.X_train, source_data.y_train,
            X_val=source_data.X_val, y_val=source_data.y_val, verbose=0
        )
        y_hat = model.predict(source_data.X_test, verbose=0)
        res = full_eval(y_hat, source_data.y_test)
        # Store
        rows.append({
            "source_id": source,
            "target_id": source,
            "type": "Scratch80SingleSession",
            "correlation": float(res["correlation"]),
            "r2_score": float(res["r2_score"])
        })
        print(rows)
        # Clear up memory
        del source_data, eeg, eeg_time, speed, speed_time, y_hat, res
        gc.collect()
        # Collect weights 
        base_weights = [w.copy() for w in model.model.get_weights()]

        # Predict on every other session
        for target in sessions:
            if target != source:

                ts = os.path.join(DATA_PATH, target)
                target_rat_id, eeg, eeg_time, speed, speed_time = load_data(ts)

                # Build target dataset
                target_data = Dataset(X=eeg, y=speed, Xt=eeg_time, yt=speed_time)
                target_data.split_dataset(splits=DATA_SPLITS).to_zscore().build(seq_size=SEQ_SIZE, mode=MODE)

                # ---------- Zero-shot ----------
                model.model.set_weights(base_weights)
                y_hat = model.predict(target_data.X_test, verbose=0)
                res = full_eval(y_hat, target_data.y_test)

                # Determine prediction type
                ptype = "CrossSession" if source_rat_id == target_rat_id else "CrossSubject"

                # Store
                rows.append({
                    "source_id": source,
                    "target_id": target,
                    "type": "ZeroShot"+ptype,
                    "correlation": float(res["correlation"]),
                    "r2_score": float(res["r2_score"])
                })
                print(rows)

                # ---------- Fine-tune on 10% ----------
                reset_seed(SEED)
                model.model.set_weights(base_weights)
                ten_percent = int(target_data.timesteps * 0.1) # Get index for 10% of session
                model.fine_tune(
                    target_data.X_train[:ten_percent], target_data.y_train[:ten_percent], 
                     X_val=target_data.X_val, y_val=target_data.y_val, verbose=0, lr=1e-4
                )
                # Predict on target data
                y_hat = model.predict(target_data.X_test, verbose=0)
                res = full_eval(y_hat, target_data.y_test)

                # Store
                rows.append({
                    "source_id": source,
                    "target_id": target,
                    "type": "FineTuning10"+ptype,
                    "correlation": float(res["correlation"]),
                    "r2_score": float(res["r2_score"])
                })
                print(rows)

                # Clear up memory
                del target_data, eeg, eeg_time, speed, speed_time, y_hat, res
                gc.collect()

        # Clear up memory
        del model, base_weights, target_rat_id, source_rat_id 
        gc.collect()
        tf.keras.backend.clear_session()

    # Build dataframe
    df = pd.DataFrame.from_records(rows)

    type_columns = [
        "ZeroShotCrossSubject",
        "ZeroShotCrossSession",
        "Scratch80SingleSession",
        "Scratch10SingleSession",
        "FineTuning10CrossSession",
        "FineTuning10CrossSubject",
    ]

    # Correlation
    df_corr = (
        df.groupby(["target_id", "type"], as_index=False)["correlation"]
        .median()
        .pivot(index="target_id", columns="type", values="correlation")
        .reindex(columns=type_columns)
        .reset_index()
        .rename(columns={"target_id": "session_id"})
    )

    # R2 Score
    df_r2 = (
        df.groupby(["target_id", "type"], as_index=False)["r2_score"]
        .median()
        .pivot(index="target_id", columns="type", values="r2_score")
        .reindex(columns=type_columns)
        .reset_index()
        .rename(columns={"target_id": "session_id"})
    )

    df_corr.to_csv(f"{OUT_PATH}/transfer_learning_correlation.csv", index=False)
    df_r2.to_csv(f"{OUT_PATH}/transfer_learning_r2_score.csv", index=False)
