#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial import Polynomial

def bootstrap_intervals_median(data, n_bootstrap=1000, axis=0):
    data = np.asarray(data) 
    if data.ndim == 1:
        data = data[:, np.newaxis] 
        axis = 0
    bootstrap_shape= (n_bootstrap, data.shape[1 - axis])
    bootstrap_medians = np.empty(bootstrap_shape)

    for i in range(n_bootstrap):
        indices = np.random.randint(data.shape[axis], size=data.shape[axis])
        resampled_data = np.take(data, indices, axis=axis)
        bootstrap_medians[i] = np.median(resampled_data, axis=axis)

    # Compute confidence intervals using the percentiles of the bootstrapped medians
    lower_percentile = np.percentile(bootstrap_medians, 2.5, axis=0)
    upper_percentile = np.percentile(bootstrap_medians, 97.5, axis=0)
    return lower_percentile.squeeze(), upper_percentile.squeeze()

if __name__ == "__main__":

    DATA_PATH = "data"
    RESULTS_PATH = "results"

    fields = ["median", "low", "high", "poly_median", "poly_low", "poly_high"]
    directions = ["forward", "reversed"]
    methods = ["EEG_based", "Speed_based"]
    scores = ["correlation", "r2_score"]

    data = {
        d: {
            m: {s: {f: [] for f in fields} for s in scores}
            for m in methods
        }
        for d in directions
    }

    intervals = np.array([0, 10, 20, 50, 100])
    interval_range = np.arange(0, 101)

    score_files = {
        "correlation": "correlation",
        "r2_score": "r2_score",
    }

    # Populate median and median CIs
    for d in directions:
        for i in intervals:
            dfs = {
                score_key: pd.read_csv(
                    f"{RESULTS_PATH}/predictions_{d}_{i}_{suffix}.csv"
                )
                for score_key, suffix in score_files.items()
            }

            for m in methods:
                for score_key, df in dfs.items():
                    med = np.median(df[m].to_numpy())
                    low, high = bootstrap_intervals_median(df[[m]])

                    bucket = data[d][m][score_key]
                    bucket["median"].append(med)
                    bucket["low"].append(low)
                    bucket["high"].append(high)

    # Fit polynomial to data
    for d in directions:
        for m in methods:
            for s in scores:
                bucket = data[d][m][s]

                y_med = np.asarray(bucket["median"], dtype=float)
                y_low = np.asarray(bucket["low"], dtype=float)
                y_high = np.asarray(bucket["high"], dtype=float)

                bucket["poly_median"] = Polynomial.fit(intervals, y_med, 2)(interval_range).tolist()
                bucket["poly_low"] = Polynomial.fit(intervals, y_low, 2)(interval_range).tolist()
                bucket["poly_high"] = Polynomial.fit(intervals, y_high, 2)(interval_range).tolist()

    # Load signal autocorrelation
    autocorrelations = np.load(f"{RESULTS_PATH}/autocorrelations.npy")
    mean_autocorrelation = np.median(autocorrelations, axis=0)
    autocorrelation_low, autocorrelation_high = bootstrap_intervals_median(autocorrelations)

    # Create figures
    plt.rcParams['axes.titlesize'] = 11

    methods = {
        "Speed_based": {"color": "steelblue", "label": "Speed-based"},
        "EEG_based":   {"color": "tomato",    "label": "EEG-based"},
    }

    scores = {
        "correlation": {"ls": "-",  "label": r"($\it{r}$)"},
        "r2_score":    {"ls": "-.", "label": "(R²)"},
    }
    
    # Future predictions
    plt.figure(figsize=(4, 4))

    horizon_range = np.arange(0, 1.01, 0.01)

    # Autocorrelation
    plt.plot(
        horizon_range,
        mean_autocorrelation,
        color="gray",
        label=r"Autocorrelation ($\it{r}$)",
    )
    plt.fill_between(
        horizon_range,
        autocorrelation_low,
        autocorrelation_high,
        alpha=0.1,
        color="gray",
    )

    # Model based
    for method, mcfg in methods.items():
        for score, scfg in scores.items():
            bucket = data["forward"][method][score]

            plt.plot(
                horizon_range,
                bucket["poly_median"],
                color=mcfg["color"],
                linestyle=scfg["ls"],
                label=f"{mcfg['label']} {scfg['label']}",
            )

            plt.fill_between(
                horizon_range,
                bucket["poly_low"],
                bucket["poly_high"],
                alpha=0.1,
                color=mcfg["color"],
            )

    plt.axvline(0, color="black", linestyle=":", lw=1, label="Current Time")
    plt.legend(fontsize=8)

    plt.ylabel("Median Performance Score")
    plt.title("Future Predictions")
    plt.xlabel("Prediction Horizon (s)")
    plt.savefig("forward_predictions.png", dpi=500, bbox_inches="tight")
    plt.close()

    # Past predictions
    plt.figure(figsize=(4, 4))

    horizon_range = np.arange(0, 1.01, 0.01) * -1

    # Autocorrelation
    plt.plot(
        horizon_range,
        mean_autocorrelation,
        color="gray",
        label=r"Autocorrelation ($\it{r}$)",
    )
    plt.fill_between(
        horizon_range,
        autocorrelation_low,
        autocorrelation_high,
        alpha=0.1,
        color="gray",
    )

    for method, mcfg in methods.items():
        for score, scfg in scores.items():
            bucket = data["reversed"][method][score]

            plt.plot(
                horizon_range,
                bucket["poly_median"],
                color=mcfg["color"],
                linestyle=scfg["ls"],
                label=f"{mcfg['label']} {scfg['label']}",
            )

            plt.fill_between(
                horizon_range,
                bucket["poly_low"],
                bucket["poly_high"],
                alpha=0.1,
                color=mcfg["color"],
            )

    # ---------------- annotations ----------------
    plt.axvline(0, color="black", linestyle=":", lw=1, label="Current Time")
    plt.legend(fontsize=8)

    plt.ylabel("Median Performance Score")
    plt.title("Past Predictions")
    plt.xlabel("Prediction Horizon (s)")
    plt.savefig("reversed_predictions.png", dpi=500, bbox_inches="tight")
    plt.close()