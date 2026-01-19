#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":

    DATA_PATH = "data"
    RESULTS_PATH = "results"

    selection_r = pd.read_csv(f"{RESULTS_PATH}/model_selection_correlation.csv")
    selection_r2 = pd.read_csv(f"{RESULTS_PATH}/model_selection_r2_score.csv")

    # Correlation
    cor_fig, axs = plt.subplots(5, 1, figsize=(4, 7), sharex=True)

    bin_width = 0.05
    bins = np.arange(0, 1 + bin_width, bin_width)

    plots = [
        ("LinearRegression", "Linear Regression", "steelblue"),
        ("RandomForest", "Random Forest", "seagreen"),
        ("FNN", "Feed-Forward Neural Network", "gray"),
        ("Transformer", "Encoder-Only Transformer", "gold"),
        ("RNN", "Recurrent Neural Network", "orange"),
    ]

    for ax, (col, title, color) in zip(axs, plots):
        sns.histplot(
            selection_r[col],
            ax=ax, color=color, fill=True, alpha=0.4,
            kde=True, bins=bins, stat="percent"
        )

        med = np.median(selection_r[col])
        ax.axvline(med, color="red", alpha=0.5, linewidth=2, label=f"Median: {med:.3f}")

        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 50)
        ax.set_ylabel("Sessions (%)")
        ax.legend(prop={"size": 8})

    # Put x ticks only on the bottom axis
    axs[-1].set_xticks(np.arange(0, 1.1, 0.1))

    plt.xlabel("Correlation ($\\it{r}$)") 
    plt.tight_layout()
    plt.savefig("model_correlations.png", dpi=500, bbox_inches="tight")
    plt.close()

    # R2 Score
    r2_fig, axs = plt.subplots(5, 1, figsize=(4, 7), sharex=True)

    bin_width = 0.05
    bins = np.arange(0, 1 + bin_width, bin_width)

    for ax, (col, title, color) in zip(axs, plots):
        # Exclude neagtive values for visual purposes
        x_hist = selection_r2[col][selection_r2[col] > 0]

        sns.histplot(
            x_hist,
            ax=ax, color=color, fill=True, alpha=0.4,
            kde=True, bins=bins, stat="percent"
        )

        med = np.median(selection_r2[col])
        ax.axvline(med, color="red", alpha=0.5, linewidth=2, label=f"Median: {med:.3f}")

        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 50)
        ax.set_ylabel("Sessions (%)")
        ax.legend(prop={"size": 8}, loc="upper left")

    axs[-1].set_xticks(np.arange(0, 1.1, 0.1))

    plt.xlabel("R²")
    plt.tight_layout()
    plt.savefig("model_determination.png", dpi=500, bbox_inches="tight")
    plt.close()