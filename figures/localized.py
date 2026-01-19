#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def build_rcorr_from_localized(localized_correlation, region_order):
    df = localized_correlation
    n = len(region_order)
    M = np.zeros((n, n), dtype=float)

    def find_pair_col(a, b):
        candidates = [f"{a}{b}", f"{b}{a}", f"{a}_{b}", f"{b}_{a}"]
        for c in candidates:
            if c in df.columns:
                return c

    for i, a in enumerate(region_order):
        # Diagonal: single-region median
        if a not in df.columns:
            raise KeyError(f"Missing single-region column: '{a}'")
        M[i, i] = df[a].median()

        # Off-diagonal: pairwise median
        for j in range(i + 1, n):
            b = region_order[j]
            col = find_pair_col(a, b)
            val = df[col].median()
            M[i, j] = val
            M[j, i] = val

    return M

if __name__ == "__main__":

    DATA_PATH = "data"
    RESULTS_PATH = "results"

    localized_correlation = pd.read_csv(f"{RESULTS_PATH}/localized_correlation.csv")
    localized_r2_score = pd.read_csv(f"{RESULTS_PATH}/localized_r2_score.csv")

    brain_regions = ['Frontal', 'Motor', 'SomatoMotor', 'Visual']
    heatmap_labels = ["Medial Prefrontal", "Motor", "Somatomotor", "Visual"]

    correlations = build_rcorr_from_localized(localized_correlation, brain_regions)
    r2_scores = build_rcorr_from_localized(localized_r2_score, brain_regions)

    # Correlation
    plt.figure(figsize=(7, 5))
    plt.title("Pairwise and Single-Region Decoding Analysis ($\it{r}$)", pad=10)
    sns.heatmap(correlations, 
                annot=True, 
                cbar=True, 
                cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                xticklabels=heatmap_labels, 
                yticklabels=heatmap_labels, 
                linecolor="gray",
                linewidths=2, 
                cbar_kws={'label': 'Correlation ($\it{r}$)', 'orientation': 'vertical'}, 
                annot_kws={"size": 10})
    
    plt.savefig('region_pairs_corr.png', dpi=500,  bbox_inches='tight')
    plt.close()

    # R2 Score
    plt.figure(figsize=(7, 5))
    plt.title("Pairwise and Single-Region Decoding Analysis (R²)", pad=10)
    sns.heatmap(r2_scores, 
                annot=True, 
                cbar=True, 
                cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                xticklabels=heatmap_labels, 
                yticklabels=heatmap_labels, 
                linecolor="gray",
                linewidths=2, 
                cbar_kws={'label': 'R²', 'orientation': 'vertical'})
    
    plt.savefig('region_pairs_r2.png', dpi=500,  bbox_inches='tight')
    plt.close()