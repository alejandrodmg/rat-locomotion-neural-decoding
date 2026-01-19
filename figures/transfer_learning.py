#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

def get_outliers(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data < lower_bound) | (data > upper_bound)]

if __name__ == "__main__":

    plt.rcParams['axes.titlesize'] = 11
    np.random.seed(123)

    DATA_PATH = "data"
    RESULTS_PATH = "results"

    transfer_r = pd.read_csv(f"{RESULTS_PATH}/transfer_learning_correlation.csv")
    transfer_r2 = pd.read_csv(f"{RESULTS_PATH}/transfer_learning_r2_score.csv")

    # Correlation
    custom_order = [
        "Scratch80SingleSession",
        "FineTuning10CrossSession",
        "FineTuning10CrossSubject",
        "Scratch10SingleSession",
        "ZeroShotCrossSession",
        "ZeroShotCrossSubject",
    ]
    custom_labels = {
        "Scratch80SingleSession": "Single-Session (Scratch-80%)",
        "FineTuning10CrossSession": "Cross-Session (Fine-Tuning-10%)",
        "FineTuning10CrossSubject": "Cross-Subject (Fine-Tuning-10%)",
        "Scratch10SingleSession": "Single-Session (Scratch-10%)",
        "ZeroShotCrossSession": "Cross-Session (Zero-Shot)",
        "ZeroShotCrossSubject": "Cross-Subject (Zero-Shot)",
    }

    pal = dict(zip(custom_order, sns.color_palette("Set2", len(custom_order))))

    # Convert wide
    cor_df = (
        transfer_r.melt(
            id_vars=["session_id"],
            value_vars=custom_order,
            var_name="Condition",
            value_name="Correlation",
        )
        .dropna(subset=["Correlation"])
    )
    cor_df["Condition"] = pd.Categorical(cor_df["Condition"], categories=custom_order, ordered=True)

    outlier_list = []
    for cond in custom_order:
        subset = cor_df[cor_df["Condition"] == cond]
        outliers = get_outliers(subset["Correlation"])
        if not outliers.empty:
            outlier_list.append(pd.DataFrame({"Condition": cond, "Correlation": outliers}))

    df_outliers = pd.concat(outlier_list, ignore_index=True) if outlier_list else pd.DataFrame(
        columns=["Condition", "Correlation"]
    )

    if not df_outliers.empty:
        df_outliers["Condition"] = pd.Categorical(df_outliers["Condition"], categories=custom_order, ordered=True)

    fig = plt.figure(figsize=(4, 7))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 0.5])
    ax_box = fig.add_subplot(gs[0])
    ax_out = fig.add_subplot(gs[1])

    # Top: boxplot without outliers
    sns.boxplot(
        x="Condition",
        y="Correlation",
        hue="Condition",
        data=cor_df,
        showfliers=False,
        ax=ax_box,
        order=custom_order,
        palette=pal,
        hue_order=custom_order,
    )

    ax_box.set_title("Decoding Performance Distribution ($\\it{r}$)")
    ax_box.set_xlabel("")
    ax_box.set_ylabel(r"Correlation ($\it{r}$)")
    ax_box.set_xticks([])

    legend = ax_box.get_legend()
    if legend is not None:
        legend.remove()

    # Medians annotation
    offsets = [-0.045, -0.040, -0.045, -0.045, -0.045, -0.045]
    for i, cond in enumerate(custom_order):
        median_val = cor_df.loc[cor_df["Condition"] == cond, "Correlation"].median()
        offset = offsets[i]
        ax_box.text(
            i,
            median_val + offset,
            f"{median_val:.2f}",
            horizontalalignment="center",
            color="black",
            weight=1000,
            fontname="DejaVu Sans",
            fontsize=7,
            alpha=0.5,
        )

    # Bottom: outliers
    sns.stripplot(
        x="Condition",
        y="Correlation",
        hue="Condition",
        data=df_outliers,
        ax=ax_out,
        palette=pal,
        size=3,
        order=custom_order,
        hue_order=custom_order,
        alpha=0.8,
        dodge=False,
        legend=False,
    )

    sns.violinplot(
        x="Condition",
        y="Correlation",
        hue="Condition",
        data=df_outliers,
        ax=ax_out,
        split=False,
        fill=False,
        inner=None,
        palette=pal,
        hue_order=custom_order,
        bw_adjust=0.9,
        dodge=False,
        legend=False,
    )

    ax_out.set_title("Outlier Distribution ($\\it{r}$)")
    ax_out.set_xlabel("")
    ax_out.set_ylabel(r"Correlation ($\it{r}$)")
    ax_out.set_xticks([])

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=custom_labels[cond],
            markerfacecolor=pal[cond],
            markersize=8,
            markeredgecolor="gray",
            markeredgewidth=1,
        )
        for cond in custom_order
    ]

    ax_out.legend(
        handles=legend_elements,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0, -0.2, 1, 0.2),
        ncol=2,
        columnspacing=2,
        handlelength=0.5,
    )

    plt.subplots_adjust(hspace=0.20)
    plt.savefig("transfer_learning_corr.png", dpi=500, bbox_inches="tight")
    plt.close()

    # R2 Score
    r2_df = (
        transfer_r2.melt(
            id_vars=["session_id"],
            value_vars=custom_order,
            var_name="Condition",
            value_name="R2",
        )
        .dropna(subset=["R2"])
    )
    r2_df["Condition"] = pd.Categorical(r2_df["Condition"], categories=custom_order, ordered=True)

    # Remove extreme values
    MIN_R_SCORE = -10
    r2_df = r2_df[r2_df["R2"] >= MIN_R_SCORE]

    outlier_list = []
    for cond in custom_order: 
        subset = r2_df[r2_df["Condition"] == cond]
        outliers = get_outliers(subset["R2"])
        if not outliers.empty:
            outlier_list.append(pd.DataFrame({"Condition": cond, "R2": outliers}))

    df_outliers = pd.concat(outlier_list, ignore_index=True) if outlier_list else pd.DataFrame(
        columns=["Condition", "R2"]
    )

    if not df_outliers.empty:
        df_outliers["Condition"] = pd.Categorical(df_outliers["Condition"], categories=custom_order, ordered=True)

    fig = plt.figure(figsize=(4, 7))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 0.5])
    ax_box = fig.add_subplot(gs[0])
    ax_out = fig.add_subplot(gs[1])

    # Top: boxplot without outliers
    sns.boxplot(
        x="Condition",
        y="R2",
        hue="Condition",
        data=r2_df,
        showfliers=False,
        ax=ax_box,
        order=custom_order,
        palette=pal,
        hue_order=custom_order,
    )

    ax_box.set_title("Decoding Performance Distribution (R²)")
    ax_box.set_xlabel("")
    ax_box.set_ylabel(r"R²")
    ax_box.set_xticks([])

    legend = ax_box.get_legend()
    if legend is not None:
        legend.remove()

    # Medians annotation
    offsets = [-0.095, -0.095, -0.14, -0.14, -0.13, -0.15]
    for i, cond in enumerate(custom_order):
        median_val = r2_df.loc[r2_df["Condition"] == cond, "R2"].median()
        offset = offsets[i]
        ax_box.text(
            i,
            median_val + offset,
            f"{median_val:.2f}",
            horizontalalignment="center",
            color="black",
            weight=1000,
            fontname="DejaVu Sans",
            fontsize=7,
            alpha=0.5,
        )

    # Bottom: outliers
    sns.stripplot(
        x="Condition",
        y="R2",
        hue="Condition",
        data=df_outliers,
        ax=ax_out,
        palette=pal,
        size=3,
        order=custom_order,
        hue_order=custom_order,
        alpha=0.8,
        dodge=False,
        legend=False,
    )

    sns.violinplot(
        x="Condition",
        y="R2",
        hue="Condition",
        data=df_outliers,
        ax=ax_out,
        split=False,
        fill=False,
        inner=None,
        palette=pal,
        hue_order=custom_order,
        bw_adjust=0.9,
        dodge=False,
        legend=False,
    )

    ax_out.set_title("Outlier Distribution (R²)")
    ax_out.set_xlabel("")
    ax_out.set_ylabel(r"R²")
    ax_out.set_xticks([])

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=custom_labels[cond],
            markerfacecolor=pal[cond],
            markersize=8,
            markeredgecolor="gray",
            markeredgewidth=1,
        )
        for cond in custom_order
    ]

    ax_out.legend(
        handles=legend_elements,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0, -0.2, 1, 0.2),
        ncol=2,
        columnspacing=2,
        handlelength=0.5,
    )

    plt.subplots_adjust(hspace=0.20)
    plt.savefig("transfer_learning_r2.png", dpi=500, bbox_inches="tight")
    plt.close()


   