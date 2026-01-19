#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mne
from tqdm import tqdm

from utils.loaders import load_data
from utils.processors import Dataset

if __name__ == "__main__":

    DATA_PATH = "data"
    RESULTS_PATH = "results"

    bands_r = pd.read_csv(f"{RESULTS_PATH}/frequency_bands_correlation.csv")
    bands_r2 = pd.read_csv(f"{RESULTS_PATH}/frequency_bands_r2_score.csv")

    plots = [
        ("Delta", "Delta (1-4 Hz)", "steelblue"),
        ("Theta", "Theta (4-8 Hz)", "seagreen"),
        ("Alpha", "Alpha (8-13 Hz)", "gold"),
        ("Beta", "Beta (13-30 Hz)", "orange"),
        ("Gamma", "Gamma (30-45 Hz)", "tomato"),
    ]

    # ----------------------------- Histograms --------------------------------
    # Correlation
    cor_fig, axs = plt.subplots(5, 1, figsize=(4, 7), sharex=True)

    bin_width = 0.05
    bins = np.arange(0, 1 + bin_width, bin_width)

    for ax, (col, title, color) in zip(axs, plots):
        sns.histplot(
            bands_r[col],
            ax=ax, color=color, fill=True, alpha=0.4,
            kde=True, bins=bins, stat="percent"
        )

        med = np.median(bands_r[col])
        ax.axvline(med, color="red", alpha=0.5, linewidth=2, label=f"Median: {med:.3f}")

        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 50)
        ax.set_ylabel("Sessions (%)")
        ax.legend(prop={"size": 8})

    axs[-1].set_xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("Correlation ($\\it{r}$)")
    plt.tight_layout()
    plt.savefig("frequencies_correlations.png", dpi=500, bbox_inches="tight")
    plt.close()

    # R2 score
    r2_fig, axs = plt.subplots(5, 1, figsize=(4, 7), sharex=True)

    for ax, (col, title, color) in zip(axs, plots):
        x_hist = bands_r2[col][bands_r2[col] > 0]

        sns.histplot(
            x_hist,
            ax=ax, color=color, fill=True, alpha=0.4,
            kde=True, bins=bins, stat="percent"
        )

        med = np.median(bands_r2[col])
        ax.axvline(med, color="red", alpha=0.5, linewidth=2, label=f"Median: {med:.3f}")

        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 50)
        ax.set_ylabel("Sessions (%)")
        ax.legend(prop={"size": 8}, loc="upper left")

    axs[-1].set_xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("R²")
    plt.tight_layout()
    plt.savefig("frequencies_determination.png", dpi=500, bbox_inches="tight")
    plt.close()

    # ---------------------- EEG power spectra across speed percentiles --------
    # Get all sessions
    sessions = os.listdir(DATA_PATH)
    sessions = [s for s in sessions if s and s[0].isdigit()]
    exclude = np.load(os.path.join(DATA_PATH, "exclusions.npy"))
    sessions = [s for s in sessions if s not in exclude]

    # Sampling rate + PSD params
    sfreq = 100
    n_bins = 10
    fmin = 0
    fmax = 45
    n_fft = 128
    n_overlap = 64

    # Precompute frequency vector and n_freqs (so padding is consistent)
    _dummy = np.zeros((1, 1024))
    _psd_dummy, freqs = mne.time_frequency.psd_array_welch(
        _dummy,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_overlap,
        average="mean",
        verbose=False,
    )
    n_freqs = len(freqs)

    def compute_avg_psd_fxp(data, sfreq):
        """
        Returns (freqs, mean_fxp):
          - raw Welch PSD per channel
          - multiplied by frequency: f * P(f) to flatten 1/f
          - averaged across channels
        """
        psds, freqs_local = mne.time_frequency.psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
            n_overlap=n_overlap,
            average="mean",
            verbose=False,
        )

        # Frequency-normalized spectrum: f * P(f)
        # PSDs shape: (n_channels, n_freqs)
        fxp = psds * freqs_local[None, :]
        mean_fxp = fxp.mean(axis=0)
        return freqs_local, mean_fxp

    psd_by_bin_all_sessions = []
    speed_by_bin_all_sessions = []

    for s in tqdm(sessions):
        sess = os.path.join(DATA_PATH, s)
        rat_id, eeg, eeg_time, speed, speed_time = load_data(sess)
        data = Dataset(X=eeg, y=speed, Xt=eeg_time, yt=speed_time)

        brain_data = data.X
        target = data.y

        if len(target) < 1000:
            continue

        speed_bins = np.percentile(target, np.linspace(0, 100, n_bins + 1))

        psd_by_bin = []
        speed_by_bin = []

        for i in range(n_bins):
            mask = (target >= speed_bins[i]) & (target < speed_bins[i + 1])

            # If too few samples in this bin, keep shapes consistent with NaNs
            if np.sum(mask) < 512:
                psd_by_bin.append(np.full(n_freqs, np.nan))
                speed_by_bin.append(np.nan)
                continue

            data_bin = brain_data[mask, :].T
            speed_bin = target[mask]

            speed_by_bin.append(np.mean(speed_bin))
            freqs_local, psd_bin = compute_avg_psd_fxp(data_bin, sfreq)
            psd_by_bin.append(psd_bin)

        psd_by_bin_all_sessions.append(psd_by_bin)
        speed_by_bin_all_sessions.append(speed_by_bin)

    psd_by_bin_all_sessions = np.asarray(psd_by_bin_all_sessions, dtype=float)
    speed_by_bin_all_sessions = np.asarray(speed_by_bin_all_sessions, dtype=float)

    # Mean speeds per bin (across sessions)
    mean_speeds = np.nanmean(speed_by_bin_all_sessions, axis=0)

    # Mean PSD (f*P(f)) and SEM across sessions
    mean_psd = np.nanmean(psd_by_bin_all_sessions, axis=0)

    n_eff = np.sum(~np.isnan(psd_by_bin_all_sessions), axis=0)
    std_psd = np.nanstd(psd_by_bin_all_sessions, axis=0, ddof=1)
    sem_psd = std_psd / np.sqrt(n_eff)
    sem_psd[n_eff <= 1] = np.nan

    # Plot
    plt.figure(figsize=(7, 5))
    colors = plt.cm.magma(np.linspace(0.1, 0.9, n_bins))[::-1]

    for i in range(n_bins):
        label = f"{int(i * 10)}–{int((i + 1) * 10)}% (μ={mean_speeds[i]:.2f})"
        plt.plot(freqs, mean_psd[i], label=label, color=colors[i])
        plt.fill_between(
            freqs,
            mean_psd[i] - sem_psd[i],
            mean_psd[i] + sem_psd[i],
            alpha=0.2,
            color=colors[i],
        )

    plt.title("Frequency-normalized EEG Power Spectra\nAcross Treadmill Speed Percentiles")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Frequency-normalized power (f × PSD, a.u.)")
    plt.xlim(0, 45)
    plt.legend(title="Speed Percentile", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("eeg_power_spectra_speeds.png", dpi=500)
    plt.close()