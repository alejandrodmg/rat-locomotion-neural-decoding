#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d

class Dataset():
    """ EEG = X,
        EEG time = Xt,
        Treadmill speed = y,
        Treadmill speed time = yt """
    def __init__(self, X, y, Xt, yt):
        self.X = self._downsample(X, Xt, yt)
        self.y = np.abs(y * 1e5)
        self.timesteps = len(self.y)
        # Speed can be slightly negative during brief backward steps.
        # We take the absolute value to represent locomotion magnitude
        # and rescale by 1e5 to improve numerical stability

    def _downsample(self, X, Xt, yt):
        # Downsample EEG (200Hz) to treadmill speed (100Hz)
        # with simple linear interpolation
        interp = interp1d(
            Xt,
            X,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )
        downX = interp(yt)
        return downX

    def split_dataset(self, splits=(0.8, 0.1, 0.1)):
        # EEG (timesteps, n_channels)
        # Speed (timesteps)
        idx_train = int(self.timesteps*splits[0])
        idx_val = int(self.timesteps*(splits[0]+splits[1]))
        # Training
        self.X_train = self.X[:idx_train, :]
        self.y_train = self.y[:idx_train]
        # Validation
        self.X_val = self.X[idx_train:idx_val, :]
        self.y_val = self.y[idx_train:idx_val]
        # Test
        self.X_test = self.X[idx_val:, :]
        self.y_test = self.y[idx_val:]
        return self

    def to_zscore(self, eps=1e-8):
        """
        Z-score EEG per channel using training-set stats only.
        """
        mu = self.X_train.mean(axis=0)
        sd = self.X_train.std(axis=0, ddof=0)
        self.zscore_mean_ = mu
        self.zscore_std_  = sd
        # Avoid division by zero for constant channels
        sd = np.where(sd < eps, 1.0, sd)
        # Apply to all splits
        self.X_train = (self.X_train - mu) / sd
        self.X_val = (self.X_val - mu) / sd
        self.X_test = (self.X_test - mu) / sd
        return self

    def make_windows(self, X, y, N=20, mode="sequence"):
        """
        Create sliding windows from a single split.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected X with shape (T, C); got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"Expected y with shape (T,); got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}")
        if N <= 0:
            raise ValueError("N must be a positive integer.")

        T, C = X.shape
        if T < N:
            if mode == "sequence":
                return np.empty((0, N, C), dtype=X.dtype), np.empty((0,), dtype=y.dtype)
            elif mode == "tabular":
                return np.empty((0, N * C), dtype=X.dtype), np.empty((0,), dtype=y.dtype)
            else:
                raise ValueError("mode must be 'sequence' or 'tabular'.")

        n = T - N + 1
        # (n, N, C): each sample is exactly (N, C) = (20, 32)
        Xw = np.stack([X[i:i+N] for i in range(n)], axis=0)
        # Edge-of-window target
        yw = y[N-1:]
        if mode == "tabular":
            Xw = Xw.reshape(n, N * C)
        elif mode == "sequence":
            pass
        else:
            raise ValueError("mode must be 'sequence' or 'tabular'.")
        return Xw, yw

    def build(self, seq_size=20, mode="sequence"):
        """
        Apply window transformations to the dataset.
        """
        for split in ("train", "val", "test"):
            X = getattr(self, f"X_{split}")
            y = getattr(self, f"y_{split}")
            Xw, yw = self.make_windows(X, y, N=seq_size, mode=mode)
            setattr(self, f"X_{split}", Xw)
            setattr(self, f"y_{split}", yw)
        return self
