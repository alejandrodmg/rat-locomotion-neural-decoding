#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf

def reset_seed(seed=123):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def shuffle_time(X, mode, seq_size):
    """ Helper function for shluffling the temporal 
    structure of the input features to decoders.
    See tests.temporal_structure for experiment."""
    Xs = X.copy()

    if mode == "sequence":
        for i in range(Xs.shape[0]):
            perm = np.random.permutation(Xs.shape[1])
            Xs[i] = Xs[i, perm, :]
        return Xs
    
    if mode == "tabular":
        n_samples = Xs.shape[0]
        n_channels = Xs.shape[1] // seq_size
        Xs = Xs.reshape(n_samples, seq_size, n_channels)

        for i in range(n_samples):
            perm = np.random.permutation(seq_size)
            Xs[i] = Xs[i, perm, :]
        return Xs.reshape(n_samples, seq_size * n_channels)
    
    raise ValueError("mode must be 'sequence' or 'tabular'.")
