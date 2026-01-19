#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf

def reset_seed(seed=123):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)