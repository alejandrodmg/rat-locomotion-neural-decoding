#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import r2_score

def full_eval(y_hat, y_test):
    r = np.corrcoef(y_test, y_hat)[0][1]
    r2 = r2_score(y_test, y_hat)
    return {"correlation": r, "r2_score": r2}
