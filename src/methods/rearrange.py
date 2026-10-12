"""This file contains a quantile rearrangement method used to enforce monotonicity."""

import numpy as np



def rearrange(f):
    """Rearrange quantile estimates to ensure monotonicity."""

    f = np.asarray(f, dtype=float)
    k = len(f)

    if k < 2:
        return f

    # Replicate the right-continuous step function used in R.
    if k == 2:
        y_eval = np.array([f[1], f[1], f[1]])
    else:
        y_eval = np.concatenate([[f[1], f[1]], f[2:-1], [f[-1], f[-1]]])

    # Sort the first k values and add the minimum and maximum.
    y_new   = np.sort(y_eval[:k])                                 
    y_final = np.concatenate([[y_new[0]], y_new, [y_new[-1]]])    

    return y_final
