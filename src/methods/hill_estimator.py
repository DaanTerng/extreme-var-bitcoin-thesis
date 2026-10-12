"""This file contains Hill estimators for the extreme value index."""

import numpy as np

from src.methods.rearrange import rearrange



def hill_estimator_CS(sorted_res, k):
    """Hill (1975) estimator for the extreme value index gamma. Based on the k largest order statistics"""
    
    n = len(sorted_res)
    
    if k >= n:
        raise ValueError(f"k={k} must be less than sample size n={n}.")
    if k < 1:
        raise ValueError(f"k={k} must be at least 1.")

    nominator = sorted_res[n - k:]  
    denominator = sorted_res[n - k - 1]

    # Compute log ratios of the upper order statistics.
    log_ratios = np.log(nominator / denominator)

    gamma_hat = log_ratios.mean()

    return gamma_hat



def hill_estimator_path(data, k_min=10, k_max=200):
    """Compute Hill estimator for a range of k values (the Hill plot)."""
    
    n = len(data)
    data = np.sort(data)
    
    if k_max is None:
        k_max = n // 2

    ks = np.arange(k_min, k_max + 1)
    gammas = np.array([hill_estimator_CS(data, k) for k in ks])

    return ks, gammas



def hill_estimator_NCS(n, quantiles, taus, eta, tol=1e-4, min_prop=0.3):
    """Hill (1975) estimator for the extreme value index. Applied to intermediate conditional quantile estimates at each prediction point."""
    
    quantiles = np.atleast_2d(quantiles)
    n_rows = len(quantiles)

    gamma_hat = np.full(n_rows, np.nan)
    q_base    = np.full(n_rows, np.nan)

    for i in range(n_rows):
        x = quantiles[i]

        # Check whether the quantile estimates are sufficiently increasing.
        diffs = np.diff(x)
        valid_diffs = diffs[~np.isnan(diffs)]
        
        if len(valid_diffs) == 0:
            continue
        
        tt = np.mean(valid_diffs > tol)
        
        if tt <= min_prop:
            continue

        # Keep positive, non-missing quantile estimates.
        idx   = np.where((x > 0) & ~np.isnan(x))[0]
        f     = x[idx]
        taus2 = taus[idx]
        
        if len(f) < 2:
            continue

        # Rearrange the quantile estimates to ensure monotonicity.
        x_rear = rearrange(f)

        # Estimate the extreme value index from the rearranged quantiles.
        gamma_hat[i] = np.mean(np.log(x_rear / x_rear[0]))
        q_base[i]    = quantiles[i, 0]

    return gamma_hat, q_base

