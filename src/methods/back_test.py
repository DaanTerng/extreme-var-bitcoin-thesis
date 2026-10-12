"""This file contains statistical evaluation criteria used to evaluate estimator performance"""

import numpy as np

from scipy import stats



def quantile_loss_vec(tau_n, log_returns, VaR_forecasts):
    """Vectorised quantile loss over arrays."""
    
    errors = np.asarray(log_returns) - np.asarray(VaR_forecasts)
    
    # Apply the asymmetric quantile loss depending on the sign of the forecast error.
    return np.where(errors < 0, (tau_n - 1) * errors, tau_n * errors)



def quantile_loss(tau_n, log_return_next_day, VaR_forecast):
    """Quantile loss for a single observation."""
    
    error = log_return_next_day - VaR_forecast
    
    return (tau_n - 1) * error if error < 0 else tau_n * error



def kupiec_test(violations, tau):
    """
    Kupiec (1995) test for the expected violation rate.
    Checks whether the observed violation rate equals 1 - tau.
    """

    violations = np.asarray(violations, dtype=float)
    violations = violations[~np.isnan(violations)]
    violations = violations.astype(bool)
    
    n = len(violations)
    
    if n == 0:
        return np.nan, np.nan, 0, 0, np.nan

    x = int(violations.sum())
    p = 1.0 - tau
    p_hat = x / n if n > 0 else np.nan

    # Handle boundary cases separately to avoid taking log(0).
    if n == 0 or x == n:
        lr = -2.0 * n * np.log(p) if x == n else np.nan
    elif x == 0:
        lr = -2.0 * n * np.log(1.0 - p)
    else:
        lr = 2.0 * (x * np.log(p_hat / p) + (n - x) * np.log((1.0 - p_hat) / (1.0 - p)))

    p_value = 1.0 - stats.chi2.cdf(lr, df=1) if not np.isnan(lr) else np.nan
    
    return lr, p_value, x, n, p_hat



def diebold_mariano(scores1, scores2, h=1, alternative='two-sided'):
    """
    Diebold-Mariano test for predictive accuracy.
    Compares the average loss of two models.
    Uses Newey-West HAC variance.
    """
    

    d = np.asarray(scores1, dtype=float) - np.asarray(scores2, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    
    # Require more observations than the requested HAC lag span.
    if T < 2 or T <= h:
        return np.nan, np.nan

    d_bar = np.mean(d)
    bw    = min(max(h - 1, int(T ** (1 / 3))), T - 1)

    # Estimate the long-run variance using the Newey-West estimator.
    lrv = np.var(d, ddof=0)
    
    for k in range(1, bw + 1):
        w       = 1 - k / (bw + 1)
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        lrv    += 2 * w * gamma_k
    
    # Do not report significance when the estimated variance is invalid.
    if not np.isfinite(lrv) or lrv <= 0:
        
        return np.nan, np.nan

    dm_stat = d_bar / np.sqrt(lrv / T)

    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternative == 'less':
        p_value = stats.norm.cdf(dm_stat)
    else:
        p_value = 1 - stats.norm.cdf(dm_stat)

    return dm_stat, p_value
