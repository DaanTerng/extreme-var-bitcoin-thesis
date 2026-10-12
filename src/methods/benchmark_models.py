"""This file contains the benchmark model used to compare the 2Stage and 3Stage estimators."""

import numpy as np

from arch import arch_model
from scipy.stats import genpareto



def GARCH11_EVT(log_returns, tau_n):
    """Estimate one-step-ahead VaR using a GARCH(1,1)-EVT model."""
    # Step 1: Fit GARCH(1,1)
    am = arch_model(log_returns * 100, vol='Garch', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    
    # Extract standardised residuals
    sigma = res.conditional_volatility / 100
    z = log_returns / sigma
    
    # Step 2: Fit GPD to residuals above threshold
    threshold_quantile = 0.90  # Threshold for the GPD fit.
    u = np.quantile(z, threshold_quantile)
    exceedances = z[z > u] - u
    
    # MLE fit of GPD: shape xi, loc=0 fixed, scale sigma
    xi, _, sigma_gpd = genpareto.fit(exceedances, floc=0)
    
    # Step 3: Compute tail quantile of residuals
    n = len(z)
    Nu = np.sum(z > u)
    Fu = 1 - Nu / n
    
    # Pickands-Balkema-de Haan inversion
    Q_z = u + (sigma_gpd / xi) * (((1 - tau_n) / (1 - Fu))**(-xi) - 1)
    
    # Step 4: One-step-ahead conditional VaR
    sigma_next = res.forecast(horizon=1).variance.values[-1, 0]**0.5 / 100
    VaR = sigma_next * Q_z
    
    return VaR