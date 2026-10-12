"""This file contains Weissman extrapolation methods for estimating extreme conditional quantiles."""

import numpy as np

from src.methods.hill_estimator import hill_estimator_NCS, hill_estimator_CS



def weissman_extrapolate(tau_n, tau_0, alpha_0, gamma_hat):
    """Extrapolate an extreme quantile using the Weissman estimator."""
    
    alpha_n = (((1 - tau_0) / (1 - tau_n)) ** gamma_hat) * alpha_0
    
    return alpha_n



def cs_extreme_quantile(X, tau_n, residuals, k, beta_c):
    """Estimate extreme conditional quantiles using the CS method."""

    n = len(residuals)

    pos_res   = residuals[residuals > 0]     # Hill requires positive support
    sorted_res = np.sort(pos_res)
    n_pos_res = len(sorted_res)
    
    tau_0 = 1 - k / n

    # Threshold: (n-k)th order statistic of residuals = a_hat(tau_0)
    alpha_0 = sorted_res[n_pos_res - k - 1] 

    gamma_hat = hill_estimator_CS(sorted_res, k)

    alpha_n = weissman_extrapolate(tau_n, tau_0, alpha_0, gamma_hat)

    q_hat = alpha_n + X @ beta_c
    
    gamma_hat_vec = np.full(n, gamma_hat) # repeat scalar for consistent output shape
    
    return q_hat, gamma_hat_vec



def extreme_quantile_NCS(tau_n, quantiles, n, k, eta, tau_0, taus):
    """Estimate extreme conditional quantiles using the NCS method."""

    gamma_hat, q_base = hill_estimator_NCS(n, quantiles, taus, eta, tol=1e-4, min_prop=0.3)

    q_hat = weissman_extrapolate(tau_n, tau_0, q_base, gamma_hat)

    return q_hat, gamma_hat

    