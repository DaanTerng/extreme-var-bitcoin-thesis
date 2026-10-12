"""This file contains methods for selecting the number of upper order statistics k."""

import numpy as np

from src.methods.power_transformation import box_cox_inv
from src.methods.quantile_regression import fit_qr_multiple



def _hill_per_row(Q):
    """Compute the Hill estimator for each row of Q."""

    n_rows = Q.shape[0]
    out = np.full(n_rows, np.nan)
    
    for i in range(n_rows):
        pos = np.sort(Q[i][Q[i] > 0])
        
        if len(pos) < 2:
            continue
        
        out[i] = np.mean(np.log(pos / pos[0]))
    
    return out



def _m2_per_row(Q):
    """Compute the second log-moment estimator for each row of Q."""

    n_rows = Q.shape[0]
    out = np.full(n_rows, np.nan)
    
    for i in range(n_rows):
        pos = np.sort(Q[i][Q[i] > 0])
        
        if len(pos) < 2:
            continue
        
        out[i] = np.mean(np.log(pos / pos[0]) ** 2)
    
    return out



def select_k(y, x, Lam_y, lam, a, max_tau, grid_k, n):
    """Select k using the moment-based criterion of Wang & Li (2013)."""

    X_fit = np.column_stack([np.ones(n), x])

    # Keep candidate k values above the minimum threshold.
    grid_k = [k for k in grid_k if k > int((1 - max_tau) * n) + 1]

    obj = []
    
    for k in grid_k:
        tau_0 = (n - k) / (n + 1)
        taus  = np.linspace(tau_0, max_tau, k)

        # Fit quantile regression at the intermediate quantile levels.
        beta  = fit_qr_multiple(Lam_y, X_fit, taus)  
        Lam_Q = X_fit @ beta.T                         

        # Back-transform the fitted quantiles to the original scale.
        with np.errstate(invalid='ignore'):
            Q = box_cox_inv(lam, Lam_Q) - a             

        # Compute Hill and moment estimators on both scales.
        gamma_x = _hill_per_row(Q)                       
        Mn1 = _hill_per_row(Lam_Q)                      
        Mn2 = _m2_per_row(Lam_Q)                        

        # Compute the moment-based extreme value index estimate.
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma_minus = 1 - 0.5 / (1 - Mn1**2 / Mn2) 
        
        gamma_star = Mn1 + gamma_minus                   

        # Compute the selection criterion.
        diff = lam * gamma_x - gamma_star
        obj.append(np.nanmean(diff**2))

    k_star = grid_k[int(np.argmin(obj))]
    
    return k_star
