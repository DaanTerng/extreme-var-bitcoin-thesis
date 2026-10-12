"""This file contains pipelines for fitting and evaluating the QR, CS, 2Stage, and 3Stage estimators."""

import numpy as np

from src.methods.weissman_extrapolation import cs_extreme_quantile
from src.methods.quantile_regression import fit_composite_slope, compute_residuals
from src.methods.weissman_extrapolation import extreme_quantile_NCS
from src.methods.quantile_regression import fit_qr_multiple
from src.methods.power_transformation import (estimate_lambda, compute_all_fitted_quantiles, box_cox)
from src.methods.select_k import select_k
            


def run_qr(tau_n, data, x_eval=None):
    """Fit quantile regression and compute conditional quantiles."""
    
    y, X = data
    n = len(y)
    X_fit = np.column_stack([np.ones(n), X])
    
    beta_qr = fit_qr_multiple(y, X_fit, [tau_n])[0]  
    
    if x_eval is not None:
        quantiles_at_x = beta_qr @ x_eval
        
        return quantiles_at_x     

    quantiles_qr       = X_fit @ beta_qr.T

    return quantiles_qr            



def run_cs(tau_n, k, tau_U, data, x_eval=None):
    """Estimate extreme quantiles using the composite slope method."""

    y, X = data
    n, p = X.shape
    
    taus = np.linspace(0.90, tau_U, 10)
    
    beta_c, alphas = fit_composite_slope(y, X, taus)
    residuals = compute_residuals(y, X, beta_c)
    
    if x_eval is not None:
        x_pred = np.asarray(x_eval)
        q_hat_arr, gamma_arr = cs_extreme_quantile(x_eval[None, :], tau_n, residuals, k, beta_c)
        
        return q_hat_arr, gamma_arr

    q_hat, gamma_hat = cs_extreme_quantile(X, tau_n, residuals, k, beta_c)
    
    return q_hat, gamma_hat



def run_2stage(data, tau_n, k_2s, n, eta, x_eval=None, seed=None):
    """Estimate extreme quantiles using the 2Stage estimator."""

    y, X = data

    tau_0_ncs   = (n - k_2s) / (n + 1)
    max_tau_ncs = (n - int(n**eta)) / (n + 1)

    taus_ncs    = np.linspace(tau_0_ncs, max_tau_ncs, k_2s)
    X_fit       = np.column_stack([np.ones(n), X])
    beta_ncs    = fit_qr_multiple(y, X_fit, taus_ncs)

    if x_eval is not None:
        x_eval         = np.concatenate([[1], x_eval])
        quantiles_at_x = beta_ncs @ x_eval
        q_hat, gamma_hat = extreme_quantile_NCS(tau_n, quantiles_at_x, n, k_2s, eta, tau_0_ncs, taus_ncs)
        
        return q_hat[0], gamma_hat[0]

    quantiles_ncs = X_fit @ beta_ncs.T
    q_hat, gamma_hat = extreme_quantile_NCS(tau_n, quantiles_ncs, n, k_2s, eta, tau_0_ncs, taus_ncs)
    
    return q_hat, gamma_hat

    
    
def run_3stage(tau_n, eta, data, n, x_eval=None, seed=None, lamb=None):
    """Estimate extreme quantiles using the 3Stage estimator."""

    y, X = data

    # Shift the response to ensure positive values for the Box-Cox transformation.
    a = max(0.0, float(-y.min()) + 1e-6)

    if lamb is None:
        lambdas = np.round(np.arange(-2.0, 2.1, 0.1), 1)

        if n == 500:
            k_range    = range(20, 71)
            eps_stage1 = 0.15
        elif n == 2000:
            k_range    = range(30, 101)
            eps_stage1 = 0.10

        tau_stage1 = np.array([1 - eps_stage1])
        lamb = estimate_lambda(y, X, tau_stage1, lambdas, a=a)
    else:
        if n <= 600:
            k_range = range(20, 71)
        else:
            k_range = range(30, 101)

    max_tau  = (n - int(n**eta)) / (n + 1)
    y_transf = box_cox(lamb, y + a)

    # Select the number of upper order statistics.
    k    = select_k(y, X, y_transf, lamb, a, max_tau, k_range, n)

    tau_0 = (n - k) / (n + 1)
    taus  = np.linspace(tau_0, max_tau, k)

    if x_eval is not None:
        raw_q_pred = compute_all_fitted_quantiles(X, y, lamb, taus, x_eval=x_eval, a=a)
        q_hat, gamma_hat = extreme_quantile_NCS(tau_n, raw_q_pred, n, k, eta, tau_0, taus)
        
        return q_hat[0], gamma_hat[0], lamb, k

    raw_quantiles = compute_all_fitted_quantiles(X, y, lamb, taus, a=a)

    q_hat, gamma_hat = extreme_quantile_NCS(tau_n, raw_quantiles, n, k, eta, tau_0, taus)

    return q_hat, gamma_hat, lamb, k
    