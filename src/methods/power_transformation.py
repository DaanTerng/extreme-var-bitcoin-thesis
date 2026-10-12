"""This file contains Box-Cox transformations and procedures for estimating transformed conditional quantiles."""

import numpy as np

from src.methods.quantile_regression import fit_qr_multiple



def box_cox(lamb, y):
    """Apply the Box-Cox transformation."""
    
    with np.errstate(invalid='ignore', divide='ignore'):
        if lamb == 0:
            return np.log(y)
        else:
            return (y**lamb - 1) / lamb



def box_cox_inv(lamb, z):
    """Apply the inverse Box-Cox transformation."""

    if lamb == 0:
        return np.exp(z)   
    else:
        base = z * lamb + 1
        
        if lamb == 1:
            return base
        else:
            return base ** (1/lamb)
        
        

def squared_lack_of_fit(X, residuals, taus):
    """Compute the squared lack-of-fit criterion for the quantile estimates."""    
    
    n = X.shape[0]
    
    # Identify observations with covariates below each prediction point.
    A = (X[:, None, :] < X[None, :,:]).all(axis=2)
   
    score = taus - (residuals <=0) # shape (n, k)
    
    # Compute the lack-of-fit criterion for each prediction point.
    R_n = A.T @ score / n # shape (n, k)
    
    return (R_n ** 2).sum()



def estimate_lambda(y, X, taus, lambdas, a=0.0):
    """Estimate the Box-Cox transformation parameter using the lack-of-fit criterion."""

    n = len(y)
    X_fit = np.column_stack([np.ones(n), X])

    squared_lack_of_fit_scores = []

    for lamb in lambdas:
        transf_response = box_cox(lamb, y + a)

        if not np.all(np.isfinite(transf_response)):
            squared_lack_of_fit_scores.append(np.inf)
            continue
        
        theta_hat = fit_qr_multiple(transf_response, X_fit, taus)
        fitted    = X_fit @ theta_hat.T
        residuals = transf_response[:, None] - fitted
        residuals = np.round(residuals, 10)  # match R's rounding

        squared_lack_of_fit_scores.append(squared_lack_of_fit(X, residuals, taus))

    lamb_hat = lambdas[np.argmin(squared_lack_of_fit_scores)]
    
    return lamb_hat



def compute_all_fitted_quantiles(X, y, lamb, taus, tol=1e-4, x_eval=None, a=0.0):
    """Compute fitted conditional quantiles after the Box-Cox transformation."""

    transf_response = box_cox(lamb, y + a)
    n = len(y)
    X_fit = np.column_stack([np.ones(n), X])
    
    theta_hat = fit_qr_multiple(transf_response, X_fit, taus)  

    if x_eval is None:
        X_eval = X_fit                                           
    else:
        x_eval = np.atleast_2d(x_eval)                          
        X_eval = np.column_stack([np.ones(len(x_eval)), x_eval])  

    Lam_Q = X_eval @ theta_hat.T                                

    if lamb == 0:
        Q = np.exp(Lam_Q) - a
    else:
        tt = (Lam_Q * lamb + 1).astype(float)

        if lamb < 0:
            pos_vals = tt[tt > 0]
            tol2 = max(tol, np.nanquantile(pos_vals, 0.002)) if len(pos_vals) > 0 else tol
        else:
            tol2 = tol

        # Discard rows where the baseline quantile is numerically invalid.
        tt[np.abs(tt[:, 0]) < tol] = np.nan

        # Discard entire row if >30% of values fall below threshold
        tt[np.nanmean(tt < tol2, axis=1) > 0.3] = np.nan

        # Set individual sub-threshold values to NaN
        tt = np.where(tt < tol2, np.nan, tt)

        with np.errstate(invalid='ignore'):
            Q = tt ** (1 / lamb) - a

    return Q  