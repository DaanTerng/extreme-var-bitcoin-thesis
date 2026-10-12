"""This file contains quantile regression and composite slope estimation methods."""

import numpy as np

from scipy.optimize import linprog
from scipy.sparse import csr_matrix, kron, eye as speye
from scipy.sparse import hstack as sp_hstack
from statsmodels.regression.quantile_regression import QuantReg



def quantile_loss(u, tau):
    """Pinball loss at quantile level tau."""
    return u * (tau - (u < 0).astype(float))



def fit_qr(y, X, tau):
    """Fit a linear quantile regression model at quantile level tau."""

    n, p = X.shape
    
    # LP formulation using positive and negative residual components:
    # y - X @ beta = u - v, with u, v >= 0.

    c = np.concatenate([np.zeros(p), tau * np.ones(n), (1 - tau) * np.ones(n)])
    

    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    b_eq = y

    # Bounds: beta unbounded, u >= 0, v >= 0
    bounds = [(None, None)] * p + [(0, None)] * n + [(0, None)] * n

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.status != 0:
        raise RuntimeError(f"QR LP did not converge: {result.message}")

    beta = result.x[:p]
    
    return beta



def fit_qr_multiple(y, X, taus):
    """Fit linear quantile regressions at multiple quantile levels."""

    return np.array([fit_qr(y, X, tau) for tau in taus])



def predict_quantile(X, beta):
    """Compute fitted quantile: X @ beta."""
    
    return X @ beta



def fit_qr_multiple_p(y, X, taus):
    """Fit quantile regressions at multiple levels using statsmodels."""
    
    model = QuantReg(y, X)
    
    return np.array([
        model.fit(q=tau, method="br", max_iter=500000).params
        for tau in taus
    ])



def fit_composite_slope(y, X, taus):
    """Estimate common slopes using composite quantile regression."""

    n, p = X.shape
    taus = np.asarray(taus)
    l = len(taus)

    # Objective function.
    c_beta = np.zeros(p)
    c_alpha = np.zeros(l)
    c_u = np.repeat(taus, n)        # tau_j repeated n times for each j
    c_v = np.repeat(1 - taus, n)
    c = np.concatenate([c_beta, c_alpha, c_u, c_v])

    # Sparse equality constraint matrix.
    b_eq = np.tile(y, l)

    I_l = speye(l, format='csr')
    I_n = speye(n, format='csr')

    beta_block  = csr_matrix(np.tile(X, (l, 1)))             
    alpha_block = kron(I_l, np.ones((n, 1)), format='csr')   
    u_block     =  kron(I_l, I_n, format='csr')              
    v_block     = -kron(I_l, I_n, format='csr')               

    A_eq = sp_hstack([beta_block, alpha_block, u_block, v_block], format='csr')

    bounds = [(None, None)] * (p + l) + [(0, None)] * (2 * l * n)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.status != 0:
        raise RuntimeError(f"Composite slope LP did not converge: {result.message}")

    beta_c = result.x[:p]
    alphas = result.x[p:p + l]

    return beta_c, alphas



def compute_residuals(y, X, beta_c):
    """Compute pseudo-residuals using the common slope estimate."""

    return y - X @ beta_c

