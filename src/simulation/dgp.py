"""This file contains data-generating processes for the simulation studies."""

import numpy as np

from src.methods.power_transformation import box_cox_inv



def generate_data_2012(n, r, case, seed=None):
    """Generate data for the simulation study of Wang, Li & He (2012)."""
   
    rng = np.random.default_rng(seed)

    X = rng.uniform(-1, 1, size=(n, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]

    if case == 1:
        # Pareto(0.2): F(x) = 1 - x^(-1/gamma) for x >= 1.
        gamma_true = 0.2
        eps = rng.pareto(1 / gamma_true, size=n) + 1
        
    elif case == 2:
        # Pareto(0.5): gamma = 0.5
        gamma_true = 0.5
        eps = rng.pareto(1 / gamma_true, size=n) + 1
        
    elif case == 3:
        # Absolute Cauchy distribution with gamma = 1.
        gamma_true = 1.0
        eps = np.abs(rng.standard_cauchy(size=n)) + 1  
    
    else:
        raise ValueError(f"case must be 1, 2, or 3. Got {case}.")

    y = x1 + x2 + (1 + r * x1) * eps

    return y, X

    

def generate_data_2013(n, case, seed=None):
    """Generate data for the simulation study of Wang & Li (2013)."""
    
    rng = np.random.default_rng(seed)

    # Generate covariates: x_i = (1, x_i1, x_i2)
    x = np.column_stack([
        rng.uniform(-1, 1, n),
        rng.uniform(-1, 1, n)
    ])
    
    x_full = np.column_stack([np.ones(n), x])

    if case == 1:
        lam = 1
        beta = np.array([2, 2, 2])
        sigma = np.array([2, 1.6, 0])
        
        u = rng.uniform(0, 1, n) 
        eps = (1 - u) ** (-1) 
        
    elif case == 2:
        lam = 0.5
        beta = np.array([2, 2, 2])
        sigma = np.array([2, 1.6, 0])
        
        u = rng.uniform(0, 1, n) 
        eps = (1 - u) ** (-0.5) 
        
    elif case == 3:
        lam = -1
        beta = np.array([-2, -1, -1])
        sigma = np.array([3, 1, 1])
        
        eps = rng.uniform(0, 1, n)

    elif case == 4:
        lam = 0
        beta = np.array([2, 1, 1])
        sigma = np.array([0.5, 0.25, 0])
        
        # Q(tau) = tau - 1 - log(1 - tau)
        u = rng.uniform(0, 1, n)
        eps = u - 1 - np.log(1 - u)
               
    elif case == 5:
        # Univariate x_i with covariate-dependent EVI:
        # gamma(x_i) = exp(-1 + x_i).
        x1 = rng.uniform(-1, 1, n)
        gamma_x = np.exp(-1 + x1)
        
        # Pareto with shape 1/gamma(x_i), generated per observation
        u = rng.uniform(0, 1, n)
        y = (1 - u) ** (-gamma_x) - 1 
        
        return y, x1.reshape(-1, 1)

    elif case == 6:
        # y_i | x_i ~ Frechet with gamma(x_i)
        x1 = rng.uniform(-1, 1, n)
        gamma_x = 0.5 * (1/10 + np.sin(np.pi * (x1 + 1) /2)) * (11/10 - 0.5 * np.exp(-16 * x1**2))
        # Frechet inverse CDF: F(y) = exp(-y^{-1/gamma}) -> y = (-log(u))^{-gamma}

        u = rng.uniform(0, 1, n)
        y = (-np.log(u)) ** (-gamma_x)
        
        return y, x1.reshape(-1, 1)

    linear = x_full @ beta + (x_full @ sigma) * eps
    y = box_cox_inv(lam, linear)
        
    return y, x



def generate_data_simu(n, case, seed=None):
    """Generate data for the simulation study in the thesis."""
    
    rng = np.random.default_rng(seed)

    # Generate covariates: x_i = (1, x_i1, x_i2)
    x = np.column_stack([
        rng.uniform(-1, 1, n),
        rng.uniform(-1, 1, n)
    ])
    
    x_full = np.column_stack([np.ones(n), x])

    if case == 1:
        lam = 1
        beta = np.array([2, 1, 1])
        sigma = np.array([1, 0.5, 0])

        u = rng.uniform(0, 1, n)
        eps = (1 - u) ** (-0.5)

    elif case == 2:
        lam = 0.5
        beta = np.array([2, 1, 1])
        sigma = np.array([1, 0.5, 0])
        
        u = rng.uniform(0, 1, n)
        eps = (1 - u) ** (-0.25)
        
    elif case == 3:
        # Log transformation with constant EVI = 0.5.
        lam = 0
        beta = np.array([2, 1, 1])
        sigma = np.array([1, 0.5, 0])

        u = rng.uniform(0, 1, n)
        eps = u - 1 - np.log(1 - u)

        linear = x_full @ beta + x_full @ sigma + 0.5 * eps
        y = box_cox_inv(lam, linear)

        return y, x
    
    elif case == 4:
        # Covariate-dependent EVI of the transformed response.
        lam = 0.5
        beta = np.array([2, 1, 1])
        sigma = np.array([0.5, 0.25, 0])
        
        gamma_x = x_full @ sigma
        u = rng.uniform(0, 1, n)
        eps = (1 - u) ** (-gamma_x)
        
        linear = x_full @ beta + eps
        y = box_cox_inv(lam, linear)
        
        return y, x

    elif case == 5:
        lam = 0
        beta = np.array([2, 1, 1])
        sigma = np.array([0.5, 0.25, 0])
        
        u = rng.uniform(0, 1, n)
        eps = u - 1 - np.log(1 - u)

    elif case == 6:
        # Univariate x_i with covariate-dependent EVI:
        # gamma(x_i) = exp(x_i - 1).
        x1 = rng.uniform(-1, 1, n)
        gamma_x = np.exp(x1 - 1)
        
        u = rng.uniform(0, 1, n)
        y = (-np.log(u)) ** (-gamma_x)
        
        return y, x1.reshape(-1, 1)

    elif case == 7:
        # Fréchet distribution with covariate-dependent EVI:
        # gamma(x_i) = 0.5 + 0.3 sin(pi x_i).        
        x1 = rng.uniform(-1, 1, n)
        gamma_x = 0.5 + 0.3 * np.sin(np.pi * x1)
        
        u = rng.uniform(0, 1, n)
        y = (-np.log(u)) ** (-gamma_x)
        
        return y, x1.reshape(-1, 1)

    linear = x_full @ beta + (x_full @ sigma) * eps
    y = box_cox_inv(lam, linear)
    
    return y, x


