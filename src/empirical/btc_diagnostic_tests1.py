"""This file contains the 3 diagnostic checks to determine whether 3S and 2S are fit for the data"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.regression.quantile_regression as smq

from pathlib import Path

from src.empirical.hill_plot import plot_hill_path


#--------------------------------------
# --- Load data ---
#--------------------------------------

data_path = Path(__file__).resolve().parents[2] / "data" / "btc_data.csv"
btc = pd.read_csv(data_path, index_col=0, parse_dates=True)


#--------------------------------------
# --- Extract losses ---
#--------------------------------------

# Extract losses as a numpy array.
losses = btc["log_loss"].dropna().values.flatten()

# Hill only works on positive values.
losses = losses[losses > 0]


#--------------------------------------
# --- Check 1: Hill plot stability ---
#--------------------------------------

plot_hill_path(losses, k_min=10, k_max=200, gamma_true=None,
               title="", filename="hill_plot.png")


#--------------------------------------
# --- Check 2: QQ plot against Pareto ---
#--------------------------------------

n = len(losses)
sorted_losses = np.sort(losses)

# Take only the upper tail - top k observations
k = 150  # use stable region from Hill plot
tail = sorted_losses[n - k:]

gamma_hat = 0.37  
i = np.arange(1, k + 1)
theoretical = ((k + 1) / (k + 1 - i)) ** gamma_hat

# QQ plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(theoretical, tail, s=10, color="#08519C", alpha=0.7, label="BTC tail quantiles")

# Reference line - perfect Pareto fit would be a straight line through origin.
slope = np.linalg.lstsq(theoretical.reshape(-1, 1), tail, rcond=None)[0][0]
x_line = np.linspace(theoretical.min(), theoretical.max(), 100)
ax.plot(x_line, slope * x_line, color="#722F37", linestyle="--", label="Reference line", linewidth=1.5, markersize=4)

ax.set_xlabel("Theoretical Pareto quantiles", fontsize=12)
ax.set_ylabel("Empirical BTC loss quantiles", fontsize=12)
ax.set_title("")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#--------------------------------------
# --- Check 3: Linearity ---
#--------------------------------------

def linearity_check(data):
    
    btc = data.copy().dropna()

    Y = btc["log_loss"].values.flatten()
    X = sm.add_constant(np.column_stack([
        btc["vol_30"].values,
        btc["drawdown"].values,
        btc["lag_return"].values,
        btc["EPU"].values,
        btc["EURUSD"].values,
        btc["VIX"].values,
        btc["T10Y3M"].values,
        btc["T5YIFR"].values,
        btc["CORP_BOND_RET"].values
    ]))

    taus = np.arange(0.90, 1.00, 0.01)
    betas = []

    for tau in taus:
        model = smq.QuantReg(Y, X)
        result = model.fit(q=tau)
        betas.append(result.params)

    betas = np.array(betas)  

    covariate_names = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
    
    fig, axes = plt.subplots(1, 9, figsize=(18, 4))
    
    for i, (ax, name) in enumerate(zip(axes, covariate_names)):
        ax.plot(taus, betas[:, i + 1], marker="o", color="#08519C", linewidth=1.5, markersize=4)
        ax.set_title(f"{name}")
        ax.set_xlabel(r"$\tau$", fontsize=12)
        ax.set_ylabel(r"$\hat{\beta}(\tau)$", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    plt.suptitle("", y=1.02)
    plt.tight_layout()
    plt.show()

linearity_check(btc)
