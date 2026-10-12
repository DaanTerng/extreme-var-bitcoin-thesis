"""This file contains the regime classification method."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from joblib import Parallel, delayed
from pathlib import Path

from src.methods.quantile_regression import fit_qr


#--------------------------------------
# --- Load data ---
#--------------------------------------

data_path = Path(__file__).resolve().parents[2] / "data" / "btc_data.csv"
btc = pd.read_csv(data_path, index_col=0, parse_dates=True)

btc = btc["2019-01-01":"2026-06-01"]

Y = btc["log_loss"].values.flatten()
X = sm.add_constant(np.column_stack([
    btc["T5YIFR"].values.flatten(),
    btc["lag_return"].values.flatten()
]))


#--------------------------------------
# --- Nonlinearity score ---
#--------------------------------------

def nonlinearity_score(y, X, taus):
    """Compute total variation in slopes across quantile levels."""

    betas = []
    
    for tau in taus:
        beta = fit_qr(y, X, tau)
        betas.append(beta)

    betas = np.array(betas)          

    slope_betas = betas[:, 1:]        
    tv_per_covariate = np.sum(np.abs(np.diff(slope_betas, axis=0)), axis=0)

    return np.sum(tv_per_covariate)
  

#--------------------------------------
# --- Rolling nonlinearity score ---
#--------------------------------------

WINDOW = 500
TAUS   = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99]

print(f"Computing rolling nonlinearity score "
      f"(window={WINDOW}, {len(btc) - WINDOW} steps)...")

nl_scores = Parallel(n_jobs=-1)(
    delayed(nonlinearity_score)(Y[t - WINDOW: t], X[t - WINDOW: t], TAUS)
    for t in range(WINDOW, len(btc))
)

nl_scores = np.array(nl_scores)
dates     = pd.DatetimeIndex(btc.index[WINDOW:])

print(f"Done. Score range: [{nl_scores.min():.4f}, {nl_scores.max():.4f}]")
print(f"Median score: {np.median(nl_scores):.4f}")


#--------------------------------------
# --- Regime classification ---
#--------------------------------------

# Primary split: median threshold
threshold_median = np.median(nl_scores)                                            
regime_median    = np.where(nl_scores > threshold_median, "nonlinear", "linear")

# Robustness splits: tertiles (bottom third = linear, top third = nonlinear)
p33 = np.percentile(nl_scores, 33)  
p67 = np.percentile(nl_scores, 67)

regime_tertile = np.where(nl_scores > p67, "nonlinear",        
                 np.where(nl_scores < p33, "linear", "middle")
                 )

n_nonlinear = (regime_median == "nonlinear").sum()
n_linear    = (regime_median == "linear").sum()
print(f"\nMedian split — nonlinear: {n_nonlinear} windows "
      f"({100*n_nonlinear/len(nl_scores):.1f}%), "
      f"linear: {n_linear} windows ({100*n_linear/len(nl_scores):.1f}%)")

n_nl_t  = (regime_tertile == "nonlinear").sum()
n_lin_t = (regime_tertile == "linear").sum()
n_mid_t = (regime_tertile == "middle").sum()
print(f"Tertile split  — nonlinear: {n_nl_t}, middle: {n_mid_t}, "
      f"linear: {n_lin_t}")


#--------------------------------------
# --- Store regime information ---
#--------------------------------------

btc_out = btc.iloc[WINDOW:].copy()
btc_out["nl_score"]         = nl_scores
btc_out["regime_median"]    = regime_median
btc_out["regime_tertile"]   = regime_tertile


#--------------------------------------
# --- Plots ---
#--------------------------------------

fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True)

fig.suptitle("Rolling Nonlinearity Score — Bitcoin (window = 500 days)",
             fontsize=13, fontweight="bold")

# Panel 1: raw score over time
axes[0].plot(dates, nl_scores, color="steelblue", linewidth=0.8)
axes[0].axhline(threshold_median, color="firebrick", linestyle="--",
                label=f"Median threshold ({threshold_median:.4f})")
axes[0].axhline(p33, color="orange", linestyle=":", label=f"33rd pct ({p33:.4f})")
axes[0].axhline(p67, color="orange", linestyle=":", label=f"67th pct ({p67:.4f})")
axes[0].set_ylabel("NL Score")
axes[0].legend(fontsize=8)
axes[0].set_title("Nonlinearity Score Over Time")

# Panel 2: regime classification (median split)
colors = ["#d62728" if r == "nonlinear" else "#1f77b4"
          for r in regime_median]
axes[1].bar(dates, nl_scores, color=colors, width=1.5, alpha=0.7)
axes[1].axhline(threshold_median, color="black", linestyle="--", linewidth=0.8)
axes[1].set_ylabel("NL Score")
axes[1].set_title("Regime Classification (Median Split) — "
                  "Red: Nonlinear, Blue: Linear")

# Panel 3: regime classification (tertile split)
colors = ["#d62728" if r == "nonlinear" else "#1f77b4" if r == "linear" else "black"
          for r in regime_tertile]
axes[2].bar(dates, nl_scores, color=colors, width=1.5, alpha=0.7)
axes[2].axhline(p33, color="orange", linestyle=":", label=f"33rd pct ({p33:.4f})")
axes[2].axhline(p67, color="orange", linestyle=":", label=f"67th pct ({p67:.4f})")
axes[2].set_ylabel("NL Score")
axes[2].set_title("Regime Classification (Tertile Split) — "
                  "Red: Nonlinear, Blue: Linear, Black: Middle")

# Panel 4: BTC price for reference
axes[3].plot(btc_out.index, btc_out["close"], color="black", linewidth=0.6)
axes[3].set_ylabel("BTC Price (USD)")
axes[3].set_title("Bitcoin Price (reference)")
axes[3].set_xlabel("Date")
plt.tight_layout()
plt.show()


#--------------------------------------
# --- Feasibility check ---
#--------------------------------------

print("\n" + "=" * 60)
print("FEASIBILITY CHECK")
print("=" * 60)

print("For the subsample VaR comparison you need enough exceedances per regime.")
print("At tau=0.99, expected exceedances per window of 500 days = ~5")
print(f"Number of nonlinear windows (median split): {n_nonlinear}")
print(f"Number of linear windows (median split):    {n_linear}")

if n_nonlinear < 50 or n_linear < 50:
    print("\nWARNING: one regime has very few windows — "
          "consider a larger window or lower tau for subsample analysis.")
else:
    print("\nBoth regimes are sufficiently populated for subsample backtesting.")


#--------------------------------------
# --- Save regime labels ---
#--------------------------------------

out_path = Path(__file__).resolve().parents[2] / "data" / "btc_regimes.csv"
btc_out[["nl_score", "regime_median", "regime_tertile"]].to_csv(out_path)
print(f"\nRegime labels saved to {out_path}")










