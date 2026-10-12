"""This file contains the covariate selection method for the bitcoin application"""

import pandas as pd
import time

from itertools import combinations
from pathlib import Path

from src.empirical.rolling_window_back_test import rolling_window_back_test

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

start_time = time.perf_counter()


#--------------------------------------
# --- Load data ---
#--------------------------------------

data_path = Path(__file__).resolve().parents[2] / "data" / "btc_data.csv"
btc = pd.read_csv(data_path, index_col=0, parse_dates=True)


#--------------------------------------
# ---  Candidate covariates  ---
#--------------------------------------

candidates = ["vol_30", "drawdown", "lag_return", "EPU", "EURUSD", "VIX", "T10Y3M", "T5YIFR", "CORP_BOND_RET"]


#--------------------------------------
# ---  Correlation Check  ---
#--------------------------------------

print("\n" + "=" * 60)
print("CHECK 1: Correlation matrix of candidate covariates")
print("=" * 60)

corr = btc[candidates].corr()
print(corr.round(3))

# Flag high correlations
for c1, c2 in combinations(candidates, 2):
    if abs(corr.loc[c1, c2]) > 0.7:
        print(f"WARNING: {c1} and {c2} are highly correlated "
              f"({corr.loc[c1, c2]:.3f}) — consider dropping one")



#--------------------------------------
# ---  Out-of-sample quantile loss  ---
#--------------------------------------

print("\n" + "=" * 60)
print("CHECK 2: Out-of-sample quantile loss")
print("=" * 60)

WINDOW = 500
tau_n  = [0.99, 0.995]

shortlist = [
    "vol_30",
    "drawdown",
    "lag_return",
    "EPU",
    "EURUSD",
    "VIX",
    "T10Y3M",
    "T5YIFR",
    "CORP_BOND_RET"
]

all_combinations = (
    [[c] for c in shortlist] +
    [list(c) for c in combinations(shortlist, 2)] +
    [list(c) for c in combinations(shortlist, 3)]
)

records = []

for tau in tau_n:
    for combination in all_combinations:
        t0 = time.perf_counter()

        btc_QL = btc["2015-01-01":"2018-12-31"].copy()

        log_losses = btc_QL["log_loss"].values
        X = btc_QL[list(combination)].values

        scores, average_loss, _, _ = rolling_window_back_test(
            log_losses, X, tau, "2stage", WINDOW
        )

        elapsed_comb = time.perf_counter() - t0

        print(f"\ntau: {tau} | Covariate: {combination}")
        print(f"  Avg quantile loss: {average_loss:.6f}")
        print(f"  Runtime: {elapsed_comb:.2f}s")
        print("=" * 60)

        records.append({
            "tau":          tau,
            "covariates":   str(combination),
            "n_covariates": len(combination),
            "avg_loss":     average_loss,
        })

out_path = Path(__file__).resolve().parents[2] / "outputs" / "tables" / "covariate_selection.csv"

out_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

end_time = time.perf_counter()
elapsed = end_time - start_time

print(f"\nTotal runtime: {elapsed:.2f} seconds")
print(f"Total runtime: {elapsed/60:.2f} minutes")
