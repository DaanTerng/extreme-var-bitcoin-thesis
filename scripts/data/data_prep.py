"""Prepare the BTC data and macroeconomic covariates for the empirical study."""

import numpy as np
import pandas as pd
import yfinance as yf

from pathlib import Path
from fredapi import Fred


# ── Settings ──────────────────────────────────────────────

START    = "2015-01-01"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "btc_data.csv"


# ── BTC price and covariates ──────────────────────────────────────────────

btc = yf.download("BTC-USD", start=START, end="2026-06-01", auto_adjust=True)
btc.columns = btc.columns.get_level_values(0)

btc["log_return"] = np.log(btc["Close"]).diff()
btc["log_loss"]   = -btc["log_return"]
btc["vol_30"]     = btc["log_return"].rolling(30).std().shift(1)

# Compute drawdown relative to the cumulative return's running maximum.
cumret          = btc["log_return"].cumsum()

btc["drawdown"] = (cumret - cumret.cummax()).shift(1)
btc["lag_return"] = btc["log_return"].shift(1)

btc = btc.dropna()

btc_clean = btc[["Close", "log_return", "log_loss", "vol_30", "drawdown", "lag_return"]].rename(
    columns={"Close": "close"})


# ── Macro covariates ──────────────────────────────────────────────

FRED_API_KEY = "<YOUR_FRED_API_KEY>"

fred = Fred(api_key=FRED_API_KEY)

macro = pd.DataFrame({
    "EPU":       fred.get_series("USEPUINDXD", observation_start=START),
    "EURUSD":    fred.get_series("DEXUSEU",    observation_start=START),
    "VIX":       fred.get_series("VIXCLS",     observation_start=START),
    "T10Y3M":    fred.get_series("T10Y3M",     observation_start=START),
    "T5YIFR":    fred.get_series("T5YIFR",     observation_start=START),
})

# Daily returns of the iShares iBoxx Investment Grade Corporate Bond ETF.
lqd = yf.download("LQD", start=START, auto_adjust=True)["Close"].squeeze().pct_change()
lqd.name = "CORP_BOND_RET" 

macro = macro.join(lqd, how="left")

# Forward-fill the log EPU index, then difference and lag it.
macro["EPU"] = np.log(macro["EPU"]).ffill()  
macro["EPU"] = macro["EPU"].diff()             
macro["EPU"] = macro["EPU"].shift(1)          

# Forward-fill and lag the remaining macro covariates.
macro["CORP_BOND_RET"] = macro["CORP_BOND_RET"].ffill().shift(1)
macro["EURUSD"]        = macro["EURUSD"].ffill().shift(1)
macro["VIX"]           = macro["VIX"].ffill().shift(1)
macro["T10Y3M"]        = macro["T10Y3M"].ffill().shift(1)
macro["T5YIFR"]        = macro["T5YIFR"].ffill().shift(1)

macro.index = macro.index.tz_localize(None)

btc_clean = btc_clean.join(macro, how="left")


# ── Save data ──────────────────────────────────────────────

btc_clean.to_csv(OUT_PATH)

print(f"Saved {len(btc_clean)} rows from {btc_clean.index[0].date()} to {btc_clean.index[-1].date()}")