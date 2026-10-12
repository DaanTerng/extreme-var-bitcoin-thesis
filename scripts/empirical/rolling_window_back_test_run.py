"""Run the rolling window backtest for the empirical thesis study."""


import time
import sys
import pandas as pd
import numpy as np

from pathlib import Path

from src.empirical.rolling_window_back_test import rolling_window_back_test
from src.methods.back_test import diebold_mariano, kupiec_test



class Tee:
    """Write output to both the console and a text file."""

    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
        
    def write(self, msg):
        self.stream.write(msg)
        
        try:
            self.file.write(msg)
        except Exception:
            pass
        
    def flush(self):
        self.stream.flush()
        
        try:
            self.file.flush()
        except Exception:
            pass


# ── File paths ──────────────────────────────────────────────

forecasts_out = Path(__file__).resolve().parents[2] / "outputs" / "tables" / "var_forecasts.csv"
results_out   = Path(__file__).resolve().parents[2] / "outputs" / "tables" / "backtest_results.txt"
results_out.parent.mkdir(parents=True, exist_ok=True)

data_path    = Path(__file__).resolve().parents[2] / "data" / "btc_data.csv"
regimes_path = Path(__file__).resolve().parents[2] / "data" / "btc_regimes.csv"


# ── Load data ──────────────────────────────────────────────

btc            = pd.read_csv(data_path,    index_col=0, parse_dates=True)
regime_median  = pd.read_csv(regimes_path, index_col=0, parse_dates=True)["regime_median"]
regime_tertile = pd.read_csv(regimes_path, index_col=0, parse_dates=True)["regime_tertile"]

btc = btc["2019-01-1":"2026-06-01"]
  
log_losses = btc["log_loss"].values
X          = btc[["T5YIFR", "lag_return"]].values


# ── Settings ──────────────────────────────────────────────

WINDOW     = 500
tau_n_list = [0.99, 0.995, 0.999]
models     = ["GARCH11_EVT", "2stage", "3stage"]

score_dates = btc.index[WINDOW:]


# ── Regime classifications ──────────────────────────────────────────────

# Align the regime classifications with the forecast dates.
aligned_median  = regime_median.reindex(score_dates)
aligned_tertile = regime_tertile.reindex(score_dates)

regime_masks = {
    "Overall":        np.ones(len(score_dates), dtype=bool),
    "Med-Linear":     (aligned_median  == "linear").values,
    "Med-Nonlinear":  (aligned_median  == "nonlinear").values,
    "Tert-Linear":    (aligned_tertile == "linear").values,
    "Tert-Nonlinear": (aligned_tertile == "nonlinear").values,
    "Tert-Middle":    (aligned_tertile == "middle").values,
}


# ── Save console output ──────────────────────────────────────────────

_log_file = open(results_out, "w", encoding="utf-8")
_original_stdout = sys.stdout
sys.stdout = Tee(_log_file, _original_stdout)

start_time = time.perf_counter()

forecast_rows = []  



#--------------------------------------
# --- Rolling window backtest ---
#--------------------------------------

for tau in tau_n_list:

    all_scores     = {}  
    all_violations = {}   

    # ── Run all models ────────────────────────────────────────────────────────
    
    for model in models:
        scores, average_loss, violations, forecasts = rolling_window_back_test(log_losses, X, tau, model, WINDOW)

        scores     = np.array(scores[:len(score_dates)],     dtype=float)
        violations = np.array(violations[:len(score_dates)], dtype=float)
        forecasts  = np.array(forecasts[:len(score_dates)],  dtype=float)
        
        all_scores[model]     = scores
        all_violations[model] = violations

        # Collect forecasts for the CSV output.
        for date, fc in zip(score_dates, forecasts):
            forecast_rows.append({"tau": tau, "model": model, "date": date, "VaR": fc})
        
        scores_s = pd.Series(scores, index=score_dates)

        linear_mask_median     = aligned_median  == "linear"
        nonlinear_mask_median  = aligned_median  == "nonlinear"
        linear_mask_tertile    = aligned_tertile == "linear"
        nonlinear_mask_tertile = aligned_tertile == "nonlinear"
        middle_mask_tertile    = aligned_tertile == "middle"


        # ── Print average losses ──────────────────────────────────────────────

        print(f"\nModel: {model}  tau: {tau}")
        print(f"  Overall avg loss:    {average_loss:.6f}")
        print(f"  Median - Linear regime:       {np.nanmean(scores_s[linear_mask_median]):.6f}  ({linear_mask_median.sum()} windows)")
        print(f"  Median - Nonlinear regime:    {np.nanmean(scores_s[nonlinear_mask_median]):.6f}  ({nonlinear_mask_median.sum()} windows)")
        print(f"  Tertile - Linear regime:       {np.nanmean(scores_s[linear_mask_tertile]):.6f}  ({linear_mask_tertile.sum()} windows)")
        print(f"  Tertile - Nonlinear regime:    {np.nanmean(scores_s[nonlinear_mask_tertile]):.6f}  ({nonlinear_mask_tertile.sum()} windows)")
        print(f"  Tertile - Medium regime:       {np.nanmean(scores_s[middle_mask_tertile]):.6f}  ({middle_mask_tertile.sum()} windows)")


    # ── Kupiec tests ──────────────────────────────────────────────────────────
    
    print(f"\n{'='*60}")
    print(f"Kupiec POF tests  tau: {tau}  (expected violation rate: {1-tau:.3f})")
    print(f"  H0: observed rate = {1-tau:.3f}  (* p<0.10  ** p<0.05  *** p<0.01)")
    print(f"{'='*60}")

    for model in models:
        viols = all_violations[model]
        lr, p, x, n, p_hat = kupiec_test(viols, tau)
        
        stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        
        print(f"  {model:<20}: violations={x}/{n} ({p_hat:.3f})  LR={lr:.3f}  p={p:.3f} {stars}")

    # ── Diebold-Mariano tests ─────────────────────────────────────────────────
    
    # A negative DM statistic indicates lower average loss for model 1.
    print(f"\n{'='*60}")
    print(f"Diebold-Mariano tests  tau: {tau}  (h={WINDOW}, bw=499 lags)")
    print("  (DM < 0 -> model 1 better;  * p<0.10  ** p<0.05  *** p<0.01)")
    print(f"{'='*60}")

    pairs = [
        ("3stage",      "GARCH11_EVT"),   
        ("3stage",      "2stage"),       
        ("2stage",      "GARCH11_EVT"),  
    ]

    for m1, m2 in pairs:
        s1 = all_scores[m1]
        s2 = all_scores[m2]
        
        print(f"\n  {m1} vs {m2}:")
        
        for label, mask in regime_masks.items():
            dm, p = diebold_mariano(s1[mask], s2[mask], h=WINDOW)
            
            if np.isnan(p):
                print(f"    {label:<20}: insufficient data")
                continue
            
            stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
            winner = m1 if dm < 0 else m2
            
            print(f"    {label:<20}: DM={dm:+.3f}  p={p:.3f} {stars:<3}  ({winner} better)")


# ── Save forecasts ──────────────────────────────────────────────

pd.DataFrame(forecast_rows).to_csv(forecasts_out, index=False)
print(f"\nVaR forecasts saved to {forecasts_out}")

end_time = time.perf_counter()

print(f"\nTotal runtime: {(end_time - start_time):.2f} seconds")
print(f"Total runtime: {(end_time - start_time)/60:.2f} minutes")

sys.stdout = _original_stdout
_log_file.close()

print(f"Results saved to {results_out}")



















