"""Plot the BTC price and log-loss series for the empirical thesis study."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms

from pathlib import Path


# ── File paths ──────────────────────────────────────────────

data_path = Path(__file__).resolve().parents[2] / "data" / "btc_data.csv"
out_dir   = Path(__file__).resolve().parents[2] / "outputs" / "figures"


# ── Load data ──────────────────────────────────────────────

btc = pd.read_csv(data_path, index_col=0, parse_dates=True)
btc = btc["2015-01-01":"2026-06-01"]   # full sample


# ── Period settings ──────────────────────────────────────────────

split_date  = pd.Timestamp("2019-01-01")
start_date  = pd.Timestamp("2015-01-01")



#--------------------------------------
# --- Plot BTC series ---
#--------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)


# ── BTC price ──────────────────────────────────────────────

ax1.plot(btc.index, btc["close"], color="#08519C", linewidth=0.5, alpha=0.85)
ax1.set_ylabel("Price (USD)", fontsize=11)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.tick_params(axis="y", labelsize=10)

# Mark the boundary between the screening and evaluation periods.
ax1.axvline(split_date, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

# Position labels using dates horizontally and axes fractions vertically.
trans1 = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)

ax1.text(start_date + pd.Timedelta(days=30), 0.96, "Screening period",
         transform=trans1, fontsize=10, va="top", ha="left", color="black", alpha=0.7)

ax1.text(split_date + pd.Timedelta(days=30), 0.96, "Evaluation period",
         transform=trans1, fontsize=10, va="top", ha="left", color="black", alpha=0.7)


# ── BTC log-loss ──────────────────────────────────────────────

ax2.plot(btc.index, btc["log_loss"], color="#08519C", linewidth=0.3, alpha=0.7)
ax2.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
ax2.set_ylabel(r"Log-loss $\ell_t$", fontsize=11)
ax2.tick_params(axis="y", labelsize=10)

ax2.axvline(split_date, color="black", linewidth=0.8, linestyle="--", alpha=0.6)


# ── Format the figure ──────────────────────────────────────────────

for ax in (ax1, ax2):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", which="major", labelsize=10)

fig.autofmt_xdate(rotation=0)
plt.tight_layout()


# ── Save the figure ──────────────────────────────────────────────
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "btc_series.png"

plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved {out_path}")