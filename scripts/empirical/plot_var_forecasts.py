"""Plot the empirical VaR forecasts and violation dates."""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.path as mpath

from pathlib import Path


# ── File paths ──────────────────────────────────────────────

data_path      = Path(__file__).resolve().parents[2] / "data" / "btc_data.csv"
forecasts_path = Path(__file__).resolve().parents[2] / "outputs" / "tables" / "var_forecasts.csv"
out_dir        = Path(__file__).resolve().parents[2] / "outputs" / "figures"


# ── Load data ──────────────────────────────────────────────

btc       = pd.read_csv(data_path, index_col=0, parse_dates=True)
forecasts = pd.read_csv(forecasts_path, parse_dates=["date"])

WINDOW = 500
btc = btc["2019-01-01":"2026-06-01"]

actuals_dates  = btc.index[WINDOW:]
actuals_values = btc["log_return"].values[WINDOW:]


# ── Plot settings ──────────────────────────────────────────────

colors = {"GARCH11_EVT": "#A03D47", "3stage": "#5E8745"}
bright_colors = {"GARCH11_EVT": "#FF6B6B", "3stage": "#7FFF00"}
labels = {"GARCH11_EVT": "GARCH(1,1)-EVT", "3stage": "3-Stage"}
models_to_plot = ["GARCH11_EVT", "3stage"]



def half_circle_marker(start_angle, end_angle):
    """Create a filled half-circle marker using angles in degrees."""

    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), 50)
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    verts = np.vstack([[0, 0], verts, [0, 0]])
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(verts) - 2) + [mpath.Path.CLOSEPOLY]
    
    return mpath.Path(verts, codes)



left_half  = half_circle_marker(90, 270)   
right_half = half_circle_marker(-90, 90)  


#--------------------------------------
# --- Plot VaR forecasts ---
#--------------------------------------

for tau in forecasts["tau"].unique():
    df_tau = forecasts[forecasts["tau"] == tau].copy()
    fig, ax = plt.subplots(figsize=(14, 3))

    ax.plot(actuals_dates, actuals_values, color="#5A5A5A", linewidth=0.3,
            alpha=0.7, label="BTC log-return", zorder=1)

    var_lines = {}
    viol_masks = {}
    
    
    # ── Plot model forecasts ──────────────────────────────────────────────

    for model in models_to_plot:
        df_m = df_tau[df_tau["model"] == model].set_index("date")["VaR"].reindex(actuals_dates)
        
        # Convert loss quantiles to return thresholds.
        var_line = -df_m.values
        var_lines[model] = var_line
        
        viol_masks[model] = actuals_values < var_line

        color = colors.get(model, None)
        
        ax.plot(actuals_dates, var_line, color=color, linewidth=0.5,
                label=labels.get(model, model), zorder=3, alpha=0.9)

    # ── Identify violation dates ──────────────────────────────────────────────

    mask_garch = viol_masks["GARCH11_EVT"]
    mask_3stage = viol_masks["3stage"]
    
    shared_mask = mask_garch & mask_3stage
    garch_only  = mask_garch & ~mask_3stage
    stage_only  = mask_3stage & ~mask_garch


    # ── Plot violations ──────────────────────────────────────────────

    # Use full circles for violations unique to one model.
    ax.scatter(actuals_dates[garch_only], actuals_values[garch_only],
               facecolor=colors["GARCH11_EVT"], edgecolor=bright_colors["GARCH11_EVT"],
               linewidth=0.8, s=15, zorder=10, alpha=0.95, marker="o")

    ax.scatter(actuals_dates[stage_only], actuals_values[stage_only],
               facecolor=colors["3stage"], edgecolor=bright_colors["3stage"],
               linewidth=0.8, s=15, zorder=10, alpha=0.95, marker="o")

    # Combine two half-circles for violations shared by both models.
    ax.scatter(actuals_dates[shared_mask], actuals_values[shared_mask],
               facecolor=colors["GARCH11_EVT"], edgecolor=bright_colors["GARCH11_EVT"],
               linewidth=0.8, s=12, zorder=11, alpha=0.95, marker=left_half)

    ax.scatter(actuals_dates[shared_mask], actuals_values[shared_mask],
               facecolor=colors["3stage"], edgecolor=bright_colors["3stage"],
               linewidth=0.8, s=12, zorder=11, alpha=0.95, marker=right_half)

    print("GARCH only:", actuals_dates[garch_only].tolist())
    print("3stage only:", actuals_dates[stage_only].tolist())
    print("Shared:", actuals_dates[shared_mask].tolist())


    # ── Format the figure ──────────────────────────────────────────────

    ax.axhline(0, color="black", linewidth=0.4, linestyle="--", alpha=0.4)
    ax.set_ylabel("Log-return")


    # Set axis limits using the central range of returns and forecasts.

    all_vals = np.concatenate([actuals_values] + list(var_lines.values()))
    lo, hi = np.nanpercentile(all_vals, [0.5, 99.5])
    margin = 0.1 * (hi - lo)
    
    ax.set_ylim(lo - margin, hi + margin)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis='x', which='major', labelsize=9)
    fig.autofmt_xdate(rotation=0)

    leg = ax.legend(fontsize=8)
   
    for line in leg.get_lines():
        line.set_linewidth(2.5)


    # ── Save the figure ──────────────────────────────────────────────

    plt.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"var_forecasts_tau{str(tau).replace('.','')}.png"
    
    plt.savefig(out_path, dpi=150)
    plt.show()
    
    print(f"Saved {out_path}")
    
    
    
    
    
    
    
    
    
    
    
    