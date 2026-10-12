import numpy as np

from joblib import Parallel, delayed

from src.methods.back_test import quantile_loss_vec
from src.methods.benchmark_models import GARCH11_EVT
from src.methods.pipelines import run_3stage, run_2stage



def _one_step(t, log_losses, X, tau_n, model, eta, WINDOW):
    """Compute one-step-ahead VaR forecast."""
    
    log_losses_window = log_losses[t - WINDOW + 1: t + 1]
    X_window          = X[t - WINDOW + 1: t + 1]

    if model == "GARCH11_EVT":
        
        return GARCH11_EVT(log_losses_window, tau_n)

    elif model == "3stage":
        shift       = max(0, -log_losses_window.min()) + 0.01
        data_window = (log_losses_window + shift, X_window)
        x_pred      = X[t + 1]
        VaR, _, _, _ = run_3stage(tau_n, eta, data_window, WINDOW, x_pred)
        
        return VaR - shift

    elif model == "2stage":
        k_2s        = int(round(4.5 * (WINDOW ** (1/3))))
        shift       = max(0, -log_losses_window.min()) + 0.01
        data_window = (log_losses_window + shift, X_window)
        x_pred      = X[t + 1]
        VaR, _      = run_2stage(data_window, tau_n, k_2s, WINDOW, eta, x_pred)
        
        return VaR - shift



def _one_step_safe(t, log_losses, X, tau_n, model, eta, WINDOW):
    """Compute one-step-ahead VaR forecast and return NaN if it fails."""
    
    try:
        return _one_step(t, log_losses, X, tau_n, model, eta, WINDOW)
    except Exception:
        return np.nan



def rolling_window_back_test(log_losses, X, tau_n, model, WINDOW):
    """Run rolling window backtest and return scores, losses, violations, and forecasts."""

    eta = 0.1

    forecasts = np.array(
        Parallel(n_jobs=-1)(
            delayed(_one_step_safe)(t, log_losses, X, tau_n, model, eta, WINDOW)
            for t in range(WINDOW - 1, len(log_losses) - 1)
        ),
        dtype=float
    )

    actuals    = log_losses[WINDOW:]
    scores     = quantile_loss_vec(tau_n, actuals, forecasts)
    violations = np.where(np.isnan(forecasts), np.nan,
                          (actuals > forecasts).astype(float))

    valid = int(np.sum(~np.isnan(scores)))
    total = len(scores)
    print(f"  Valid windows: {valid}/{total}")

    return scores, np.nanmean(scores), violations, forecasts