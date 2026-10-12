"""Monte Carlo simulation study replicating Wang & Li (2013)."""

import time
import numpy as np
import pandas as pd

from pathlib import Path
from joblib import Parallel, delayed

from src.simulation.dgp import generate_data_2013
from src.methods.power_transformation import box_cox_inv, box_cox, estimate_lambda
from src.methods.quantile_regression import fit_qr_multiple
from src.methods.pipelines import run_3stage, run_2stage, run_qr


def true_conditional_quantile(tau, case, X):
    """Compute the true conditional quantile for the given simulation case."""

    if case == 5:
        x1 = X[:, 0]
        gamma_x = np.exp(-1 + x1)
        
        return (1 - tau) ** (-gamma_x) - 1

    elif case == 6:
        x1 = X[:, 0]
        gamma_x = 0.5 * (1/10 + np.sin(np.pi * (x1 + 1) / 2)) * (11/10 - 0.5 * np.exp(-16 * x1**2))
        
        return (-np.log(tau)) ** (-gamma_x)

    n = X.shape[0]
    x_full = np.column_stack([np.ones(n), X])

    if case == 1:
        lam, beta, sigma = 1,   np.array([2, 2, 2]),    np.array([2, 1.6, 0])
        Q_eps = (1 - tau) ** (-1)
        
    elif case == 2:
        lam, beta, sigma = 0.5, np.array([2, 2, 2]),    np.array([2, 1.6, 0])
        Q_eps = (1 - tau) ** (-0.5)
        
    elif case == 3:
        lam, beta, sigma = -1,  np.array([-2, -1, -1]), np.array([3, 1, 1])
        Q_eps = tau
    
    elif case == 4:
        lam, beta, sigma = 0,   np.array([2, 1, 1]),    np.array([0.5, 0.25, 0])
        Q_eps = tau - 1 - np.log(1 - tau)

    linear = x_full @ beta + (x_full @ sigma) * Q_eps
    
    return box_cox_inv(lam, linear)


def run_one_mc(mc, seed, case, n, tau_n, eta, k_2s, true_lamb=None):
    """Run one Monte Carlo replication for a given simulation case."""

    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    data_seed, seed_3s, seed_2s = ss.spawn(3)

    y, X = generate_data_2013(n, case, data_seed)
    q_true = true_conditional_quantile(tau_n, case, X)

    out = {
        'bias_3s':    np.full(n, np.nan), 'sq_3s':    np.full(n, np.nan),
        'gamma_3s':   np.full(n, np.nan), 'lamb': np.nan, 'k': np.nan,
        'bias_3s_tl': np.full(n, np.nan), 'sq_3s_tl': np.full(n, np.nan),
        'gamma_3s_tl': np.full(n, np.nan), 'k_tl': np.nan,
        'bias_ncs':   np.full(n, np.nan), 'sq_ncs':   np.full(n, np.nan),
        'gamma_ncs':  np.full(n, np.nan),
        'bias_qr':    np.full(n, np.nan), 'sq_qr':    np.full(n, np.nan),
        'bias_qrt':   np.full(n, np.nan), 'sq_qrt':   np.full(n, np.nan),
    }

    # ── 3stage ──────────────────────────────────────────────
    
    try:
        q_hat, gamma_hat, lamb, k = run_3stage(tau_n, eta, (y, X), n, seed=seed_3s)
        
        out['bias_3s']  = q_hat - q_true
        out['sq_3s']    = (q_hat - q_true)**2
        out['gamma_3s'] = gamma_hat
        out['lamb']     = lamb
        out['k']        = k
    
    except Exception:
        pass

    # ── 3stage (true lambda) ─────────────────────────────────
    
    if true_lamb is not None:
        try:
            q_hat_tl, gamma_hat_tl, _, k_tl = run_3stage(tau_n, eta, (y, X), n, lamb=true_lamb)
            
            out['bias_3s_tl']   = q_hat_tl - q_true
            out['sq_3s_tl']     = (q_hat_tl - q_true)**2
            out['gamma_3s_tl']  = gamma_hat_tl
            out['k_tl']         = k_tl
        
        except Exception:
            pass

    # ── QR (direct) ─────────────────────────────────────────
    
    try:
        q_hat_qr = run_qr(tau_n, (y, X))
        
        out['bias_qr'] = q_hat_qr - q_true
        out['sq_qr']   = (q_hat_qr - q_true)**2
    
    except Exception:
        pass

    # ── QR (via transformation) ──────────────────────────────
    
    try:
        a_qrt     = max(0.0, float(-y.min()) + 1e-6)
        lambdas   = np.round(np.arange(-2.0, 2.1, 0.1), 1)
        eps_s1    = 0.15 if n <= 600 else 0.10
        lamb_qrt  = estimate_lambda(y, X, np.array([1 - eps_s1]), lambdas, a=a_qrt)
        
        y_tr      = box_cox(lamb_qrt, y + a_qrt)
        X_fit     = np.column_stack([np.ones(n), X])
        
        beta_qrt  = fit_qr_multiple(y_tr, X_fit, [tau_n])[0]
        q_hat_qrt = box_cox_inv(lamb_qrt, X_fit @ beta_qrt) - a_qrt
        
        out['bias_qrt'] = q_hat_qrt - q_true
        out['sq_qrt']   = (q_hat_qrt - q_true)**2
    
    except Exception:
        pass

    # ── 2stage ──────────────────────────────────────────────
    
    try:
        q_hat_2s, gamma_2s = run_2stage((y, X), tau_n, k_2s, n, eta, seed=seed_2s)
        
        out['bias_ncs']  = q_hat_2s - q_true
        out['sq_ncs']    = (q_hat_2s - q_true)**2
        out['gamma_ncs'] = gamma_2s
        
    except Exception:
        pass

    return out


#--------------------------------------
# --- Simulation loop ---
#--------------------------------------

if __name__ == '__main__':

    TRUE_LAMBDA = {1: 1.0, 2: 0.5, 3: -1.0}

    MC           = 500
    eta          = 0.1
    tau_n        = 0.99 # Target quantile level. Change and rerun for other levels.
    cases        = [1, 2, 3, 4, 5, 6]
    sample_sizes = [500, 2000]

    records = []
    t_total = time.perf_counter()

    for n in sample_sizes:
        k_2s = int(round(4.5 * n**(1/3)))

        ss = np.random.SeedSequence(22)
        child_seeds = ss.spawn(MC)

        for case in cases:
            true_lamb = TRUE_LAMBDA.get(case, None)

            print(f"\n{'='*50}")
            print(f"  CASE {case}, n={n}")
            print(f"{'='*50}")
            
            t_case = time.perf_counter()

            results = Parallel(n_jobs=-1)(
                delayed(run_one_mc)(mc, child_seeds[mc], case, n, tau_n, eta, k_2s, true_lamb)
                for mc in range(MC)
            )


            # ── Collect Simulation results ──────────────────────────────────────────────

            bias_3s    = np.array([r['bias_3s']    for r in results])
            sq_3s      = np.array([r['sq_3s']      for r in results])
            gamma_3s   = np.array([r['gamma_3s']   for r in results])
            lamb_vec   = np.array([r['lamb']        for r in results])
            k_vec      = np.array([r['k']           for r in results])
            bias_3s_tl = np.array([r['bias_3s_tl'] for r in results])
            sq_3s_tl   = np.array([r['sq_3s_tl']   for r in results])
            gamma_3s_tl = np.array([r['gamma_3s_tl'] for r in results])
            k_vec_tl   = np.array([r['k_tl']        for r in results])
            bias_ncs   = np.array([r['bias_ncs']   for r in results])
            sq_ncs     = np.array([r['sq_ncs']     for r in results])
            gamma_ncs  = np.array([r['gamma_ncs']  for r in results])
            bias_qr    = np.array([r['bias_qr']    for r in results])
            sq_qr      = np.array([r['sq_qr']      for r in results])
            bias_qrt   = np.array([r['bias_qrt']   for r in results])
            sq_qrt     = np.array([r['sq_qrt']     for r in results])

            elapsed = time.perf_counter() - t_case

            # ── Print simulation results ──────────────────────────────────────────────

            print(f"\n  Case {case}, n={n} done in {elapsed:.1f}s")

            print("\n  --- 3stage ---")
            print(f"  mean_lambda : {np.nanmean(lamb_vec):.4f}")
            print(f"  mean_k      : {np.nanmean(k_vec):.2f}")
            print(f"  mean_gamma  : {np.nanmean(gamma_3s):.4f}")
            print(f"  IBias       : {np.nanmean(np.nanmean(bias_3s,  axis=1)):.4f}")
            print(f"  RIMSE       : {np.sqrt(np.nanmean(np.nanmean(sq_3s,  axis=1))):.4f}")
            print(f"  Valid_obs   : {int(np.sum(~np.isnan(bias_3s)))} / {MC * n}")

            if true_lamb is not None:
                print(f"\n  --- 3stage (true lambda={true_lamb}) ---")
                print(f"  mean_k      : {np.nanmean(k_vec_tl):.2f}")
                print(f"  mean_gamma  : {np.nanmean(gamma_3s_tl):.4f}")
                print(f"  IBias       : {np.nanmean(np.nanmean(bias_3s_tl, axis=1)):.4f}")
                print(f"  RIMSE       : {np.sqrt(np.nanmean(np.nanmean(sq_3s_tl, axis=1))):.4f}")
                print(f"  Valid_obs   : {int(np.sum(~np.isnan(bias_3s_tl)))} / {MC * n}")

            print("\n  --- QR (direct) ---")
            print(f"  IBias       : {np.nanmean(np.nanmean(bias_qr, axis=1)):.4f}")
            print(f"  RIMSE       : {np.sqrt(np.nanmean(np.nanmean(sq_qr, axis=1))):.4f}")
            print(f"  Valid_obs   : {int(np.sum(~np.isnan(bias_qr)))} / {MC * n}")

            print("\n  --- QR (transf) ---")
            print(f"  IBias       : {np.nanmean(np.nanmean(bias_qrt, axis=1)):.4f}")
            print(f"  RIMSE       : {np.sqrt(np.nanmean(np.nanmean(sq_qrt, axis=1))):.4f}")
            print(f"  Valid_obs   : {int(np.sum(~np.isnan(bias_qrt)))} / {MC * n}")

            print("\n  --- 2stage ---")
            print(f"  k_2s        : {k_2s}")
            print(f"  mean_gamma  : {np.nanmean(gamma_ncs):.4f}")
            print(f"  IBias       : {np.nanmean(np.nanmean(bias_ncs, axis=1)):.4f}")
            print(f"  RIMSE       : {np.sqrt(np.nanmean(np.nanmean(sq_ncs, axis=1))):.4f}")
            print(f"  Valid_obs   : {int(np.sum(~np.isnan(bias_ncs)))} / {MC * n}")

            # ── Store results ──────────────────────────────────────────────

            for method, bias, sq, gamma in [
                ('3stage',       bias_3s,    sq_3s,    gamma_3s),
                ('3stage_tl',    bias_3s_tl, sq_3s_tl, gamma_3s_tl),
                ('qr',           bias_qr,    sq_qr,    np.full((MC, n), np.nan)),
                ('qr_transf',    bias_qrt,   sq_qrt,   np.full((MC, n), np.nan)),
                ('2stage',       bias_ncs,   sq_ncs,   gamma_ncs),
            ]:
                records.append({
                    'Method':      method,
                    'Case':        case,
                    'n':           n,
                    'tau_n':       tau_n,
                    'IBias':       np.nanmean(np.nanmean(bias, axis=1)),
                    'RIMSE':       np.sqrt(np.nanmean(np.nanmean(sq, axis=1))),
                    'gamma_hat':   np.nanmean(gamma),
                    'k_2stage':    k_2s                  if method == '2stage'    else np.nan,
                    'mean_k':      np.nanmean(k_vec)      if method == '3stage'    else
                                   np.nanmean(k_vec_tl)  if method == '3stage_tl' else np.nan,
                    'mean_lambda': np.nanmean(lamb_vec)   if method == '3stage'    else
                                   true_lamb              if method == '3stage_tl' else np.nan,
                    'Valid_obs':   int(np.sum(~np.isnan(bias))),
                })

    # ── Save results ──────────────────────────────────────────────

    results_df = pd.DataFrame(records)

    tables_dir = Path(__file__).resolve().parents[2] / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "simulation_2013.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(results_df.to_string(index=False))
    print(f"\nTotal time: {time.perf_counter() - t_total:.1f}s")
