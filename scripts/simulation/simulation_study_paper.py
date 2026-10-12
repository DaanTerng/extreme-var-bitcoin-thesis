"""Monte Carlo simulation study in the thesis."""

import time
import numpy as np
import pandas as pd

from pathlib import Path
from joblib import Parallel, delayed

from src.simulation.dgp import generate_data_simu
from src.methods.power_transformation import box_cox_inv
from src.methods.quantile_regression import fit_qr_multiple
from src.methods.pipelines import run_3stage, run_2stage



def true_conditional_quantile(tau, case, X):
    """Compute the true conditional quantile for the given simulation case."""

    n = X.shape[0]
    x_full = np.column_stack([np.ones(n), X])

    if case == 1:
        lam, beta, sigma = 1, np.array([2, 1, 1]), np.array([1, 0.5, 0])
        Q_eps = (1 - tau) ** (-0.5)

    elif case == 2:
        lam, beta, sigma = 0.5, np.array([2, 1, 1]), np.array([1, 0.5, 0])
        Q_eps = (1 - tau) ** (-0.25)

    elif case == 3:
        beta = np.array([2, 1, 1])
        sigma = np.array([1, 0.5, 0])

        Q_eps = tau - 1 - np.log(1 - tau)
        linear = x_full @ beta + x_full @ sigma + 0.5 * Q_eps

        return box_cox_inv(0, linear)
    
    elif case == 4:
        gamma_x = x_full @ np.array([0.5, 0.25, 0])
        Q_eps   = (1 - tau) ** (-gamma_x)
        linear  = x_full @ np.array([2, 1, 1]) + Q_eps
        
        return box_cox_inv(0.5, linear)

    elif case == 5:
        lam, beta, sigma = 0, np.array([2, 1, 1]), np.array([0.5, 0.25, 0])
        Q_eps = tau - 1 - np.log(1 - tau)

    elif case == 6:
        x1 = X[:, 0]
        gamma_x = np.exp(x1 - 1)
        
        return (-np.log(tau)) ** (-gamma_x)

    elif case == 7:
        x1 = X[:, 0]
        gamma_x = 0.5 + 0.3 * np.sin(np.pi * x1)
        
        return (-np.log(tau)) ** (-gamma_x)

    linear = x_full @ beta + (x_full @ sigma) * Q_eps
    
    return box_cox_inv(lam, linear)


def run_one_mc(mc, seed, case, n, tau_ns, eta, k_2s, true_lamb=None):
    """Run one Monte Carlo replication for a given simulation case."""

    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    data_seed, seed_3s, seed_2s = ss.spawn(3)

    y, X = generate_data_simu(n, case, data_seed)
    q_trues = [true_conditional_quantile(tau_n, case, X) for tau_n in tau_ns]

    def _empty():
        
        return {
            'bias_3s':    np.full(n, np.nan), 'sq_3s':    np.full(n, np.nan), 'sq_3s_n':    np.full(n, np.nan),
            'gamma_3s':   np.full(n, np.nan), 'lamb': np.nan, 'k': np.nan,
            'bias_3s_tl': np.full(n, np.nan), 'sq_3s_tl': np.full(n, np.nan), 'sq_3s_tl_n': np.full(n, np.nan),
            'gamma_3s_tl':np.full(n, np.nan), 'k_tl': np.nan,
            'bias_ncs':   np.full(n, np.nan), 'sq_ncs':   np.full(n, np.nan), 'sq_ncs_n':   np.full(n, np.nan),
            'gamma_ncs':  np.full(n, np.nan),
            'bias_qr':    np.full(n, np.nan), 'sq_qr':    np.full(n, np.nan), 'sq_qr_n':    np.full(n, np.nan),
        }

    out_list = [_empty() for _ in tau_ns]
    X_fit    = np.column_stack([np.ones(n), X])
    tau_ref  = tau_ns[0]
    lamb     = np.nan

    # ── 3stage: estimate once at tau_ref, scale Weissman to all tau_ns ───────
    try:
        q_hat_0, gamma_hat, lamb, k = run_3stage(tau_ref, eta, (y, X), n, seed=seed_3s)
       
        for i, tau_n in enumerate(tau_ns):
            q_hat  = q_hat_0 * ((1 - tau_ref) / (1 - tau_n)) ** gamma_hat
            q_true = q_trues[i]
            
            out_list[i]['bias_3s']  = q_hat - q_true
            out_list[i]['sq_3s']    = (q_hat - q_true) ** 2
            out_list[i]['sq_3s_n']  = ((q_hat - q_true) / q_true) ** 2
            out_list[i]['gamma_3s'] = gamma_hat
            out_list[i]['lamb']     = lamb
            out_list[i]['k']        = k
    
    except Exception:
        pass

    # ── 3stage (true lambda): estimate once, scale to all tau_ns ─────────────
    
    if true_lamb is not None:
        try:
            q_hat_tl_0, gamma_hat_tl, _, k_tl = run_3stage(tau_ref, eta, (y, X), n, lamb=true_lamb)
            
            for i, tau_n in enumerate(tau_ns):
                q_hat_tl = q_hat_tl_0 * ((1 - tau_ref) / (1 - tau_n)) ** gamma_hat_tl
                q_true   = q_trues[i]
                
                out_list[i]['bias_3s_tl']  = q_hat_tl - q_true
                out_list[i]['sq_3s_tl']    = (q_hat_tl - q_true) ** 2
                out_list[i]['sq_3s_tl_n']  = ((q_hat_tl - q_true) / q_true) ** 2
                out_list[i]['gamma_3s_tl'] = gamma_hat_tl
                out_list[i]['k_tl']        = k_tl
       
        except Exception:
            pass

    # ── QR (direct): fit at all tau_ns in one call ───────────────────────────
    
    try:
        betas_qr = fit_qr_multiple(y, X_fit, tau_ns)
        
        for i, tau_n in enumerate(tau_ns):
            q_hat_qr = X_fit @ betas_qr[i]
            q_true   = q_trues[i]
            
            out_list[i]['bias_qr'] = q_hat_qr - q_true
            out_list[i]['sq_qr']   = (q_hat_qr - q_true) ** 2
            out_list[i]['sq_qr_n'] = ((q_hat_qr - q_true) / q_true) ** 2
    
    except Exception:
        pass

    # ── 2stage: estimate once at tau_ref, scale to all tau_ns ────────────────
   
    try:
        q_hat_2s_0, gamma_2s = run_2stage((y, X), tau_ref, k_2s, n, eta, seed=seed_2s)
        
        for i, tau_n in enumerate(tau_ns):
            q_hat_2s = q_hat_2s_0 * ((1 - tau_ref) / (1 - tau_n)) ** gamma_2s
            q_true   = q_trues[i]
            
            out_list[i]['bias_ncs']  = q_hat_2s - q_true
            out_list[i]['sq_ncs']    = (q_hat_2s - q_true) ** 2
            out_list[i]['sq_ncs_n']  = ((q_hat_2s - q_true) / q_true) ** 2
            out_list[i]['gamma_ncs'] = gamma_2s
    
    except Exception:
        pass

    return out_list


#--------------------------------------
# --- Simulation loop ---
#--------------------------------------

if __name__ == '__main__':

    TRUE_LAMBDA = {1: 1.0, 2: 0.5, 3: 0.0, 4: 0.5, 5: 0.0}
    MC           = 500
    eta          = 0.1
    tau_n_list   = [0.99, 0.995, 0.999]
    cases        = [1, 2, 3, 4, 5, 6, 7]
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
                delayed(run_one_mc)(mc, child_seeds[mc], case, n, tau_n_list, eta, k_2s, true_lamb)
                
                for mc in range(MC)
            )

            elapsed = time.perf_counter() - t_case
            print(f"\n  Case {case}, n={n} done in {elapsed:.1f}s")

            # ── Collect Simulation results ──────────────────────────────────────────────

            for i, tau_n in enumerate(tau_n_list):
                res_i = [r[i] for r in results]

                bias_3s     = np.array([r['bias_3s']     for r in res_i])
                sq_3s       = np.array([r['sq_3s']       for r in res_i])
                sq_3s_n     = np.array([r['sq_3s_n']     for r in res_i])
                gamma_3s    = np.array([r['gamma_3s']    for r in res_i])
                lamb_vec    = np.array([r['lamb']         for r in res_i])
                k_vec       = np.array([r['k']            for r in res_i])
                bias_3s_tl  = np.array([r['bias_3s_tl']  for r in res_i])
                sq_3s_tl    = np.array([r['sq_3s_tl']    for r in res_i])
                sq_3s_tl_n  = np.array([r['sq_3s_tl_n']  for r in res_i])
                gamma_3s_tl = np.array([r['gamma_3s_tl'] for r in res_i])
                k_vec_tl    = np.array([r['k_tl']         for r in res_i])
                bias_ncs    = np.array([r['bias_ncs']    for r in res_i])
                sq_ncs      = np.array([r['sq_ncs']      for r in res_i])
                sq_ncs_n    = np.array([r['sq_ncs_n']    for r in res_i])
                gamma_ncs   = np.array([r['gamma_ncs']   for r in res_i])
                bias_qr     = np.array([r['bias_qr']     for r in res_i])
                sq_qr       = np.array([r['sq_qr']       for r in res_i])
                sq_qr_n     = np.array([r['sq_qr_n']     for r in res_i])

                # ── Print simulation results ──────────────────────────────────────────────

                print(f"\n  tau_n = {tau_n:.4f}")

                print("\n  --- 3stage ---")
                print(f"  mean_lambda : {np.nanmean(lamb_vec):.4f}")
                print(f"  mean_k      : {np.nanmean(k_vec):.2f}")
                print(f"  mean_gamma  : {np.nanmean(gamma_3s):.4f}")
                print(f"  IBias       : {np.nanmean(np.nanmean(bias_3s,   axis=1)):.4f}")
                print(f"  RIMSE       : {np.sqrt(np.nanmean(np.nanmean(sq_3s,   axis=1))):.4f}")
                # RIMSE_n is documented as RMSPE in the report
                print(f"  RIMSE_n     : {np.sqrt(np.nanmean(np.nanmean(sq_3s_n, axis=1))):.4f}")
                print(f"  Valid_obs   : {int(np.sum(~np.isnan(bias_3s)))} / {MC * n}")

                if true_lamb is not None:
                    print(f"\n  --- 3stage (true lambda={true_lamb}) ---")
                    print(f"  mean_k      : {np.nanmean(k_vec_tl):.2f}")
                    print(f"  mean_gamma  : {np.nanmean(gamma_3s_tl):.4f}")
                    print(f"  IBias       : {np.nanmean(np.nanmean(bias_3s_tl,   axis=1)):.4f}")
                    print(f"  RIMSE       : {np.sqrt(np.nanmean(np.nanmean(sq_3s_tl,   axis=1))):.4f}")
                    # RIMSE_n is documented as RMSPE in the report
                    print(f"  RIMSE_n     : {np.sqrt(np.nanmean(np.nanmean(sq_3s_tl_n, axis=1))):.4f}")
                    print(f"  Valid_obs   : {int(np.sum(~np.isnan(bias_3s_tl)))} / {MC * n}")

                print("\n  --- QR (direct) ---")
                print(f"  IBias       : {np.nanmean(np.nanmean(bias_qr,   axis=1)):.4f}")
                print(f"  RIMSE       : {np.sqrt(np.nanmean(np.nanmean(sq_qr,   axis=1))):.4f}")
                print(f"  RIMSE_n     : {np.sqrt(np.nanmean(np.nanmean(sq_qr_n, axis=1))):.4f}")
                print(f"  Valid_obs   : {int(np.sum(~np.isnan(bias_qr)))} / {MC * n}")

                print("\n  --- 2stage ---")
                print(f"  k_2s        : {k_2s}")
                print(f"  mean_gamma  : {np.nanmean(gamma_ncs):.4f}")
                print(f"  IBias       : {np.nanmean(np.nanmean(bias_ncs,   axis=1)):.4f}")
                print(f"  RIMSE       : {np.sqrt(np.nanmean(np.nanmean(sq_ncs,   axis=1))):.4f}")
                # RIMSE_n is documented as RMSPE in the report
                print(f"  RIMSE_n     : {np.sqrt(np.nanmean(np.nanmean(sq_ncs_n, axis=1))):.4f}")
                print(f"  Valid_obs   : {int(np.sum(~np.isnan(bias_ncs)))} / {MC * n}")

                nan_mc = np.full((MC, n), np.nan)
                
                # ── Store results ──────────────────────────────────────────────

                for method, bias, sq, sq_n, gamma in [
                    ('3stage',    bias_3s,    sq_3s,    sq_3s_n,    gamma_3s),
                    ('3stage_tl', bias_3s_tl, sq_3s_tl, sq_3s_tl_n, gamma_3s_tl),
                    ('qr',        bias_qr,    sq_qr,    sq_qr_n,    nan_mc),
                    ('2stage',    bias_ncs,   sq_ncs,   sq_ncs_n,   gamma_ncs),
                ]:
                    records.append({
                        'Method':      method,
                        'Case':        case,
                        'n':           n,
                        'tau_n':       tau_n,
                        'IBias':       np.nanmean(np.nanmean(bias,  axis=1)),
                        'RIMSE':       np.sqrt(np.nanmean(np.nanmean(sq,   axis=1))),
                        # RIMSE_n is documented as RMSPE in the report
                        'RIMSE_n':     np.sqrt(np.nanmean(np.nanmean(sq_n, axis=1))),
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
    out_path = tables_dir / "simulation_thesis.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(results_df.to_string(index=False))
    print(f"\nTotal time: {time.perf_counter() - t_total:.1f}s")
