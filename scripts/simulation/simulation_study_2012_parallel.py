"""Monte Carlo simulation study replicating Wang, Li & He (2012)."""

import time
import numpy as np
import pandas as pd

from pathlib import Path
from joblib import Parallel, delayed

from src.simulation.dgp import generate_data_2012
from src.methods.pipelines import run_2stage, run_cs
from src.methods.quantile_regression import fit_qr_multiple


# ── Settings ──────────────────────────────────────────────────────────────────
cases          = [1, 2, 3]
r_values       = [0, 0.9]
tau_n_values   = [0.991, 0.995]
sample_sizes   = [200, 500, 1000]
MC_simulations = 500

tau_U_values = {"CS1": 0.95, "CS2": 0.99}
eta          = 0.1

x_points = {
    "(0,0)":     np.array([0.0, 0.0]),
    "(0.5,0.5)": np.array([0.5, 0.5]),
}



def true_eps_quantile(tau, case):
    """Compute the true conditional quantile for the given simulation case."""

    if case == 1:    # Pareto(0.2), gamma = 0.2
        return (1 - tau) ** (-0.2)
    elif case == 2:  # Pareto(0.5), gamma = 0.5
        return (1 - tau) ** (-0.5)
    elif case == 3:
        return 1 + np.tan(np.pi * tau / 2)
    
    
    
def true_conditional_quantile(x, tau, case, r):
    """Q_Y(tau | x) = x1 + x2 + (1 + r*x1) * Q_eps(tau)"""
    return x[0] + x[1] + (1 + r * x[0]) * true_eps_quantile(tau, case)



def run_one_rep(n, case, r, tau_n_values, x_points, k, tau_U_values, eta):
    """Run one Monte Carlo replication for a given simulation case."""
   
    data  = generate_data_2012(n, r, case)
    y, X = data
    
    out   = {}

    # ── CS1 and CS2 ──────────────────────────────────────────────────────────
    
    for cs_label, tau_U in tau_U_values.items():
        for tau_n in tau_n_values:
            for x_label, x in x_points.items():
                q_hat, _ = run_cs(tau_n, k, tau_U, data, x) 
                q_true = true_conditional_quantile(x, tau_n, case, r)
                
                out[(cs_label, tau_n, x_label)] = (q_hat - q_true) ** 2
                
    # ── NCS ────────────────────────────────────────────────────────────────
    
    for tau_n in tau_n_values:
        for x_label, x in x_points.items():
            q_hat, _ = run_2stage(
                data,
                tau_n,
                k,
                n,
                eta,
                x_eval=x
            )
            
            q_true = true_conditional_quantile(x, tau_n, case, r)
            
            out[('NCS', tau_n, x_label)] = (q_hat - q_true) ** 2
            
    # ── QR ───────────────────────────────────────────────────────────────────
    
    n_obs = len(y)
    X_fit = np.column_stack([np.ones(n_obs), X])
    
    for tau_n in tau_n_values:
        beta = fit_qr_multiple(y, X_fit, [tau_n])[0]
        
        for x_label, x in x_points.items():
            q_hat  = float(np.concatenate([[1], x]) @ beta)
            q_true = true_conditional_quantile(x, tau_n, case, r)
            
            out[('QR', tau_n, x_label)] = (q_hat - q_true) ** 2

    return out


#--------------------------------------
# --- Simulation loop ---
#--------------------------------------

if __name__ == '__main__':
    
    start_time = time.perf_counter()
    records = []
    
    total = (len(cases) * len(sample_sizes) * len(r_values)
             * len(tau_n_values) * len(x_points) * (len(tau_U_values) + 2))  # CS1, CS2, NCS, QR
    done  = 0

    for case in cases:
        for n in sample_sizes:
            k = int(round(4.5 * n ** (1/3)))
            
            for r in r_values:
                results_list = Parallel(n_jobs=-1)(  
                    delayed(run_one_rep)(n, case, r, tau_n_values, x_points, k, tau_U_values, eta)
                    
                    for _ in range(MC_simulations))

                for method in (*tau_U_values, 'NCS', 'QR'):
                    for tau_n in tau_n_values:
                        for x_label in x_points:
                            sq    = np.array([rep[(method, tau_n, x_label)] for rep in results_list])
                            valid = ~np.isnan(sq)
                            mse = sq[valid].mean()
                            se    = sq[valid].std() / np.sqrt(valid.sum())

                            records.append({
                                "Method": method,
                                "Case":   case,
                                "n":      n,
                                "r":      r,
                                "tau_n":  tau_n,
                                "x":      x_label,
                                "MSE":    mse ,
                                "SE":     se  ,
                                "Valid":  valid.sum(),
                            })
                            
                            scale = {1: 1e2, 2: 1e0, 3: 1e-2}[case]
                            done += 1
                            
                            print(f"[{done}/{total}] {method:3s} Case {case}, n={n}, r={r}, "
                                  f"tau={tau_n}, x={x_label}  "
                                  f"MSE={mse*scale:.2f}  ({valid.sum()}/{MC_simulations})")

    results = pd.DataFrame(records)

    tables_dir = Path(__file__).resolve().parents[2] / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "simulation_2012.csv"
    results.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(results.to_string(index=False))

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
