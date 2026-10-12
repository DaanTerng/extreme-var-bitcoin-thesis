"""Run the Python estimators on shared datasets for comparison with R."""

import sys
import time

import numpy as np
import pandas as pd

from pathlib import Path


# ── File paths ──────────────────────────────────────────────

project_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_dir))

from src.methods.pipelines import run_2stage, run_3stage

data_dir = project_dir / "data" / "raw"
out_dir = project_dir / "outputs" / "implementation_checks"
out_dir.mkdir(parents=True, exist_ok=True)


# ── Settings ───────────────────────────────────────────────

cases = range(1, 7)
sample_size = 2000       # Use 500 for a quicker check.
mc_id = 0
eta = 0.1
tau_n = 0.99


# ── Run the shared-data checks ──────────────────────────────

for case in cases:
    start_time = time.perf_counter()

    data_path = data_dir / f"mc_samples_case{case}_n{sample_size}.csv"

    print(f"\n{'=' * 60}")
    print(f"CASE {case}: loading {data_path.name}", flush=True)

    # Load one replication and preserve observation order.
    df = pd.read_csv(data_path)

    if "mc" in df.columns:
        df = df.loc[df["mc"] == mc_id].copy()

    if "obs" in df.columns:
        df = df.sort_values("obs")

    if len(df) != sample_size:
        raise ValueError(
            f"Case {case}: expected {sample_size} rows, got {len(df)}."
        )

    covariates = [c for c in ["x1", "x2"] if c in df.columns]

    if not covariates:
        raise ValueError(f"Case {case}: no covariates found.")

    y = df["y"].to_numpy(dtype=float)
    X = df[covariates].to_numpy(dtype=float)
    n = len(y)

    obs = (
        df["obs"].to_numpy()
        if "obs" in df.columns
        else np.arange(n)
    )

    print(f"mc={mc_id}, n={n}, covariates={covariates}", flush=True)


    # ── Fit the estimators ──────────────────────────────────

    print("Running 3Stage...", flush=True)

    q3, gamma3, lamb, k3 = run_3stage(
        tau_n=tau_n,
        eta=eta,
        data=(y, X),
        n=n,
    )

    print("Running 2Stage...", flush=True)

    k2 = int(round(4.5 * n ** (1 / 3)))

    q2, gamma2 = run_2stage(
        data=(y, X),
        tau_n=tau_n,
        k_2s=k2,
        n=n,
        eta=eta,
    )

    q3 = np.asarray(q3).reshape(-1)
    gamma3 = np.asarray(gamma3).reshape(-1)
    q2 = np.asarray(q2).reshape(-1)
    gamma2 = np.asarray(gamma2).reshape(-1)


    # ── Recover the intermediate quantiles ──────────────────

    # Derived from the predictions, rather than independently estimated.
    tau03 = (n - k3) / (n + 1)
    tau02 = (n - k2) / (n + 1)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        base3 = q3 / (((1 - tau03) / (1 - tau_n)) ** gamma3)
        base2 = q2 / (((1 - tau02) / (1 - tau_n)) ** gamma2)


    # ── Save results ────────────────────────────────────────

    results_df = pd.DataFrame({
        "case": case,
        "mc": mc_id,
        "obs": obs,
        "lambda_3stage": lamb,
        "k_3stage": k3,
        "k_2stage": k2,
        "gamma_3stage": gamma3,
        "q_base_3stage": base3,
        "q_hat_3stage": q3,
        "gamma_2stage": gamma2,
        "q_base_2stage": base2,
        "q_hat_2stage": q2,
    })

    out_path = out_dir / f"python_case{case}_n{n}_mc{mc_id}.csv"
    results_df.to_csv(out_path, index=False)


    # ── Print summaries and all observations ────────────────

    print(f"\n3Stage: lambda={lamb}, k={k3}")
    print(f"2Stage: k={k2}")

    for name, predictions, gamma in [
        ("3Stage", q3, gamma3),
        ("2Stage", q2, gamma2),
    ]:
        finite_gamma = gamma[np.isfinite(gamma)]
        mean_gamma = (
            finite_gamma.mean() if finite_gamma.size else np.nan
        )
        valid_count = np.isfinite(predictions).sum()

        print(
            f"{name}: mean gamma={mean_gamma:.8f}, "
            f"finite predictions={valid_count}/{n}"
        )

    print(results_df.to_string(
        index=False,
        max_rows=None,
        max_cols=None,
        float_format=lambda value: f"{value:.10g}",
    ))

    elapsed = time.perf_counter() - start_time

    print(f"\nSaved: {out_path}")
    print(f"Total time: {elapsed:.1f}s", flush=True)