"""Generate shared datasets for the Python–R implementation checks."""

import sys

import numpy as np
import pandas as pd

from pathlib import Path

# ── File paths ──────────────────────────────────────────────

project_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_dir))

from src.simulation.dgp import generate_data_2013

out_dir = project_dir / "data" / "raw"
out_dir.mkdir(parents=True, exist_ok=True)


# ── Settings ───────────────────────────────────────────────

MC = 1
cases = [1, 2, 3, 4, 5, 6]
n_vals = [2000]


# ── Generate and save shared datasets ───────────────────────

for case in cases:
    for n in n_vals:
        samples = []

        for mc in range(MC):
            y, X = generate_data_2013(n, case=case, seed=mc)

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            covariates = [f"x{j + 1}" for j in range(X.shape[1])]

            sample_df = pd.DataFrame(X, columns=covariates)
            sample_df.insert(0, "y", y)
            sample_df.insert(0, "obs", np.arange(n))
            sample_df.insert(0, "mc", mc)

            samples.append(sample_df)

        results_df = pd.concat(samples, ignore_index=True)

        out_path = out_dir / f"mc_samples_case{case}_n{n}.csv"
        results_df.to_csv(out_path, index=False)

        print(
            f"Case {case}, n={n}: saved {MC} replication(s), "
            f"{len(results_df)} rows -> {out_path}"
        )