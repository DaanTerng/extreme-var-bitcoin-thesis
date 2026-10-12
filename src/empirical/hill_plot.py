"""This file contains methods for plotting Hill estimator paths."""

import matplotlib.pyplot as plt

from pathlib import Path

from src.methods.hill_estimator import hill_estimator_path

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"



def plot_hill_path(data, k_min=10, k_max=200, gamma_true=None, title="Hill Plot", filename="hill_plot.png"):
    """Plot Hill estimator gamma_hat as a function of k."""
    
    ks, gammas = hill_estimator_path(data, k_min=k_min, k_max=k_max)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, gammas, color="#08519C", linewidth=1.5, label=r"$\hat{\gamma}(k)$")

    if gamma_true is not None:
        ax.axhline(gamma_true, color="red", linestyle="--", linewidth=1,
                   label=f"True $\\gamma$ = {gamma_true}")

    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel(r"$\hat{\gamma}$", fontsize=12)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")
