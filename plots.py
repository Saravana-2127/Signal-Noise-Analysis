"""
Figure builders for the dashboard (kept separate so they can be tested).
All plots use the actual simulated data; theoretical curves come from the
same Gaussian model (core.gaussian_pdf / gaussian_cdf / stress_testing).
"""

import numpy as np
import matplotlib.pyplot as plt

from core import gaussian_pdf, gaussian_cdf
from stress_testing import theoretical_reliability

MAX_VARIANCE = 4.0                 # largest variance offered in the UI
NOISE_XLIM = (-9.0, 9.0)           # FIXED noise axis so spread changes are visible


def _y_limit(amplitude: float) -> float:
    """Fixed y-limit for X/Y plots: amplitude + 3 sigma at the largest variance."""
    return amplitude + 3.0 * np.sqrt(MAX_VARIANCE)


# ---------------------------------------------------------------------------
# Distribution analysis: histogram vs PDF, empirical CDF vs theoretical CDF
# ---------------------------------------------------------------------------
def distribution_figures(noise: np.ndarray, mean: float, variance: float):
    xs = np.linspace(*NOISE_XLIM, 900)
    peak = 1.0 / np.sqrt(2 * np.pi * variance)
    sigma = np.sqrt(variance)

    # PDF
    fig_pdf, ax = plt.subplots(figsize=(5, 4))
    bins = np.linspace(*NOISE_XLIM, 181)                 # fixed bins, width 0.1
    ax.hist(noise, bins=bins, density=True, alpha=0.6, label="Observed noise (density)")
    ax.plot(xs, gaussian_pdf(xs, mean, variance), "r-", lw=2,
            label=f"Theoretical PDF  N({mean:g}, {variance:g})")
    ax.set_xlim(*NOISE_XLIM)
    ax.set_ylim(0, max(0.9, 1.1 * peak))                 # same y-scale for variance >= ~0.2
    ax.set_xlabel("Noise value")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Noise PDF: histogram vs theory (σ² = {variance:g}, σ = {sigma:.2f})", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # CDF (empirical evaluated on the same fixed grid as the theory)
    fig_cdf, ax = plt.subplots(figsize=(5, 4))
    sorted_noise = np.sort(noise)
    ecdf = np.searchsorted(sorted_noise, xs, side="right") / len(sorted_noise)
    tcdf = gaussian_cdf(xs, mean, variance)
    ax.plot(xs, ecdf, lw=2, label="Empirical CDF")
    ax.plot(xs, tcdf, "r--", lw=2, label="Theoretical CDF")
    ax.set_xlim(*NOISE_XLIM)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Noise value")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"Noise CDF: empirical vs theory (σ² = {variance:g})", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    ks_distance = float(np.max(np.abs(ecdf - tcdf)))
    return fig_pdf, fig_cdf, ks_distance


# ---------------------------------------------------------------------------
# Transmitted vs received signal
# ---------------------------------------------------------------------------
def signal_figure(x, y, variance: float, amplitude: float, n_show: int = 100):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(x[:n_show], label="Transmitted (X)", drawstyle="steps-mid", lw=1.8)
    ax.plot(y[:n_show], label="Received (Y = X + N)", alpha=0.75, lw=1.2, marker="o", ms=2.5)
    ax.axhline(0, color="gray", ls=":", lw=1, label="Decision threshold (0)")
    lim = _y_limit(amplitude)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Signal amplitude")
    ax.set_title(f"Transmitted vs received signal, first {n_show} samples "
                 f"(noise variance σ² = {variance:g})")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    return fig


# ---------------------------------------------------------------------------
# Joint distribution of X and Y (three views of the same simulated data)
# ---------------------------------------------------------------------------
def _joint_axes(ax, amplitude, variance, title):
    ax.set_xlim(-amplitude - 0.75, amplitude + 0.75)
    ax.set_ylim(-_y_limit(amplitude), _y_limit(amplitude))
    ax.set_xticks([-amplitude, amplitude])
    ax.set_xticklabels([f"−{amplitude:g}", f"+{amplitude:g}"])
    ax.set_xlabel("Transmitted signal X")
    ax.set_ylabel("Received signal Y")
    ax.set_title(f"{title} (σ² = {variance:g})", fontsize=10)


def joint_scatter_figure(x, y, amplitude, variance, max_points=3000):
    idx = np.random.default_rng(0).choice(len(x), size=min(max_points, len(x)), replace=False)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(x[idx], y[idx], s=10, alpha=0.15, edgecolors="none")
    _joint_axes(ax, amplitude, variance, "Joint distribution of X and Y: scatter")
    return fig


def joint_hexbin_figure(x, y, amplitude, variance):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    lim = _y_limit(amplitude)
    hb = ax.hexbin(x, y, gridsize=(24, 40), mincnt=1, cmap="viridis",
                   extent=(-amplitude - 0.75, amplitude + 0.75, -lim, lim))
    fig.colorbar(hb, ax=ax, label="Sample count")
    _joint_axes(ax, amplitude, variance, "Joint distribution of X and Y: hexbin density")
    return fig


def joint_conditional_figure(x, y, amplitude, noise_mean, variance):
    """Histogram of Y given each X value, with the theoretical N(X+mu, sigma^2) PDF."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ys = np.linspace(*NOISE_XLIM, 600)
    bins = np.linspace(*NOISE_XLIM, 121)
    for val, color in ((-amplitude, "tab:blue"), (amplitude, "tab:orange")):
        ax.hist(y[x == val], bins=bins, density=True, alpha=0.45, color=color,
                label=f"Y | X = {val:+g}")
        ax.plot(ys, gaussian_pdf(ys, val + noise_mean, variance), color=color, lw=2)
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlim(*NOISE_XLIM)
    ax.set_xlabel("Received signal Y")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Y given X: histograms vs theory (σ² = {variance:g})", fontsize=10)
    ax.legend(fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# Stress test: reliability vs noise variance
# ---------------------------------------------------------------------------
def stress_figure(rows, required, v_crit, amplitude, noise_mean=0.0, show_theory=True):
    vs = np.array([r["variance"] for r in rows])
    rel = np.array([r["reliability"] for r in rows])
    se = np.array([r["std_error"] for r in rows])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(vs, rel - 1.96 * se, rel + 1.96 * se, color="blue", alpha=0.18,
                    label="95% sampling interval")
    ax.plot(vs, rel, "b-o", ms=3, label="Simulated reliability")
    if show_theory:
        vf = np.linspace(vs.min(), vs.max(), 300)
        ax.plot(vf, theoretical_reliability(vf, amplitude, noise_mean), color="orange",
                lw=1.5, label="Theoretical reliability Φ(A/σ)" if noise_mean == 0
                else "Theoretical reliability")
    ax.axhline(required, color="gray", linestyle="--",
               label=f"Required reliability ({required*100:.0f}%)")
    if v_crit is not None and vs.min() <= v_crit <= vs.max():
        ax.axvline(v_crit, color="red", linestyle="--",
                   label=f"Critical variance ≈ {v_crit:.3f}")
    ax.set_xlim(vs.min(), vs.max())
    ax.set_xlabel("Gaussian noise variance")
    ax.set_ylabel("Reliability")
    ax.set_title("Reliability vs noise variance")
    ax.legend(fontsize=8)
    return fig
