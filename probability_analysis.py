"""
Week 2 - Probability Analysis
Builds directly on core.py's simulation output (bits, X, noise, Y, detected).

Covers:
  Step 11: Conditional probability      P(A|B)
  Step 12: Bayes theorem                P(A|B) = P(B|A)P(A) / P(B)
  Step 13: Joint distribution of X and Y
  Step 14: Covariance and correlation between X and Y
  Step 15: Likelihood-based anomaly detection
  Step 19: Monte Carlo simulation (repeat + average)
  Step 20: Central Limit Theorem experiment
"""

import numpy as np
from scipy.stats import norm
from core import run_simulation, generate_noise


# ---------------------------------------------------------------------------
# Step 11-12: Conditional probability + Bayes theorem
#
# Event A = "detection is incorrect"   (original bit != detected bit)
# Event B = "noise exceeds a threshold" (|noise| > threshold)
# ---------------------------------------------------------------------------
def conditional_and_bayes(bits: np.ndarray, detected: np.ndarray, noise: np.ndarray,
                           noise_threshold: float) -> dict:
    event_A = bits != detected                 # detection error
    event_B = np.abs(noise) > noise_threshold   # large noise

    p_A = np.mean(event_A)
    p_B = np.mean(event_B)

    # P(A|B) = P(A and B) / P(B)  -- computed directly (empirical conditional probability)
    both = event_A & event_B
    p_A_and_B = np.mean(both)
    p_A_given_B = p_A_and_B / p_B if p_B > 0 else float("nan")

    # P(B|A) = P(A and B) / P(A)
    p_B_given_A = p_A_and_B / p_A if p_A > 0 else float("nan")

    # Bayes theorem reconstruction of P(A|B) from P(B|A), P(A), P(B)
    # (should match the direct empirical value above -- this is the validation)
    p_A_given_B_bayes = (p_B_given_A * p_A) / p_B if p_B > 0 else float("nan")

    return {
        "P(A) error rate": p_A,
        "P(B) large-noise rate": p_B,
        "P(A|B) direct": p_A_given_B,
        "P(B|A)": p_B_given_A,
        "P(A|B) via Bayes": p_A_given_B_bayes,
    }


# ---------------------------------------------------------------------------
# Step 13-14: Joint distribution of X (transmitted) and Y (received),
# plus covariance and correlation coefficient.
# ---------------------------------------------------------------------------
def joint_and_correlation(x: np.ndarray, y: np.ndarray) -> dict:
    covariance = float(np.cov(x, y, ddof=0)[0, 1])
    correlation = float(np.corrcoef(x, y)[0, 1])
    return {
        "covariance": covariance,
        "correlation": correlation,
    }


def joint_histogram(x: np.ndarray, y: np.ndarray, bins: int = 40):
    """Return a 2D histogram (counts, x_edges, y_edges) for plotting X vs Y."""
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    return counts, x_edges, y_edges


# ---------------------------------------------------------------------------
# Step 15: Likelihood-based anomaly detection
#
# Under the assumed noise model N ~ N(mean, variance), compute the
# probability density of each observation. Very low density => anomaly.
# ---------------------------------------------------------------------------
def anomaly_detection(observations: np.ndarray, mean: float, variance: float,
                       density_threshold: float = 0.01) -> dict:
    std = np.sqrt(variance)
    densities = norm.pdf(observations, loc=mean, scale=std)
    is_anomaly = densities < density_threshold
    return {
        "densities": densities,
        "is_anomaly": is_anomaly,
        "anomaly_rate": float(np.mean(is_anomaly)),
    }


# ---------------------------------------------------------------------------
# Step 19: Monte Carlo simulation
#
# Repeat the whole communication simulation many times and average the
# reliability estimate, to show the estimate stabilizing.
# ---------------------------------------------------------------------------
def monte_carlo_reliability(n_bits: int, noise_variance: float,
                             n_runs_list=(10, 100, 1000)) -> dict:
    results = {}
    for n_runs in n_runs_list:
        run_reliabilities = []
        for run_idx in range(n_runs):
            sim = run_simulation(n_bits=n_bits, noise_variance=noise_variance,
                                  seed=1000 + run_idx)
            run_reliabilities.append(sim["reliability"]["reliability"])
        run_reliabilities = np.array(run_reliabilities)
        results[n_runs] = {
            "mean_reliability": float(np.mean(run_reliabilities)),
            "std_reliability": float(np.std(run_reliabilities)),
        }
    return results


# ---------------------------------------------------------------------------
# Step 20: Central Limit Theorem experiment
#
# Draw samples from a (deliberately non-Gaussian-looking-in-small-samples)
# distribution, form sample means for several sample sizes, and show the
# distribution of sample means approaching a Gaussian shape.
# ---------------------------------------------------------------------------
def clt_experiment(sample_sizes=(10, 50, 100), n_trials: int = 5000,
                    seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    results = {}
    for n in sample_sizes:
        # underlying population: uniform(-1, 1) -- clearly non-Gaussian itself
        samples = rng.uniform(-1, 1, size=(n_trials, n))
        sample_means = samples.mean(axis=1)
        results[n] = sample_means
    return results


if __name__ == "__main__":
    sim = run_simulation(n_bits=100_000, noise_variance=0.5, seed=42)

    print("=== Conditional probability & Bayes ===")
    cb = conditional_and_bayes(sim["bits"], sim["detected"], sim["noise"],
                                noise_threshold=1.0)
    for k, v in cb.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Joint distribution: covariance & correlation (X vs Y) ===")
    jc = joint_and_correlation(sim["x"], sim["y"])
    for k, v in jc.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Anomaly detection on noise samples ===")
    ad = anomaly_detection(sim["noise"], mean=0.0, variance=0.5)
    print(f"  anomaly_rate: {ad['anomaly_rate']:.4f}")

    print("\n=== Monte Carlo reliability convergence ===")
    mc = monte_carlo_reliability(n_bits=5000, noise_variance=0.5,
                                  n_runs_list=(10, 100))
    for n_runs, stats in mc.items():
        print(f"  n_runs={n_runs}: mean={stats['mean_reliability']:.4f}, "
              f"std={stats['std_reliability']:.5f}")

    print("\n=== CLT experiment (sample-mean std shrinks as sample size grows) ===")
    clt = clt_experiment()
    for n, means in clt.items():
        print(f"  sample_size={n}: mean of means={np.mean(means):.4f}, "
              f"std of means={np.std(means):.4f}")
