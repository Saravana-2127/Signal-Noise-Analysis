"""
Week 3 - Stress Testing & What-If Analysis (project novelty)

Covers:
  Step 16: Probabilistic stress testing -- sweep noise variance, measure
           reliability/correlation/anomaly-rate at each point.
  Step 17: Critical reliability condition -- find the noise variance at
           which reliability crosses a required threshold (e.g. 90%).
  Step 18: What-if scenario comparison.

Model (single, consistent): X in {-A, +A}, N ~ N(mu, sigma^2), Y = X + N,
decision threshold 0 (Y >= 0 -> bit 1).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from core import run_simulation
from probability_analysis import joint_and_correlation, anomaly_detection

DECISION_THRESHOLD = 0.0   # fixed in core.detect_bits; identical for every run


# ---------------------------------------------------------------------------
# Theoretical reliability
#   mu = 0 :  R(sigma^2) = Phi(A / sigma)
#   general:  R = 0.5 * [ Phi((A+mu)/sigma) + Phi((A-mu)/sigma) ]
# ---------------------------------------------------------------------------
def theoretical_reliability(variance, amplitude: float = 1.0, noise_mean: float = 0.0):
    sigma = np.sqrt(variance)
    r = 0.5 * (norm.cdf((amplitude + noise_mean) / sigma)
               + norm.cdf((amplitude - noise_mean) / sigma))
    return float(r) if np.ndim(r) == 0 else r


# ---------------------------------------------------------------------------
# Step 16: Stress test -- run the full simulation across a range of noise
# variances and record reliability + correlation + anomaly rate at each.
#
# The bit sequence is identical for every variance (same `seed`). With
# independent_noise=True each variance gets its own independent noise draw,
# so the curve shows genuine sampling variation (see std_error).
# ---------------------------------------------------------------------------
def stress_test(n_bits: int, amplitude: float, variances, seed: int = 42,
                noise_mean: float = 0.0, independent_noise: bool = True) -> list[dict]:
    rows = []
    for i, var in enumerate(variances):
        sim = run_simulation(n_bits=n_bits, amplitude=amplitude,
                              noise_mean=noise_mean, noise_variance=var, seed=seed,
                              noise_seed=(seed + 1000 + i) if independent_noise else None)
        jc = joint_and_correlation(sim["x"], sim["y"])
        ad = anomaly_detection(sim["noise"], mean=noise_mean, variance=var)

        r = sim["reliability"]["reliability"]
        rows.append({
            "variance": var,
            "reliability": r,
            "error_probability": sim["reliability"]["error_probability"],
            "correlation": jc["correlation"],
            "anomaly_rate": ad["anomaly_rate"],
            "theoretical_reliability": theoretical_reliability(var, amplitude, noise_mean),
            "std_error": float(np.sqrt(r * (1 - r) / n_bits)),   # binomial standard error
        })
    return rows


# ---------------------------------------------------------------------------
# Step 17: Critical reliability condition.
# Solve theoretical_reliability(variance) == required_reliability by root
# finding (calculated dynamically from the selected target, never hardcoded).
# Raises ValueError when no finite critical variance exists.
# ---------------------------------------------------------------------------
def find_critical_variance(required_reliability: float, amplitude: float = 1.0,
                            search_range=(1e-6, 1e8), noise_mean: float = 0.0) -> float:
    f = lambda v: theoretical_reliability(v, amplitude, noise_mean) - required_reliability
    lo, hi = search_range
    if f(lo) < 0:
        raise ValueError("The required reliability cannot be reached even with almost "
                         "no noise (check the noise mean and amplitude).")
    if f(hi) > 0:
        raise ValueError("The required reliability is met at every noise variance "
                         "(no finite critical variance exists).")
    return brentq(f, lo, hi)


def critical_condition_from_table(rows: list[dict], required_reliability: float) -> dict:
    """
    Locate the critical noise variance from the discrete stress-test table by
    finding the bracketing pair of variances where reliability crosses the
    requirement, then linearly interpolating between them.
    """
    rows_sorted = sorted(rows, key=lambda r: r["variance"])
    for i in range(len(rows_sorted) - 1):
        r_lo, r_hi = rows_sorted[i], rows_sorted[i + 1]
        if r_lo["reliability"] >= required_reliability >= r_hi["reliability"]:
            v_lo, R_lo = r_lo["variance"], r_lo["reliability"]
            v_hi, R_hi = r_hi["variance"], r_hi["reliability"]
            if R_hi == R_lo:
                v_crit = v_lo
            else:
                frac = (required_reliability - R_lo) / (R_hi - R_lo)
                v_crit = v_lo + frac * (v_hi - v_lo)
            return {
                "bracket": (v_lo, v_hi),
                "reliability_bracket": (R_lo, R_hi),
                "critical_variance_interpolated": v_crit,
            }
    return {"bracket": None, "message": "Requirement not crossed in tested range."}


# ---------------------------------------------------------------------------
# Step 18: What-if scenario comparison
# ---------------------------------------------------------------------------
def compare_scenarios(scenarios: list[dict], n_bits: int = 100_000, seed: int = 42,
                      noise_mean: float = 0.0, independent_noise: bool = True) -> list[dict]:
    """
    Each scenario dict needs: name, amplitude, noise_variance (extra keys such
    as 'level' are passed through). Bits, amplitude handling, bit count and
    decision threshold are identical across scenarios; only the variance
    changes. Returns each scenario enriched with simulated + theoretical
    reliability.
    """
    results = []
    for i, sc in enumerate(scenarios):
        sim = run_simulation(n_bits=n_bits, amplitude=sc["amplitude"],
                              noise_mean=noise_mean, noise_variance=sc["noise_variance"],
                              seed=seed,
                              noise_seed=(seed + 2000 + i) if independent_noise else None)
        results.append({
            **sc,
            "reliability": sim["reliability"]["reliability"],
            "error_probability": sim["reliability"]["error_probability"],
            "theoretical_reliability": theoretical_reliability(
                sc["noise_variance"], sc["amplitude"], noise_mean),
        })
    return results


if __name__ == "__main__":
    variances = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]

    print("=== Stress test: reliability vs noise variance ===")
    rows = stress_test(n_bits=100_000, amplitude=1.0, variances=variances)
    for r in rows:
        print(f"  var={r['variance']:>4}: reliability={r['reliability']:.4f} "
              f"(theory {r['theoretical_reliability']:.4f})  "
              f"corr={r['correlation']:.4f}  anomaly_rate={r['anomaly_rate']:.4f}")

    print("\n=== Critical reliability condition (target = 90%) ===")
    v_crit_theory = find_critical_variance(required_reliability=0.90, amplitude=1.0)
    print(f"  theoretical critical variance: {v_crit_theory:.4f}")

    crit_from_table = critical_condition_from_table(rows, required_reliability=0.90)
    print(f"  from stress-test table: {crit_from_table}")

    print("\n=== What-if scenario comparison ===")
    scenarios = [
        {"name": "Scenario A", "amplitude": 1.0, "noise_variance": 0.5},
        {"name": "Scenario B", "amplitude": 1.0, "noise_variance": 1.5},
        {"name": "Scenario C", "amplitude": 1.0, "noise_variance": 2.5},
    ]
    for sc in compare_scenarios(scenarios):
        print(f"  {sc['name']}: amplitude={sc['amplitude']}, "
              f"variance={sc['noise_variance']}, "
              f"reliability={sc['reliability']:.4f} (theory {sc['theoretical_reliability']:.4f})")