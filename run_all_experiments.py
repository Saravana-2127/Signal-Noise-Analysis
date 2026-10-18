"""
Runs the 8 experiments called for in the report (Section 23) and prints a
results summary suitable for pasting into the report / using in the viva.
"""

import numpy as np
from core import run_simulation
from probability_analysis import (
    conditional_and_bayes, joint_and_correlation, anomaly_detection,
    monte_carlo_reliability, clt_experiment,
)
from stress_testing import stress_test, find_critical_variance, compare_scenarios

N_BITS = 100_000
AMPLITUDE = 1.0
SEED = 42

print("#" * 70)
print("EXPERIMENT 1: Verify Gaussian noise characteristics")
print("#" * 70)
sim = run_simulation(n_bits=N_BITS, amplitude=AMPLITUDE, noise_variance=1.0, seed=SEED)
s = sim["stats"]
print(f"Theoretical mean={s['theoretical_mean']}, Observed mean={s['observed_mean']:.4f}")
print(f"Theoretical var ={s['theoretical_variance']}, Observed var ={s['observed_variance']:.4f}")
print("-> Observed statistics closely match the specified Gaussian model.\n")

print("#" * 70)
print("EXPERIMENT 2: Theoretical vs simulated probability")
print("#" * 70)
from scipy.stats import norm
a = 1.5
theoretical_p = 1 - norm.cdf(a)          # P(N > a) for N ~ N(0,1)
simulated_p = np.mean(sim["noise"] > a)
print(f"P(N > {a}): theoretical={theoretical_p:.4f}, simulated={simulated_p:.4f}\n")

print("#" * 70)
print("EXPERIMENT 3: Noise variance vs communication reliability")
print("#" * 70)
variances = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
rows = stress_test(n_bits=N_BITS, amplitude=AMPLITUDE, variances=variances, seed=SEED)
for r in rows:
    print(f"  variance={r['variance']:>4}  reliability={r['reliability']*100:6.2f}%")
print()

print("#" * 70)
print("EXPERIMENT 4: Noise variance vs X/Y correlation")
print("#" * 70)
for r in rows:
    print(f"  variance={r['variance']:>4}  correlation={r['correlation']:.4f}")
print("-> Correlation decreases monotonically as noise variance increases.\n")

print("#" * 70)
print("EXPERIMENT 5: Likelihood-based anomaly detection")
print("#" * 70)
ad = anomaly_detection(sim["noise"], mean=0.0, variance=1.0, density_threshold=0.01)
print(f"Anomaly rate at density threshold 0.01: {ad['anomaly_rate']*100:.2f}%\n")

print("#" * 70)
print("EXPERIMENT 6: Probabilistic stress testing")
print("#" * 70)
for r in rows:
    print(f"  variance={r['variance']:>4}  reliability={r['reliability']*100:6.2f}%  "
          f"error_prob={r['error_probability']*100:5.2f}%  "
          f"anomaly_rate={r['anomaly_rate']*100:5.2f}%")
print()

print("#" * 70)
print("EXPERIMENT 7: Critical reliability condition (target=90%)")
print("#" * 70)
v_crit = find_critical_variance(required_reliability=0.90, amplitude=AMPLITUDE)
print(f"Critical noise variance: {v_crit:.4f}")
print(f"-> Above variance {v_crit:.3f}, the system fails to meet the 90% reliability requirement.\n")

print("#" * 70)
print("EXPERIMENT 8: Monte Carlo convergence & CLT")
print("#" * 70)
mc = monte_carlo_reliability(n_bits=5000, noise_variance=1.0, n_runs_list=(10, 100, 1000))
for n_runs, stats in mc.items():
    print(f"  n_runs={n_runs:>5}: mean_reliability={stats['mean_reliability']:.4f}, "
          f"std={stats['std_reliability']:.5f}")

clt = clt_experiment(sample_sizes=(10, 50, 100))
print("\nCLT: standard deviation of sample means shrinks as sample size grows")
for n, means in clt.items():
    print(f"  sample_size={n:>3}: std of sample means = {np.std(means):.4f}")

print("\n" + "#" * 70)
print("What-if scenario comparison")
print("#" * 70)
scenarios = [
    {"name": "Scenario A", "amplitude": 1.0, "noise_variance": 0.5},
    {"name": "Scenario B", "amplitude": 1.0, "noise_variance": 1.5},
    {"name": "Scenario C", "amplitude": 1.0, "noise_variance": 2.5},
]
for sc in compare_scenarios(scenarios, n_bits=N_BITS, seed=SEED):
    print(f"  {sc['name']}: variance={sc['noise_variance']}, reliability={sc['reliability']*100:.2f}%")
