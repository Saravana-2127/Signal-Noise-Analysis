import numpy as np
import matplotlib.pyplot as plt
from stress_testing import stress_test, find_critical_variance

variances_fine = np.linspace(0.05, 3.0, 40)
rows = stress_test(n_bits=50_000, amplitude=1.0, variances=variances_fine)

vs = [r["variance"] for r in rows]
rel = [r["reliability"] for r in rows]
corr = [r["correlation"] for r in rows]

v_crit = find_critical_variance(required_reliability=0.90, amplitude=1.0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(vs, rel, 'b-o', ms=3)
axes[0].axhline(0.90, color='gray', linestyle='--', label='Required reliability (90%)')
axes[0].axvline(v_crit, color='red', linestyle='--', label=f'Critical variance ≈ {v_crit:.2f}')
axes[0].set_xlabel("Gaussian noise variance")
axes[0].set_ylabel("Reliability")
axes[0].set_title("Reliability vs Noise Variance")
axes[0].legend()

axes[1].plot(vs, corr, 'g-o', ms=3)
axes[1].set_xlabel("Gaussian noise variance")
axes[1].set_ylabel("Correlation (X, Y)")
axes[1].set_title("Correlation vs Noise Variance")

plt.tight_layout()
plt.savefig("stress_test_results.png", dpi=120)
print("Saved stress_test_results.png")
print(f"Critical variance for 90% reliability requirement: {v_crit:.4f}")
