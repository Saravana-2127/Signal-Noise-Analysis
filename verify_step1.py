import numpy as np
import matplotlib.pyplot as plt
from core import run_simulation, gaussian_pdf, gaussian_cdf

result = run_simulation(n_bits=100_000, noise_variance=0.5, seed=42)
noise = result["noise"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# PDF: histogram vs theoretical curve
axes[0].hist(noise, bins=80, density=True, alpha=0.6, label="Observed noise")
xs = np.linspace(noise.min(), noise.max(), 400)
axes[0].plot(xs, gaussian_pdf(xs, 0.0, 0.5), 'r-', lw=2, label="Theoretical PDF")
axes[0].set_title("Noise PDF: observed vs theoretical")
axes[0].set_xlabel("Noise value")
axes[0].legend()

# CDF: empirical vs theoretical
sorted_noise = np.sort(noise)
empirical_cdf = np.arange(1, len(sorted_noise) + 1) / len(sorted_noise)
axes[1].plot(sorted_noise, empirical_cdf, label="Empirical CDF", lw=2)
axes[1].plot(xs, gaussian_cdf(xs, 0.0, 0.5), 'r--', lw=2, label="Theoretical CDF")
axes[1].set_title("Noise CDF: observed vs theoretical")
axes[1].set_xlabel("Noise value")
axes[1].legend()

plt.tight_layout()
plt.savefig("step1_verification.png", dpi=120)
print("Saved step1_verification.png")
