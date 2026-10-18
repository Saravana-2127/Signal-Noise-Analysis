"""
Week 1 - Mathematical Core
Gaussian Noise-Based Probabilistic Reliability Analysis of a Wireless
Communication Signal.

Pipeline:
  bits -> transmitted signal X -> + Gaussian noise N -> received signal Y
       -> threshold detection -> compare to original bits -> reliability
       -> mean/variance of noise -> PDF/CDF
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Step 1-2: Random bits + Step 3: bits -> signal (bipolar mapping)
# ---------------------------------------------------------------------------
def generate_bits(n_bits: int, seed: int | None = None) -> np.ndarray:
    """Generate n_bits random bits (0/1)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits)


def bits_to_signal(bits: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    """Map bit 0 -> -amplitude, bit 1 -> +amplitude."""
    return np.where(bits == 1, amplitude, -amplitude).astype(float)


# ---------------------------------------------------------------------------
# Step 4: Gaussian noise
# ---------------------------------------------------------------------------
def generate_noise(n_samples: int, mean: float = 0.0, variance: float = 1.0,
                    seed: int | None = None) -> np.ndarray:
    """Generate Gaussian noise N ~ N(mean, variance)."""
    rng = np.random.default_rng(seed)
    std = np.sqrt(variance)
    return rng.normal(loc=mean, scale=std, size=n_samples)


# ---------------------------------------------------------------------------
# Step 5: Received signal Y = X + N
# ---------------------------------------------------------------------------
def received_signal(x: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return x + noise


# ---------------------------------------------------------------------------
# Step 6: Threshold detection
# ---------------------------------------------------------------------------
def detect_bits(y: np.ndarray) -> np.ndarray:
    """Y >= 0 -> bit 1, Y < 0 -> bit 0."""
    return (y >= 0).astype(int)


# ---------------------------------------------------------------------------
# Step 7: Reliability / error probability
# ---------------------------------------------------------------------------
def reliability(original_bits: np.ndarray, detected_bits: np.ndarray) -> dict:
    total = len(original_bits)
    correct = int(np.sum(original_bits == detected_bits))
    errors = total - correct
    r = correct / total
    p_error = errors / total
    return {
        "total_bits": total,
        "correct_bits": correct,
        "errors": errors,
        "reliability": r,
        "error_probability": p_error,
    }


# ---------------------------------------------------------------------------
# Step 8: Statistical analysis of the noise (mean & variance, theoretical vs observed)
# ---------------------------------------------------------------------------
def noise_statistics(noise: np.ndarray, theoretical_mean: float, theoretical_variance: float) -> dict:
    observed_mean = float(np.mean(noise))
    observed_variance = float(np.var(noise))
    return {
        "theoretical_mean": theoretical_mean,
        "observed_mean": observed_mean,
        "theoretical_variance": theoretical_variance,
        "observed_variance": observed_variance,
    }


# ---------------------------------------------------------------------------
# Step 9: PDF and CDF (theoretical, for plotting against the noise histogram)
# ---------------------------------------------------------------------------
def gaussian_pdf(x_values: np.ndarray, mean: float, variance: float) -> np.ndarray:
    std = np.sqrt(variance)
    return norm.pdf(x_values, loc=mean, scale=std)


def gaussian_cdf(x_values: np.ndarray, mean: float, variance: float) -> np.ndarray:
    std = np.sqrt(variance)
    return norm.cdf(x_values, loc=mean, scale=std)


# ---------------------------------------------------------------------------
# Convenience: run the whole Week-1 pipeline in one call
# ---------------------------------------------------------------------------
def run_simulation(n_bits: int = 100_000, amplitude: float = 1.0,
                    noise_mean: float = 0.0, noise_variance: float = 1.0,
                    seed: int | None = 42, noise_seed: int | None = None) -> dict:
    """
    seed       -> controls the transmitted bit sequence (same seed = same bits).
    noise_seed -> controls the noise samples. If None it defaults to seed + 1
                  (original behaviour). Pass different noise_seed values to get
                  independent noise for different variances with the SAME bits.
    """
    bits = generate_bits(n_bits, seed=seed)
    x = bits_to_signal(bits, amplitude=amplitude)
    if noise_seed is None:
        noise_seed = None if seed is None else seed + 1
    noise = generate_noise(n_bits, mean=noise_mean, variance=noise_variance,
                            seed=noise_seed)
    y = received_signal(x, noise)
    detected = detect_bits(y)

    rel = reliability(bits, detected)
    stats = noise_statistics(noise, noise_mean, noise_variance)

    return {
        "bits": bits,
        "x": x,
        "noise": noise,
        "y": y,
        "detected": detected,
        "reliability": rel,
        "stats": stats,
    }


if __name__ == "__main__":
    result = run_simulation(n_bits=100_000, amplitude=1.0,
                             noise_mean=0.0, noise_variance=0.5, seed=42)

    print("=== Reliability ===")
    for k, v in result["reliability"].items():
        print(f"  {k}: {v}")

    print("\n=== Noise statistics (theoretical vs observed) ===")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")

    print("\n=== Sanity check: first 10 samples ===")
    print("bits:     ", result["bits"][:10])
    print("X:        ", np.round(result["x"][:10], 3))
    print("noise:    ", np.round(result["noise"][:10], 3))
    print("Y:        ", np.round(result["y"][:10], 3))
    print("detected: ", result["detected"][:10])