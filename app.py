"""
Wireless Communication Reliability & Stress Analysis Dashboard
Combined Single-File Application
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.io import wavfile

# =============================================================================
# 1. CONSTANTS & GLOBALS
# =============================================================================
MAX_VARIANCE = 4.0
NOISE_XLIM = (-9.0, 9.0)
DECISION_THRESHOLD = 0.0

# =============================================================================
# 2. CORE MATHEMATICAL MODEL
# =============================================================================
def generate_bits(n_bits: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits)

def bits_to_signal(bits: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    return np.where(bits == 1, amplitude, -amplitude).astype(float)

def generate_noise(n_samples: int, mean: float = 0.0, variance: float = 1.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = np.sqrt(variance)
    return rng.normal(loc=mean, scale=std, size=n_samples)

def received_signal(x: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return x + noise

def detect_bits(y: np.ndarray) -> np.ndarray:
    return (y >= DECISION_THRESHOLD).astype(int)

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

def noise_statistics(noise: np.ndarray, theoretical_mean: float, theoretical_variance: float) -> dict:
    return {
        "theoretical_mean": theoretical_mean,
        "observed_mean": float(np.mean(noise)),
        "theoretical_variance": theoretical_variance,
        "observed_variance": float(np.var(noise)),
    }

def gaussian_pdf(x_values: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return norm.pdf(x_values, loc=mean, scale=np.sqrt(variance))

def gaussian_cdf(x_values: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return norm.cdf(x_values, loc=mean, scale=np.sqrt(variance))

def run_simulation(n_bits: int = 100_000, amplitude: float = 1.0, noise_mean: float = 0.0, 
                   noise_variance: float = 1.0, seed: int | None = 42, noise_seed: int | None = None,
                   input_bits: np.ndarray = None) -> dict:
    
    if input_bits is not None:
        bits = input_bits
        n_bits = len(bits)
    else:
        bits = generate_bits(n_bits, seed=seed)
        
    x = bits_to_signal(bits, amplitude=amplitude)
    
    if noise_seed is None:
        noise_seed = None if seed is None else seed + 1
        
    noise = generate_noise(n_bits, mean=noise_mean, variance=noise_variance, seed=noise_seed)
    y = received_signal(x, noise)
    detected = detect_bits(y)
    
    return {
        "bits": bits,
        "x": x,
        "noise": noise,
        "y": y,
        "detected": detected,
        "reliability": reliability(bits, detected),
        "stats": noise_statistics(noise, noise_mean, noise_variance),
    }

# =============================================================================
# 3. AUDIO PROCESSING MODEL (PCM)
# =============================================================================
@st.cache_data(show_spinner=False)
def cached_audio_to_bits(file_bytes, max_duration_sec=5):
    rate, data = wavfile.read(io.BytesIO(file_bytes))
    if data.ndim > 1:
        data = data.mean(axis=1)
        
    max_samples = int(rate * max_duration_sec)
    if len(data) > max_samples:
        data = data[:max_samples]
        
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)
    data_8bit = (data_norm * 255).astype(np.uint8)
    bits = np.unpackbits(data_8bit)
    return bits, rate

def bits_to_audio(bits, rate):
    trim_length = len(bits) - (len(bits) % 8)
    bits = bits[:trim_length]
    data_8bit = np.packbits(bits)
    data_audio = (data_8bit.astype(np.float32) / 255.0) * 2.0 - 1.0
    buffer = io.BytesIO()
    wavfile.write(buffer, rate, data_audio)
    return buffer.getvalue()

# =============================================================================
# 4. PROBABILITY ANALYSIS
# =============================================================================
def conditional_and_bayes(bits: np.ndarray, detected: np.ndarray, noise: np.ndarray, noise_threshold: float) -> dict:
    event_A = bits != detected
    event_B = np.abs(noise) > noise_threshold
    p_A = np.mean(event_A)
    p_B = np.mean(event_B)
    both = event_A & event_B
    p_A_and_B = np.mean(both)
    
    p_A_given_B = p_A_and_B / p_B if p_B > 0 else float("nan")
    p_B_given_A = p_A_and_B / p_A if p_A > 0 else float("nan")
    p_A_given_B_bayes = (p_B_given_A * p_A) / p_B if p_B > 0 else float("nan")
    
    return {
        "P(A) error rate": p_A,
        "P(B) large-noise rate": p_B,
        "P(A|B) direct": p_A_given_B,
        "P(B|A)": p_B_given_A,
        "P(A|B) via Bayes": p_A_given_B_bayes,
    }

def joint_and_correlation(x: np.ndarray, y: np.ndarray) -> dict:
    return {
        "covariance": float(np.cov(x, y, ddof=0)[0, 1]),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
    }

def anomaly_detection(observations: np.ndarray, mean: float, variance: float, density_threshold: float = 0.01) -> dict:
    densities = norm.pdf(observations, loc=mean, scale=np.sqrt(variance))
    is_anomaly = densities < density_threshold
    return {
        "densities": densities,
        "is_anomaly": is_anomaly,
        "anomaly_rate": float(np.mean(is_anomaly)),
    }

# =============================================================================
# 5. STRESS TESTING
# =============================================================================
def theoretical_reliability(variance, amplitude: float = 1.0, noise_mean: float = 0.0):
    sigma = np.sqrt(variance)
    r = 0.5 * (norm.cdf((amplitude + noise_mean) / sigma) + norm.cdf((amplitude - noise_mean) / sigma))
    return float(r) if np.ndim(r) == 0 else r

def stress_test(n_bits: int, amplitude: float, variances, seed: int = 42, noise_mean: float = 0.0, input_bits: np.ndarray = None) -> list[dict]:
    rows = []
    for i, var in enumerate(variances):
        sim = run_simulation(n_bits=n_bits, amplitude=amplitude, noise_mean=noise_mean, 
                             noise_variance=var, seed=seed, noise_seed=(seed + 1000 + i), input_bits=input_bits)
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
            "std_error": float(np.sqrt(r * (1 - r) / len(sim["bits"]))),
        })
    return rows

def find_critical_variance(required_reliability: float, amplitude: float = 1.0, noise_mean: float = 0.0) -> float:
    f = lambda v: theoretical_reliability(v, amplitude, noise_mean) - required_reliability
    lo, hi = 1e-6, 1e8
    if f(lo) < 0:
        raise ValueError("Required reliability cannot be reached even with almost no noise.")
    if f(hi) > 0:
        raise ValueError("Required reliability is met at every noise variance.")
    return brentq(f, lo, hi)

def compare_scenarios(scenarios: list[dict], n_bits: int = 100_000, seed: int = 42, noise_mean: float = 0.0, input_bits: np.ndarray = None) -> list[dict]:
    results = []
    for i, sc in enumerate(scenarios):
        sim = run_simulation(n_bits=n_bits, amplitude=sc["amplitude"], noise_mean=noise_mean, 
                             noise_variance=sc["noise_variance"], seed=seed, noise_seed=(seed + 2000 + i), input_bits=input_bits)
        results.append({
            **sc,
            "reliability": sim["reliability"]["reliability"],
            "theoretical_reliability": theoretical_reliability(sc["noise_variance"], sc["amplitude"], noise_mean),
        })
    return results

# =============================================================================
# 6. PLOTTING FUNCTIONS
# =============================================================================
def _y_limit(amplitude: float) -> float:
    return amplitude + 3.0 * np.sqrt(MAX_VARIANCE)

def distribution_figures(noise: np.ndarray, mean: float, variance: float):
    xs = np.linspace(*NOISE_XLIM, 900)
    fig_pdf, ax = plt.subplots(figsize=(5, 4))
    ax.hist(noise, bins=np.linspace(*NOISE_XLIM, 181), density=True, alpha=0.6, label="Observed")
    ax.plot(xs, gaussian_pdf(xs, mean, variance), "r-", lw=2, label="Theoretical")
    ax.set_xlim(*NOISE_XLIM)
    ax.set_title(f"Noise PDF (σ² = {variance:g})", fontsize=10)
    ax.legend(fontsize=8)

    fig_cdf, ax2 = plt.subplots(figsize=(5, 4))
    sorted_noise = np.sort(noise)
    ecdf = np.searchsorted(sorted_noise, xs, side="right") / len(sorted_noise)
    tcdf = gaussian_cdf(xs, mean, variance)
    ax2.plot(xs, ecdf, lw=2, label="Empirical CDF")
    ax2.plot(xs, tcdf, "r--", lw=2, label="Theoretical CDF")
    ax2.set_xlim(*NOISE_XLIM)
    ax2.set_title(f"Noise CDF (σ² = {variance:g})", fontsize=10)
    ax2.legend(fontsize=8)
    
    return fig_pdf, fig_cdf, float(np.max(np.abs(ecdf - tcdf)))

def signal_figure(x, y, variance: float, amplitude: float, n_show: int = 100):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(x[:n_show], label="Transmitted (X)", drawstyle="steps-mid", lw=1.8)
    ax.plot(y[:n_show], label="Received (Y = X + N)", alpha=0.75, lw=1.2, marker="o", ms=2.5)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_ylim(-_y_limit(amplitude), _y_limit(amplitude))
    ax.set_title(f"Signal first {n_show} samples (σ² = {variance:g})")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    return fig

def joint_scatter_figure(x, y, amplitude, variance, max_points=3000):
    idx = np.random.default_rng(0).choice(len(x), size=min(max_points, len(x)), replace=False)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(x[idx], y[idx], s=10, alpha=0.15, edgecolors="none")
    ax.set_xlim(-amplitude - 0.75, amplitude + 0.75)
    ax.set_ylim(-_y_limit(amplitude), _y_limit(amplitude))
    ax.set_title(f"Joint distribution: scatter (σ² = {variance:g})", fontsize=10)
    return fig

def joint_hexbin_figure(x, y, amplitude, variance):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    lim = _y_limit(amplitude)
    hb = ax.hexbin(x, y, gridsize=(24, 40), mincnt=1, cmap="viridis", extent=(-amplitude - 0.75, amplitude + 0.75, -lim, lim))
    fig.colorbar(hb, ax=ax, label="Sample count")
    ax.set_title(f"Joint distribution: hexbin (σ² = {variance:g})", fontsize=10)
    return fig

def joint_conditional_figure(x, y, amplitude, noise_mean, variance):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ys = np.linspace(*NOISE_XLIM, 600)
    bins = np.linspace(*NOISE_XLIM, 121)
    for val, color in ((-amplitude, "tab:blue"), (amplitude, "tab:orange")):
        ax.hist(y[x == val], bins=bins, density=True, alpha=0.45, color=color, label=f"Y | X = {val:+g}")
        ax.plot(ys, gaussian_pdf(ys, val + noise_mean, variance), color=color, lw=2)
    ax.set_xlim(*NOISE_XLIM)
    ax.set_title(f"Y given X (σ² = {variance:g})", fontsize=10)
    ax.legend(fontsize=8)
    return fig

def stress_figure(rows, required, v_crit, amplitude, noise_mean=0.0):
    vs = np.array([r["variance"] for r in rows])
    rel = np.array([r["reliability"] for r in rows])
    se = np.array([r["std_error"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(vs, rel - 1.96 * se, rel + 1.96 * se, color="blue", alpha=0.18, label="95% sampling interval")
    ax.plot(vs, rel, "b-o", ms=3, label="Simulated reliability")
    vf = np.linspace(vs.min(), vs.max(), 300)
    ax.plot(vf, theoretical_reliability(vf, amplitude, noise_mean), color="orange", lw=1.5, label="Theoretical reliability")
    ax.axhline(required, color="gray", linestyle="--", label=f"Required ({required*100:.0f}%)")
    if v_crit is not None and vs.min() <= v_crit <= vs.max():
        ax.axvline(v_crit, color="red", linestyle="--", label=f"Critical σ² ≈ {v_crit:.3f}")
    ax.set_xlim(vs.min(), vs.max())
    ax.set_xlabel("Gaussian noise variance")
    ax.set_ylabel("Reliability")
    ax.legend(fontsize=8)
    return fig

# =============================================================================
# 7. STREAMLIT DASHBOARD UI
# =============================================================================
def show(fig):
    st.pyplot(fig)
    plt.close(fig)

st.set_page_config(page_title="Wireless Signal Reliability Analysis", layout="wide")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🟢 Data Source")
data_mode = st.sidebar.radio("Select signal input:", ["Random Bits", "Custom Audio (WAV)"])

source_bits = None
sample_rate = None
audio_file = None

if data_mode == "Custom Audio (WAV)":
    audio_file = st.sidebar.file_uploader("Upload Audio", type=["wav"])
    if audio_file is not None:
        with st.sidebar.status("Processing Audio...", expanded=False):
            source_bits, sample_rate = cached_audio_to_bits(audio_file.read(), max_duration_sec=5)
            st.write(f"Digitized into {len(source_bits):,} bits.")

st.sidebar.header("🟢 Signal Configuration")
if data_mode == "Random Bits":
    n_bits = st.sidebar.number_input("Number of bits", min_value=1000, max_value=1_000_000, value=100_000, step=1000)
else:
    n_bits = len(source_bits) if source_bits is not None else 100_000

# Added unique keys to strictly lock the UI state so it doesn't reset to default when interacting
amplitude = st.sidebar.slider("Signal amplitude", 0.5, 3.0, 1.0, 0.1, key="amp_slider")
noise_mean = st.sidebar.slider("Noise mean", -1.0, 1.0, 0.0, 0.1, key="mean_slider")
noise_variance = st.sidebar.slider("Noise variance", 0.05, float(MAX_VARIANCE), 1.0, 0.05, key="var_slider")
reliability_threshold = st.sidebar.slider("Reliability threshold (%)", 50, 99, 90, key="rel_slider") / 100

# --- MAIN SIMULATION EXECUTION ---
sim = run_simulation(n_bits=int(n_bits), amplitude=amplitude, noise_mean=noise_mean, 
                     noise_variance=noise_variance, seed=42, input_bits=source_bits)

# --- MAIN UI ---
st.title("📡 Gaussian Noise-Based Reliability & Stress Analysis")
st.caption("Wireless digital signal + Gaussian noise, simulated end-to-end.")

if data_mode == "Custom Audio (WAV)" and audio_file is not None:
    st.header("🎵 Real-Time Audio Transmission Playback")
    st.caption("The dashboard is currently analyzing your audio file. Listen to the signal degradation below.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Original Audio")
        st.audio(audio_file, format="audio/wav")
    with col2:
        st.subheader(f"2. Received Audio (σ² = {noise_variance:.2f})")
        received_audio_bytes = bits_to_audio(sim["detected"], sample_rate)
        # Removed the unsupported key parameter from audio
        st.audio(received_audio_bytes, format="audio/wav")
    st.markdown("---")

st.header("🟢 Signal Analysis")
show(signal_figure(sim["x"], sim["y"], noise_variance, amplitude, n_show=100))

st.header("🟢 Distribution Analysis")
fig_pdf, fig_cdf, ks_distance = distribution_figures(sim["noise"], noise_mean, noise_variance)
col1, col2 = st.columns(2)
with col1: show(fig_pdf)
with col2: show(fig_cdf)

stats = sim["stats"]
c0, c1, c2, c3, c4 = st.columns(5)
c0.metric("Selected variance (σ²)", f"{noise_variance:.2f}")
c1.metric("Theoretical mean", f"{stats['theoretical_mean']:.3f}")
c2.metric("Observed mean", f"{stats['observed_mean']:.3f}")
c3.metric("Theoretical variance", f"{stats['theoretical_variance']:.3f}")
c4.metric("Observed variance", f"{stats['observed_variance']:.3f}")

st.header("🟢 Reliability")
rel = sim["reliability"]
c1, c2, c3 = st.columns(3)
c1.metric("Reliability", f"{rel['reliability']*100:.2f}%")
c2.metric("Error probability", f"{rel['error_probability']*100:.2f}%")
c3.metric("Errors / Total", f"{rel['errors']:,} / {rel['total_bits']:,}")

st.header("🟢 Correlation")
jc = joint_and_correlation(sim["x"], sim["y"])
c1, c2 = st.columns(2)
c1.metric("Covariance(X, Y)", f"{jc['covariance']:.4f}")
c2.metric("Correlation(X, Y)", f"{jc['correlation']:.4f}")

tab_scatter, tab_hex, tab_cond = st.tabs(["Scatter", "Hexbin density", "Y given X"])
with tab_scatter: show(joint_scatter_figure(sim["x"], sim["y"], amplitude, noise_variance))
with tab_hex: show(joint_hexbin_figure(sim["x"], sim["y"], amplitude, noise_variance))
with tab_cond: show(joint_conditional_figure(sim["x"], sim["y"], amplitude, noise_mean, noise_variance))

st.header("🟢 Anomaly Detection")
density_threshold = st.slider("Anomaly density threshold", 0.001, 0.1, 0.01, 0.001, key="anomaly_slider")
ad = anomaly_detection(sim["noise"], mean=noise_mean, variance=noise_variance, density_threshold=density_threshold)
st.metric("Anomaly rate", f"{ad['anomaly_rate']*100:.2f}%")

st.header("🔥 Stress Testing")
st.caption("Testing the current signal across a full sweep of Gaussian noise variances.")
variances = np.linspace(0.05, MAX_VARIANCE, 25)

# Limit stress test bits slightly to prevent UI lag on slider changes, but maintain statistical validity
stress_bits = source_bits[:50000] if source_bits is not None else None
rows = stress_test(n_bits=min(int(n_bits), 50000), amplitude=amplitude, variances=variances, 
                   noise_mean=noise_mean, input_bits=stress_bits)

try:
    v_crit = find_critical_variance(required_reliability=reliability_threshold, amplitude=amplitude, noise_mean=noise_mean)
    crit_error = None
except ValueError as e:
    v_crit, crit_error = None, str(e)

show(stress_figure(rows, reliability_threshold, v_crit, amplitude, noise_mean))

if v_crit is not None:
    st.success(f"Critical noise variance for {reliability_threshold*100:.0f}% reliability requirement: **{v_crit:.3f}** (σ ≈ {np.sqrt(v_crit):.3f}).")
else:
    st.warning(crit_error)

st.subheader("What-If Scenario Comparison")
levels = [("Scenario A", "Low", 0.5), ("Scenario B", "Moderate", 1.0), ("Scenario C", "High", 1.5)]
scenarios = [{"name": n, "level": lv, "amplitude": amplitude, "noise_variance": noise_variance * m} for n, lv, m in levels]
sc_results = compare_scenarios(scenarios, n_bits=int(n_bits), noise_mean=noise_mean, input_bits=source_bits)

st.table([
    {"Scenario": r["name"], "Noise level": f"{r['level']} ({m:g}×)", "Variance": round(r["noise_variance"], 3),
     "Simulated reliability": f"{r['reliability']*100:.2f}%", "Required": f"{reliability_threshold*100:.0f}%",
     "Status": "✅ PASS" if r["reliability"] >= reliability_threshold else "❌ FAIL"}
    for r, (_, _, m) in zip(sc_results, levels)
])