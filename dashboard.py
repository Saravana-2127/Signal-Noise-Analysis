"""
Week 4 - Streamlit Dashboard

Run with:  streamlit run dashboard.py

Sections (matching the report's dashboard spec):
  Signal Configuration | Signal Analysis | Distribution Analysis |
  Reliability | Correlation | Anomaly Detection | Stress Testing
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm as _norm

from core import run_simulation
from probability_analysis import (
    conditional_and_bayes, joint_and_correlation, anomaly_detection,
)
from stress_testing import (
    stress_test, find_critical_variance, compare_scenarios,
    critical_condition_from_table, theoretical_reliability,
)
from plots import (
    MAX_VARIANCE, distribution_figures, signal_figure, joint_scatter_figure,
    joint_hexbin_figure, joint_conditional_figure, stress_figure,
)


def show(fig):
    """Render a matplotlib figure in Streamlit and free it."""
    st.pyplot(fig)
    plt.close(fig)


st.set_page_config(page_title="Wireless Signal Reliability Analysis", layout="wide")
st.title("📡 Gaussian Noise-Based Reliability & Stress Analysis")
st.caption("Wireless digital signal + Gaussian noise, simulated end-to-end.")

# ---------------------------------------------------------------------
# 🟢 Signal Configuration (sidebar)
# ---------------------------------------------------------------------
st.sidebar.header("🟢 Signal Configuration")
n_bits = st.sidebar.number_input("Number of bits", min_value=1000, max_value=1_000_000,
                                  value=100_000, step=1000)
amplitude = st.sidebar.slider("Signal amplitude", 0.5, 3.0, 1.0, 0.1)
noise_mean = st.sidebar.slider("Noise mean", -1.0, 1.0, 0.0, 0.1)
noise_variance = st.sidebar.slider("Noise variance", 0.05, float(MAX_VARIANCE), 1.0, 0.05)
reliability_threshold = st.sidebar.slider("Reliability threshold (%)", 50, 99, 90) / 100

# Same seed -> same transmitted bits for every parameter setting; only the
# noise (scaled by the selected variance) and the received signal change.
sim = run_simulation(n_bits=int(n_bits), amplitude=amplitude,
                      noise_mean=noise_mean, noise_variance=noise_variance, seed=42)

# ---------------------------------------------------------------------
# 🟢 Signal Analysis
# ---------------------------------------------------------------------
st.header("🟢 Signal Analysis")
show(signal_figure(sim["x"], sim["y"], noise_variance, amplitude, n_show=100))
st.caption("The transmitted bits are fixed (same seed); changing the noise variance "
           "changes only the received signal. The y-axis is fixed so the effect is visible.")

# ---------------------------------------------------------------------
# 🟢 Distribution Analysis
# ---------------------------------------------------------------------
st.header("🟢 Distribution Analysis")
fig_pdf, fig_cdf, ks_distance = distribution_figures(sim["noise"], noise_mean, noise_variance)
col1, col2 = st.columns(2)
with col1:
    show(fig_pdf)
with col2:
    show(fig_cdf)
st.caption("Fixed x-axis (−9 to 9) so the spread is comparable across variances. "
           "Histogram uses density normalization.")

stats = sim["stats"]
c0, c1, c2, c3, c4 = st.columns(5)
c0.metric("Selected variance (σ²)", f"{noise_variance:.2f}")
c1.metric("Theoretical mean", f"{stats['theoretical_mean']:.3f}")
c2.metric("Observed mean", f"{stats['observed_mean']:.3f}")
c3.metric("Theoretical variance", f"{stats['theoretical_variance']:.3f}")
c4.metric("Observed variance", f"{stats['observed_variance']:.3f}")
st.caption(f"Max gap between empirical and theoretical CDF: {ks_distance:.4f}")

# ---------------------------------------------------------------------
# 🟢 Reliability
# ---------------------------------------------------------------------
st.header("🟢 Reliability")
rel = sim["reliability"]
c1, c2, c3 = st.columns(3)
c1.metric("Reliability", f"{rel['reliability']*100:.2f}%")
c2.metric("Error probability", f"{rel['error_probability']*100:.2f}%")
c3.metric("Errors / Total", f"{rel['errors']} / {rel['total_bits']}")

# ---------------------------------------------------------------------
# 🟢 Correlation
# ---------------------------------------------------------------------
st.header("🟢 Correlation")
jc = joint_and_correlation(sim["x"], sim["y"])
c1, c2 = st.columns(2)
c1.metric("Covariance(X, Y)", f"{jc['covariance']:.4f}")
c2.metric("Correlation(X, Y)", f"{jc['correlation']:.4f}")

# Joint distribution: three views of the same simulated (X, Y) data
tab_scatter, tab_hex, tab_cond = st.tabs(["Scatter", "Hexbin density", "Y given X"])
with tab_scatter:
    show(joint_scatter_figure(sim["x"], sim["y"], amplitude, noise_variance))
with tab_hex:
    show(joint_hexbin_figure(sim["x"], sim["y"], amplitude, noise_variance))
with tab_cond:
    show(joint_conditional_figure(sim["x"], sim["y"], amplitude, noise_mean, noise_variance))
st.caption("X takes only two values (−A, +A), so the joint distribution is two vertical "
           "bands; transparency / density views show how Y spreads around each.")

# ---------------------------------------------------------------------
# 🟢 Anomaly Detection
# ---------------------------------------------------------------------
st.header("🟢 Anomaly Detection")
density_threshold = st.slider("Anomaly density threshold", 0.001, 0.1, 0.01, 0.001)
ad = anomaly_detection(sim["noise"], mean=noise_mean, variance=noise_variance,
                        density_threshold=density_threshold)
st.metric("Anomaly rate", f"{ad['anomaly_rate']*100:.2f}%")

manual_obs = st.number_input("Test a specific observation", value=0.5)
density = _norm.pdf(manual_obs, loc=noise_mean, scale=np.sqrt(noise_variance))
status = "ANOMALOUS" if density < density_threshold else "Normal"
st.write(f"Observation = {manual_obs} → likelihood density = {density:.5f} → **{status}**")

# ---------------------------------------------------------------------
# 🔥 Stress Testing + Critical Reliability Condition
# ---------------------------------------------------------------------
st.header("🔥 Stress Testing")
variances = np.linspace(0.05, MAX_VARIANCE, 25)
n_stress = min(int(n_bits), 20_000)
rows = stress_test(n_bits=n_stress, amplitude=amplitude, variances=variances,
                   noise_mean=noise_mean)

try:
    v_crit = find_critical_variance(required_reliability=reliability_threshold,
                                    amplitude=amplitude, noise_mean=noise_mean)
    crit_error = None
except ValueError as e:
    v_crit, crit_error = None, str(e)

show_theory = st.checkbox("Show theoretical reliability curve (validation)", value=True)
show(stress_figure(rows, reliability_threshold, v_crit, amplitude, noise_mean, show_theory))

if v_crit is not None:
    st.success(f"Critical noise variance for {reliability_threshold*100:.0f}% reliability "
               f"requirement: **{v_crit:.3f}** (σ ≈ {np.sqrt(v_crit):.3f}). "
               f"Beyond this variance, reliability falls below the "
               f"{reliability_threshold*100:.0f}% target.")
    crossing = critical_condition_from_table(rows, reliability_threshold)
    if crossing.get("bracket"):
        st.caption(f"Simulated curve crosses the target at ≈ "
                   f"{crossing['critical_variance_interpolated']:.3f} "
                   f"(differences from theory come from random sampling; "
                   f"{n_stress:,} bits per point, shaded band = 95% interval).")
else:
    st.warning(crit_error)

with st.expander("Simulated vs theoretical reliability values"):
    st.dataframe([
        {"Variance": round(float(r["variance"]), 3),
         "Simulated reliability (%)": round(r["reliability"] * 100, 2),
         "Theoretical reliability (%)": round(r["theoretical_reliability"] * 100, 2),
         "± 95% (%)": round(1.96 * r["std_error"] * 100, 2)}
        for r in rows
    ])

# ---------------------------------------------------------------------
# What-if scenario comparison
# ---------------------------------------------------------------------
st.subheader("What-If Scenario Comparison")
levels = [("Scenario A", "Low noise variance", 0.5),
          ("Scenario B", "Moderate noise variance", 1.0),
          ("Scenario C", "High noise variance", 1.5)]
scenarios = [{"name": n, "level": lv, "amplitude": amplitude,
              "noise_variance": noise_variance * m} for n, lv, m in levels]
# Same amplitude, bit count, bits and decision threshold in every scenario;
# only the noise variance differs (independent noise draw per scenario).
sc_results = compare_scenarios(scenarios, n_bits=int(n_bits), noise_mean=noise_mean)
st.table([
    {"Scenario": r["name"],
     "Noise level": f"{r['level']} ({m:g}× selected)",
     "Variance": round(r["noise_variance"], 3),
     "Simulated reliability": f"{r['reliability']*100:.2f}%",
     "Theoretical reliability": f"{r['theoretical_reliability']*100:.2f}%",
     "Required": f"{reliability_threshold*100:.0f}%",
     "Status": "✅ PASS" if r["reliability"] >= reliability_threshold else "❌ FAIL"}
    for r, (_, _, m) in zip(sc_results, levels)
])
st.caption("Status compares the simulated reliability with the required threshold in the sidebar.")