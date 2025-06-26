import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="A/B Test Power Calculator", layout="wide")


def get_bloom_multiplier(alpha, power, two_sided=False):
    """Get Bloom's multiplier M = t_{1-κ} + t_α"""
    t_power = stats.norm.ppf(power)
    if two_sided:
        t_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        t_alpha = stats.norm.ppf(1 - alpha)
    return t_power + t_alpha


def calculate_mde(
    n,
    baseline_rate,
    treatment_fraction=0.5,
    alpha=0.05,
    power=0.8,
    two_sided=False,
    r_squared=0,
):
    """Calculate MDE given sample size"""
    variance_term = baseline_rate * (1 - baseline_rate) * (1 - r_squared)
    denominator = treatment_fraction * (1 - treatment_fraction)
    multiplier = get_bloom_multiplier(alpha, power, two_sided)

    se = np.sqrt(variance_term / (denominator * n))
    return multiplier * se


def calculate_sample_size(
    mde,
    baseline_rate,
    treatment_fraction=0.5,
    alpha=0.05,
    power=0.8,
    two_sided=False,
    r_squared=0,
):
    """Calculate sample size given MDE"""
    variance_term = baseline_rate * (1 - baseline_rate) * (1 - r_squared)
    denominator = treatment_fraction * (1 - treatment_fraction)
    multiplier = get_bloom_multiplier(alpha, power, two_sided)

    return int(np.ceil((multiplier**2 * variance_term) / (mde**2 * denominator)))

##     ## #### ########
##     ##  ##       ##
##     ##  ##      ##
##     ##  ##     ##
 ##   ##   ##    ##
  ## ##    ##   ##
   ###    #### ########

st.title("Power Calculator for Binary Outcomes")

st.markdown("""

The general MDE formula is

$$
\\text{MDE} = M \\times \\sqrt{\\frac{1}{P ( 1- P)} \\frac{\\sigma^2 (1-R^2)}{n}}
$$

where $\\sigma^2$ is the variance of the outcome.
- $M$ is **Bloom's Multiplier:** $M = t_{1-κ} + t_α$ where $t_{1-κ}$ is the critical value for power (1-β) and $t_α$ is the critical value for significance level (α). For two-sided tests, use $t_{1-κ/2}$.
- $R^2$ = covariate adjustment term (0 if no adjustment)
- $P$ = treatment fraction - propensity score
- $n$ = sample size

In this case, we use the fact that $y$ is binary so $\\sigma^2 = \\Pi(1-\\Pi)$.

$$
\\text{MDE} = M \\times \\sqrt{\\frac{\\Pi(1-\\Pi)(1-R^2)}{P(1-P)n}}
$$

Where $\\Pi$ = baseline rate of outcome (e.g., conversion rate)

References:
- [Bloom 1995](https://journals.sagepub.com/doi/abs/10.1177/0193841X9501900504?casa_token=FChz76X2H_oAAAAA:9J-0ktpAMAJiGzBORUEtgPavbvk7GH3eAmGUQi5M5tPG2eChKb4lHo3kWg4VEzgz1pZb5OjSx5SoP6E&casa_token=Rl5v1YO9GaIAAAAA:udb_jf59f3E-Zcu8JOpmK8e9rRWfTeK_yjpaBPAiYgMMKtEJ9SadW5f1fF0wRgvQMg1V7SRx6Ezwr3s)
- [Duflo, Glennerster, Kremer 2007](https://www.povertyactionlab.org/sites/default/files/research-paper/Using-Randomization-in-Development-Economics.pdf)
""")

left_col, right_col = st.columns(2)

# LEFT PANEL: Basic Calculator
with left_col:
    st.subheader("Basic Calculator")

    # Common parameters
    alpha = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)
    power = st.selectbox("Power (1-β)", [0.8, 0.9, 0.95], index=0)
    two_sided = st.checkbox("Two-sided test", value=False)
    treatment_fraction = st.slider("Treatment Allocation", 0.1, 0.9, 0.5, 0.05)
    r_squared = st.slider("Covariate Adjustment (R²)", 0.0, 0.8, 0.0, 0.05)
    st.markdown("---")

    calculation_type = st.radio("Calculate:", ["Sample Size", "MDE"])

    if calculation_type == "Sample Size":
        mde = st.number_input("Target MDE", 0.001, 0.5, 0.05, 0.001, format="%.3f")
        baseline_rate = st.slider("Baseline Rate", 0.01, 0.99, 0.1, 0.01)

        if st.button("Calculate Sample Size"):
            n = calculate_sample_size(
                mde,
                baseline_rate,
                treatment_fraction,
                alpha,
                power,
                two_sided,
                r_squared,
            )
            st.success(f"**Required Sample Size: {n:,}**")

            # Show breakdown
            multiplier = get_bloom_multiplier(alpha, power, two_sided)
            variance_term = baseline_rate * (1 - baseline_rate) * (1 - r_squared)
            denominator = treatment_fraction * (1 - treatment_fraction)

            st.markdown(f"""
            **Calculation:**
            - Multiplier: {multiplier:.2f}
            - Variance term: {variance_term:.4f}
            - Allocation term: {denominator:.3f}
            - Formula: n = ({multiplier:.2f})² × {variance_term:.4f} / ({mde:.3f})² × {denominator:.3f}
            """)

    elif calculation_type == "MDE":
        n = st.number_input("Sample Size", 100, 1000000, 1000, 100)
        baseline_rate = st.slider("Baseline Rate", 0.01, 0.99, 0.1, 0.01)

        if st.button("Calculate MDE"):
            mde = calculate_mde(
                n, baseline_rate, treatment_fraction, alpha, power, two_sided, r_squared
            )
            st.success(f"**Detectable MDE: {mde:.3f}**")

            # Show calculation
            multiplier = get_bloom_multiplier(alpha, power, two_sided)
            variance_term = baseline_rate * (1 - baseline_rate) * (1 - r_squared)
            se = np.sqrt(
                variance_term / (treatment_fraction * (1 - treatment_fraction) * n)
            )

            st.markdown(f"""
            **Calculation:**
            - Standard Error: {se:.4f}
            - MDE = {multiplier:.2f} × {se:.4f} = {mde:.3f}
            """)

# RIGHT PANEL: Advanced Adjustments
with right_col:
    st.subheader("Advanced Adjustments")

    tab1, tab2 = st.tabs(["Grouped Errors", "Noncompliance"])

    # GROUPED ERRORS TAB
    with tab1:
        st.markdown("""
        **Clustered randomization** (e.g., randomizing schools, not students):

        $$\\text{Design Effect} = \\sqrt{1 + (n-1)\\rho}$$

        Where $n$ = cluster size, $\\rho$ = intracluster correlation
        """)

        # Basic inputs
        individual_n = st.number_input(
            "Sample size (individual randomization)", 100, 100000, 1000, 100
        )
        cluster_size = st.number_input("Average cluster size", 2, 1000, 25, 1)
        rho = st.slider("Intracluster correlation (ρ)", 0.0, 0.5, 0.05, 0.01)

        # Calculate design effect and required clusters
        design_effect = np.sqrt(1 + (cluster_size - 1) * rho)
        effective_n = individual_n * design_effect**2
        required_clusters = int(np.ceil(effective_n / cluster_size))

        st.markdown(f"""
        **Results:**
        - Design Effect: {design_effect:.2f}
        - Effective sample size needed: {effective_n:,.0f}
        - Required clusters: {required_clusters:,}
        - Total individuals: {required_clusters * cluster_size:,}
        """)

        if design_effect > 2:
            st.warning(
                f"⚠️ Large design effect ({design_effect:.1f}x)! Consider reducing cluster size or ρ."
            )

        # Show sensitivity
        st.markdown("**Sensitivity to ρ:**")
        rho_values = np.linspace(0, 0.3, 31)
        design_effects = [np.sqrt(1 + (cluster_size - 1) * r) for r in rho_values]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rho_values, design_effects, "b-", linewidth=2)
        ax.axhline(
            y=design_effect,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Current: {design_effect:.2f}",
        )
        ax.set_xlabel("Intracluster Correlation (ρ)")
        ax.set_ylabel("Design Effect")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    # NONCOMPLIANCE TAB
    with tab2:
        st.markdown("""
        **Partial compliance** inflates required sample size:

        $$\\text{MDE}_{\\text{compliance}} = \\frac{\\text{MDE}_{\\text{perfect}}}{c - s}$$

        Where $c$ = treatment group compliance, $s$ = control group compliance
        """)

        # Compliance rates
        c = st.slider("Treatment group compliance rate", 0.1, 1.0, 0.8, 0.05)
        s = st.slider("Control group compliance rate", 0.0, 0.5, 0.05, 0.05)

        compliance_diff = c - s
        if compliance_diff <= 0:
            st.error("Treatment compliance must exceed control compliance")
        else:
            # Base calculation
            base_n = st.number_input(
                "Sample size (perfect compliance)", 100, 100000, 1000, 100
            )
            base_mde = st.number_input(
                "MDE (perfect compliance)", 0.001, 0.5, 0.05, 0.001, format="%.3f"
            )

            # Adjusted calculations
            compliance_factor = 1 / compliance_diff
            adjusted_mde = base_mde * compliance_factor
            adjusted_n = int(base_n * compliance_factor**2)

            st.markdown(f"""
            **Results:**
            - Compliance difference: {compliance_diff:.2f}
            - Inflation factor: {compliance_factor:.2f}x
            - Required sample size: {adjusted_n:,} ({compliance_factor**2:.1f}x larger)
            - Detectable MDE: {adjusted_mde:.3f} ({compliance_factor:.1f}x larger)
            """)

            if compliance_factor > 2:
                st.warning(
                    f"⚠️ Low compliance inflates requirements by {compliance_factor:.1f}x!"
                )

            # Show compliance impact
            st.markdown("**Impact of Compliance Rate:**")
            c_values = np.linspace(0.3, 1.0, 71)
            factors = [1 / (c_val - s) for c_val in c_values]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(c_values, factors, "g-", linewidth=2)
            ax.axvline(
                x=c, color="red", linestyle="--", alpha=0.7, label=f"Current c={c:.2f}"
            )
            ax.axhline(
                y=compliance_factor,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Factor={compliance_factor:.2f}",
            )
            ax.set_xlabel("Treatment Compliance Rate")
            ax.set_ylabel("Sample Size Inflation Factor")
            ax.set_ylim(1, min(10, max(factors)))
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

st.markdown("---")
