import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="A/B Test Power Calculator", layout="wide")
######################################################################
######################################################################
######################################################################

def get_bloom_multiplier(alpha, power, two_sided=False):
    """Get Bloom's multiplier M = t_{1-κ} + t_α"""
    t_power = stats.norm.ppf(power)
    if two_sided:
        t_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        t_alpha = stats.norm.ppf(1 - alpha)
    return t_power + t_alpha


def calculate_mde_individual(
    n,
    variance,
    treatment_fraction=0.5,
    alpha=0.05,
    power=0.8,
    two_sided=False,
    r_squared=0,
):
    """Calculate MDE for individual randomization"""
    multiplier = get_bloom_multiplier(alpha, power, two_sided)
    se = np.sqrt(
        (variance * (1 - r_squared))
        / (treatment_fraction * (1 - treatment_fraction) * n)
    )
    return multiplier * se


def calculate_mde_clustered(
    n_clusters,
    cluster_size,
    variance,
    rho,
    treatment_fraction=0.5,
    alpha=0.05,
    power=0.8,
    two_sided=False,
    r_squared=0,
):
    """Calculate MDE for cluster randomization using Duflo et al. formula"""
    # Degrees of freedom adjustment for cluster-level inference
    if two_sided:
        t_alpha = stats.t.ppf(1 - alpha / 2, df=n_clusters - 2)
    else:
        t_alpha = stats.t.ppf(1 - alpha, df=n_clusters - 2)

    t_power = stats.norm.ppf(power)
    multiplier = t_alpha + t_power

    # Clustered variance formula: √[ρ + (1-ρ)/n] × σ
    cluster_variance_factor = np.sqrt(rho + (1 - rho) / cluster_size)
    adjusted_variance = variance * (1 - r_squared) * cluster_variance_factor**2

    se = np.sqrt(
        adjusted_variance / (treatment_fraction * (1 - treatment_fraction) * n_clusters)
    )
    return multiplier * se


def calculate_sample_size_individual(
    mde,
    variance,
    treatment_fraction=0.5,
    alpha=0.05,
    power=0.8,
    two_sided=False,
    r_squared=0,
):
    """Calculate sample size for individual randomization"""
    multiplier = get_bloom_multiplier(alpha, power, two_sided)
    variance_term = variance * (1 - r_squared)
    denominator = treatment_fraction * (1 - treatment_fraction)

    return int(np.ceil((multiplier**2 * variance_term) / (mde**2 * denominator)))


def calculate_clusters_needed(
    mde,
    cluster_size,
    variance,
    rho,
    treatment_fraction=0.5,
    alpha=0.05,
    power=0.8,
    two_sided=False,
    r_squared=0,
):
    """Calculate number of clusters needed for cluster randomization"""
    # Use iterative approach since t-distribution depends on df
    for n_clusters in range(4, 10000):
        calculated_mde = calculate_mde_clustered(
            n_clusters,
            cluster_size,
            variance,
            rho,
            treatment_fraction,
            alpha,
            power,
            two_sided,
            r_squared,
        )
        if calculated_mde <= mde:
            return n_clusters
    return None  # No solution found


def apply_compliance_adjustment(
    mde_or_n, compliance_treatment, compliance_control, adjust_type="mde"
):
    """Apply compliance adjustment to MDE or sample size"""
    compliance_diff = compliance_treatment - compliance_control
    if compliance_diff <= 0:
        return None

    if adjust_type == "mde":
        return mde_or_n / compliance_diff
    else:  # sample size
        return int(np.ceil(mde_or_n * (1 / compliance_diff) ** 2))

##     ## #### ########
##     ##  ##       ##
##     ##  ##      ##
##     ##  ##     ##
 ##   ##   ##    ##
  ## ##    ##   ##
   ###    #### ########

st.title("Power Calculator for Two-arm Trials")

st.markdown("""
**Core MDE Formula:**

For **individual randomization:**
$$\\text{MDE} = M \\times \\sqrt{\\frac{\\sigma^2(1-R^2)}{P(1-P)n}}$$

For **cluster randomization:**
$$\\text{MDE} = M_{J-2} \\times \\sqrt{\\frac{1}{P(1-P)J}} \\times \\sqrt{\\rho + \\frac{1-\\rho}{n}} \\times \\sigma$$

Where: $M$ = multiplier, $\\sigma^2$ = outcome variance, $R^2$ = covariate adjustment, $P$ = treatment fraction, $\\rho$ = intracluster correlation, $J$ = clusters, $n$ = cluster size

**References:** [Bloom 1995](https://journals.sagepub.com/doi/abs/10.1177/0193841X9501900504), [Duflo, Glennerster, Kremer 2007](https://www.povertyactionlab.org/sites/default/files/research-paper/Using-Randomization-in-Development-Economics.pdf)
""")

left_col, right_col = st.columns(2)

# LEFT PANEL: Basic Calculator
with left_col:
    st.subheader("Power Calculator")

    # Outcome type selection
    outcome_type = st.radio("Outcome Type:", ["Binary", "Continuous"])

    # Variance specification
    if outcome_type == "Binary":
        baseline_rate = st.slider("Baseline Rate (π)", 0.01, 0.99, 0.1, 0.01)
        variance = baseline_rate * (1 - baseline_rate)
        st.write(f"Implied variance: σ² = π(1-π) = {variance:.4f}")
    else:
        variance = st.number_input("Outcome Variance (σ²)", 0.01, 100.0, 1.0, 0.01)

    # Common parameters
    alpha = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)
    power = st.selectbox("Power (1-β)", [0.8, 0.9, 0.95], index=0)
    two_sided = st.checkbox("Two-sided test", value=False)
    treatment_fraction = st.slider("Treatment Allocation (P)", 0.1, 0.9, 0.5, 0.05)
    r_squared = st.slider("Covariate Adjustment (R²)", 0.0, 0.8, 0.0, 0.05)

    # Randomization type
    randomization_type = st.radio("Randomization Level:", ["Individual", "Cluster"])

    if randomization_type == "Cluster":
        cluster_size = st.number_input("Average Cluster Size", 2, 1000, 25, 1)
        rho = st.slider("Intracluster Correlation (ρ)", 0.0, 0.5, 0.05, 0.01)

    st.markdown("---")

    calculation_type = st.radio("Calculate:", ["Sample Size", "MDE"])

    if calculation_type == "Sample Size":
        mde = st.number_input("Target MDE", 0.001, 10.0, 0.05, 0.001, format="%.3f")

        if st.button("Calculate Sample Size"):
            if randomization_type == "Individual":
                n = calculate_sample_size_individual(
                    mde,
                    variance,
                    treatment_fraction,
                    alpha,
                    power,
                    two_sided,
                    r_squared,
                )
                st.success(f"**Required Sample Size: {n:,}**")

                # Show breakdown
                multiplier = get_bloom_multiplier(alpha, power, two_sided)
                st.markdown(f"""
                **Calculation:**
                - Multiplier (M): {multiplier:.2f}
                - Variance term: {variance * (1 - r_squared):.4f}
                - Allocation term: {treatment_fraction * (1 - treatment_fraction):.3f}
                """)
            else:  # Cluster
                n_clusters = calculate_clusters_needed(
                    mde,
                    cluster_size,
                    variance,
                    rho,
                    treatment_fraction,
                    alpha,
                    power,
                    two_sided,
                    r_squared,
                )
                if n_clusters:
                    total_n = n_clusters * cluster_size
                    st.success(f"**Required Clusters: {n_clusters:,}**")
                    st.success(f"**Total Sample Size: {total_n:,}**")

                    # Show cluster-specific breakdown
                    design_effect = np.sqrt(rho + (1 - rho) / cluster_size)
                    st.markdown(f"""
                    **Cluster Calculation:**
                    - Clusters needed: {n_clusters:,}
                    - Design effect: {design_effect:.2f}
                    - Effective variance inflation: {design_effect**2:.2f}x
                    """)
                else:
                    st.error(
                        "Could not find feasible solution - try larger MDE or different parameters"
                    )

    elif calculation_type == "MDE":
        if randomization_type == "Individual":
            n = st.number_input("Sample Size", 100, 1000000, 1000, 100)

            if st.button("Calculate MDE"):
                mde = calculate_mde_individual(
                    n, variance, treatment_fraction, alpha, power, two_sided, r_squared
                )
                st.success(f"**Detectable MDE: {mde:.3f}**")

                # Show calculation
                multiplier = get_bloom_multiplier(alpha, power, two_sided)
                se = np.sqrt(
                    (variance * (1 - r_squared))
                    / (treatment_fraction * (1 - treatment_fraction) * n)
                )
                st.markdown(f"""
                **Calculation:**
                - Standard Error: {se:.4f}
                - MDE = {multiplier:.2f} × {se:.4f} = {mde:.3f}
                """)
        else:  # Cluster
            n_clusters = st.number_input("Number of Clusters", 4, 10000, 20, 1)

            if st.button("Calculate MDE"):
                mde = calculate_mde_clustered(
                    n_clusters,
                    cluster_size,
                    variance,
                    rho,
                    treatment_fraction,
                    alpha,
                    power,
                    two_sided,
                    r_squared,
                )
                total_n = n_clusters * cluster_size
                st.success(f"**Detectable MDE: {mde:.3f}**")
                st.info(
                    f"With {n_clusters} clusters of size {cluster_size} (total N = {total_n:,})"
                )

                # Show cluster calculation details
                design_effect = np.sqrt(rho + (1 - rho) / cluster_size)
                st.markdown(f"""
                **Cluster Calculation:**
                - Design effect: {design_effect:.2f}
                - Intracluster correlation: {rho:.3f}
                - Effective variance inflation: {design_effect**2:.2f}x
                """)

########     ###    ##    ## ######## ##        ######
##     ##   ## ##   ###   ## ##       ##       ##    ##
##     ##  ##   ##  ####  ## ##       ##       ##
########  ##     ## ## ## ## ######   ##        ######
##        ######### ##  #### ##       ##             ##
##        ##     ## ##   ### ##       ##       ##    ##
##        ##     ## ##    ## ######## ########  ######

# RIGHT PANEL: Compliance Adjustment
with right_col:
    st.subheader("Compliance Adjustment")

    st.markdown("""
    **Imperfect compliance** reduces effective treatment contrast:

    $$\\text{MDE}_{\\text{adjusted}} = \\frac{\\text{MDE}_{\\text{perfect}}}{c - s}$$

    Where $c$ = treatment compliance, $s$ = control compliance
    """)

    # Compliance parameters
    enable_compliance = st.checkbox("Apply compliance adjustment", value=False)

    if enable_compliance:
        c = st.slider("Treatment group compliance", 0.1, 1.0, 0.8, 0.05)
        s = st.slider("Control group compliance", 0.0, 0.5, 0.05, 0.05)

        compliance_diff = c - s
        if compliance_diff <= 0:
            st.error("Treatment compliance must exceed control compliance")
        else:
            compliance_factor = 1 / compliance_diff
            st.markdown(f"""
            **Compliance Impact:**
            - Effective contrast: {compliance_diff:.2f}
            - Inflation factor: {compliance_factor:.2f}x
            - Sample size penalty: {compliance_factor**2:.1f}x
            """)

            if compliance_factor > 2:
                st.warning(
                    f"⚠️ Low compliance inflates requirements by {compliance_factor:.1f}x!"
                )

            # Show compliance sensitivity
            c_values = np.linspace(max(s + 0.1, 0.3), 1.0, 50)
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
            ax.set_ylabel("MDE Inflation Factor")
            ax.set_ylim(1, min(10, max(factors)))
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
    else:
        compliance_factor = 1.0

    # Apply compliance adjustment to results
    if "mde" in locals() and enable_compliance and compliance_diff > 0:
        adjusted_mde = apply_compliance_adjustment(mde, c, s, "mde")
        st.markdown(f"""
        ### **Compliance-Adjusted Results:**
        - **Perfect compliance MDE:** {mde:.3f}
        - **Realistic MDE:** {adjusted_mde:.3f}
        """)

    if (
        "n" in locals()
        and enable_compliance
        and compliance_diff > 0
        and randomization_type == "Individual"
    ):
        adjusted_n = apply_compliance_adjustment(n, c, s, "sample_size")
        st.markdown(f"""
        ### **Compliance-Adjusted Results:**
        - **Perfect compliance N:** {n:,}
        - **Realistic N:** {adjusted_n:,}
        """)

    if (
        "n_clusters" in locals()
        and enable_compliance
        and compliance_diff > 0
        and randomization_type == "Cluster"
    ):
        adjusted_clusters = apply_compliance_adjustment(n_clusters, c, s, "sample_size")
        adjusted_total = adjusted_clusters * cluster_size
        st.markdown(f"""
        ### **Compliance-Adjusted Results:**
        - **Perfect compliance clusters:** {n_clusters:,}
        - **Realistic clusters:** {adjusted_clusters:,}
        - **Realistic total N:** {adjusted_total:,}
        """)

st.markdown("---")
st.markdown(
    "*Calculate statistical power for randomized controlled trials with binary or continuous outcomes*"
)
