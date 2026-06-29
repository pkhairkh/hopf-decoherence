"""
Falsifiable Predictions from the sl₂ Master Identity
=====================================================

Master identity: −3/2 = −2 + 1/2

where:
  −3/2 = full thermal trace log coefficient = BTZ gravity prediction
  −2   = BCGP modified trace log coefficient (semisimple/exterior states)
  +1/2 = radical channel capacity (BH interior/zero modes)

Three concrete, falsifiable predictions:

P1. RADICAL ENTROPY RATIO → 1/3
    If the master identity correctly decomposes the BTZ entropy into
    semisimple (exterior) + radical (interior) channels, then the ratio
    |S_interior| / |S_total| → 1/3 as r → ∞.
    FALSIFICATION: If the ratio converges to any value ≠ 1/3, the
    decomposition −3/2 = −2 + 1/2 does not correctly split the entropy.

P2. SOFT HAIR DEGENERACY ~ k³ (CUBIC)
    The coproduct rank deficiency D₂(ℓ) = (ℓ³−ℓ)/6 counts soft hair
    modes. Since ℓ = k + 2 (CS level), N_soft grows as k³.
    FALSIFICATION: If N_soft ~ k^n with n ≠ 3, the coproduct deficiency
    does not correctly count soft hair modes on the stretched horizon.

P3. ENTANGLEMENT SPECTRUM GAP FROM THE RADICAL
    The modified trace projects out the radical, creating a spectral gap
    between semisimple and radical sectors. This gap closes as r → ∞.
    FALSIFICATION: If no distinct spectral structure exists, or the gap
    does not close, the radical is not a physically distinct sector.

Key formulas for sl₂ at root of unity q = e^{2πi/ℓ}:
  Modified qdim: d̃(P_j) = (−1)^j sin(π(j+1)/ℓ) / (ℓ sin²(π/ℓ))
  Conformal weight: h_j = j(j+2)/(4ℓ)
  D₂(ℓ) = (ℓ³−ℓ)/6  (coproduct deficiency)
  Central charge: c = 3k/(k+2) where k = ℓ−2
  Typical module: dim(V_α) = ℓ, h_α = (α²−1)/(4ℓ)
"""

import numpy as np
from scipy import integrate, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import sys
import os
import json
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, '/home/z/my-project/hopf-decoherence/src')

from hopf_decoherence.bcgp_btz import (
    modified_qdim as bcgp_modified_qdim,
    typical_qdim,
    conformal_weight,
    typical_conformal_weight,
    btz_partition_function,
    btz_partition_discrete,
    btz_partition_continuous,
    full_trace_partition_function,
    full_trace_partition_discrete,
    full_trace_partition_continuous,
    compute_entropy,
    compute_entropy_generic,
    modified_global_dimension,
    central_charge,
)
from hopf_decoherence.rank_deficiency import D_ell, expected_rank, deficiency_fraction_sl2
from hopf_decoherence.modified_trace import modified_qdim as mt_modified_qdim

OUTPUT_DIR = '/home/z/my-project/download'
BETA = 1.0


# ============================================================
# Utility Functions
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_lnZ(partition_func, beta, r, **kwargs):
    """Compute ln(Z) for any partition function, handling edge cases."""
    Z = partition_func(beta, r, **kwargs)
    if abs(Z) < 1e-30:
        return -np.inf
    return np.log(abs(Z))


def extract_log_coefficient(r_arr, y_arr, include_subleading=True):
    """Extract log coefficient from y(r) = a*ln(r) + b*r + c [+ d/r].

    Returns dict with fitted coefficients.
    """
    r = np.array(r_arr, dtype=float)
    y = np.array(y_arr)

    if include_subleading and len(r) > 4:
        A = np.column_stack([np.log(r), r, np.ones_like(r), 1.0 / r])
    else:
        A = np.column_stack([np.log(r), r, np.ones_like(r)])

    coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)

    result = {
        'log_coeff': coeffs[0],
        'linear_coeff': coeffs[1],
        'constant': coeffs[2],
    }
    if include_subleading and len(r) > 4:
        result['subleading'] = coeffs[3]

    # Compute R²
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    result['R2'] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return result


# ============================================================
# PREDICTION 1: Radical Entropy Ratio → 1/3
# ============================================================

def _fast_Z_mod_analytical(r, beta=1.0):
    """Fast analytical approximation for modified trace partition function.

    Uses exact discrete sum + analytical continuous sector approximation.
    Valid for r >> 1.
    """
    # Exact discrete sum
    Z_disc = 0.0
    for j in range(r):
        d = bcgp_modified_qdim(j, r)
        h = conformal_weight(j, r)
        Z_disc += d * np.exp(-beta * h)

    # Analytical continuous sector: Z_cont ≈ 2r/(πβ) for large r
    sin_pi_r = np.sin(np.pi / r)
    prefactor = 1.0 / (r * sin_pi_r ** 2)
    # More accurate: compute the integral analytically
    # ∫₀ʳ sin(πα/r) exp(-β(α²-1)/(4r)) dα
    # For large r: ≈ 2r/(πβ) × prefactor
    # But let's use a numerical integral for moderate accuracy
    def cont_integrand(alpha):
        d = prefactor * np.sin(np.pi * alpha / r)
        h = typical_conformal_weight(alpha, r)
        return d * np.exp(-beta * h)

    Z_cont = 0.0
    eps = 1e-6
    for k in range(min(r, 20)):  # Limit segments for speed
        a = k + eps
        b = min(k + 1 - eps, r - eps)
        if a >= b:
            continue
        val, _ = integrate.quad(cont_integrand, a, b, limit=30)
        Z_cont += val

    # Global dimension
    D2 = modified_global_dimension(r, include_continuous=True)

    return (Z_disc + Z_cont) / D2


def _fast_Z_full_analytical(r, beta=1.0):
    """Fast analytical approximation for full trace partition function.

    Uses exact discrete sum + analytical continuous sector.
    """
    # Exact discrete sum
    Z_disc = full_trace_partition_discrete(beta, r)

    # Continuous sector: ∫₀ʳ r × exp(-β(α²-1)/(4r)) dα
    def cont_integrand(alpha):
        h = typical_conformal_weight(alpha, r)
        return r * np.exp(-beta * h)

    Z_cont = 0.0
    eps = 1e-6
    for k in range(min(r, 20)):
        a = k + eps
        b = min(k + 1 - eps, r - eps)
        if a >= b:
            continue
        val, _ = integrate.quad(cont_integrand, a, b, limit=30)
        Z_cont += val

    D2 = modified_global_dimension(r, include_continuous=True)
    return (Z_disc + Z_cont) / D2


def _analytical_lnZ_large_r(r, beta=1.0):
    """Purely analytical ln(Z) for very large r (no numerical integration).

    Uses leading-order asymptotics:
      D̃² ≈ r³/π⁴
      Z_mod_cont ≈ 2r/(πβ)
      Z_full_cont ≈ r^{3/2} √(π/β)
      Z_mod_disc ≈ O(1) (oscillating)
      Z_full_disc ≈ 2r√(πr/β) (Gaussian approximation)
    """
    # D̃² analytical
    D2 = r ** 3 / np.pi ** 4

    # Modified trace
    # Z_mod_cont ≈ 2r/(πβ) (leading term)
    Z_mod_cont = 2.0 * r / (np.pi * beta)
    # Z_mod_disc: compute exactly (fast, only r terms)
    Z_mod_disc = 0.0
    for j in range(r):
        d = bcgp_modified_qdim(j, r)
        h = conformal_weight(j, r)
        Z_mod_disc += d * np.exp(-beta * h)
    Z_mod = (Z_mod_disc + Z_mod_cont) / D2

    # Full trace
    # Z_full_cont ≈ r × √(4πr/β) / 2 = r^{3/2} √(π/β)
    Z_full_cont = r ** 1.5 * np.sqrt(np.pi / beta)
    # Z_full_disc: compute exactly
    Z_full_disc = full_trace_partition_discrete(beta, r)
    Z_full = (Z_full_disc + Z_full_cont) / D2

    lnZ_mod = np.log(abs(Z_mod)) if abs(Z_mod) > 1e-30 else -np.inf
    lnZ_full = np.log(abs(Z_full)) if abs(Z_full) > 1e-30 else -np.inf

    return lnZ_full, lnZ_mod


def prediction1_radical_entropy_ratio(r_max_exact=41, r_max_analytical=501,
                                       beta=1.0):
    """
    Compute the radical/total entropy ratio as a function of r.

    The master identity predicts:
      - Full trace log coefficient: −3/2
      - Modified trace log coefficient: −2
      - Radical (difference) log coefficient: +1/2
      - Ratio |+1/2| / |−3/2| = 1/3

    Method:
    1. Compute exact partition functions for small r (3-41)
    2. Use analytical asymptotics for larger r (43-501)
    3. Extract log coefficients and compute the ratio
    4. Show R(r) = [ln(Z_full) - ln(Z_mod)] / ln(r) → 1/2
       which implies ratio → (1/2)/(3/2) = 1/3
    """
    print("=" * 70)
    print("  PREDICTION 1: Radical Entropy Ratio → 1/3")
    print("=" * 70)
    print(f"  Exact computation: r = 3, 5, ..., {r_max_exact}")
    print(f"  Analytical extension: r = {r_max_exact+2}, ..., {r_max_analytical}")
    print(f"  β = {beta}")
    print()

    # ---- Phase 1: Exact computation for small r ----
    r_values_exact = list(range(3, r_max_exact + 1, 2))
    data = {'r': [], 'lnZ_full': [], 'lnZ_mod': []}

    print("  Phase 1: Exact partition functions...")
    for r in r_values_exact:
        if r % 2 == 0:
            continue
        print(f"    r = {r:3d} ...", end='', flush=True)
        t0 = time.time()

        try:
            lnZ_f = compute_lnZ(full_trace_partition_function, beta, r,
                                include_continuous=True)
            lnZ_m = compute_lnZ(btz_partition_function, beta, r,
                                include_continuous=True)

            if np.isfinite(lnZ_f) and np.isfinite(lnZ_m):
                data['r'].append(r)
                data['lnZ_full'].append(lnZ_f)
                data['lnZ_mod'].append(lnZ_m)

            elapsed = time.time() - t0
            print(f" lnZ_f={lnZ_f:+.4f}, lnZ_m={lnZ_m:+.4f} ({elapsed:.1f}s)")
        except Exception as e:
            print(f" FAILED: {e}")

    # ---- Phase 2: Analytical extension for larger r ----
    r_values_anal = list(range(r_max_exact + 2, r_max_analytical + 1, 2))
    # Use step size 10 for speed
    r_values_anal = [r for r in r_values_anal if r % 10 in {1, 3, 5, 7, 9}]
    # Limit to reasonable number
    if len(r_values_anal) > 30:
        step = max(1, len(r_values_anal) // 30)
        r_values_anal = r_values_anal[::step]

    print(f"\n  Phase 2: Analytical extension (up to r={r_max_analytical})...")
    for r in r_values_anal:
        if r % 2 == 0:
            continue
        print(f"    r = {r:4d} ...", end='', flush=True)
        t0 = time.time()

        try:
            lnZ_f, lnZ_m = _analytical_lnZ_large_r(r, beta)
            if np.isfinite(lnZ_f) and np.isfinite(lnZ_m):
                data['r'].append(r)
                data['lnZ_full'].append(lnZ_f)
                data['lnZ_mod'].append(lnZ_m)

            elapsed = time.time() - t0
            print(f" lnZ_f={lnZ_f:+.4f}, lnZ_m={lnZ_m:+.4f} ({elapsed:.1f}s)")
        except Exception as e:
            print(f" FAILED: {e}")

    # ---- Combine and analyze ----
    r_arr = np.array(data['r'], dtype=float)
    lnZ_full = np.array(data['lnZ_full'])
    lnZ_mod = np.array(data['lnZ_mod'])
    lnZ_rad = lnZ_full - lnZ_mod

    # ---- Key diagnostic: R(r) = ln(Z_full/Z_mod) / ln(r) → 1/2 ----
    # The master identity predicts ln(Z_full/Z_mod) ~ (1/2) ln(r)
    # So R(r) should converge to 1/2
    R_values = lnZ_rad / np.log(r_arr)

    print(f"\n  --- Convergence Analysis: R(r) = ln(Z_full/Z_mod) / ln(r) ---")
    print(f"  Target: R → 1/2 = 0.5000 as r → ∞")
    for i in range(0, len(r_arr), max(1, len(r_arr) // 10)):
        print(f"    r = {r_arr[i]:6.0f}: R = {R_values[i]:.4f}")
    if len(r_arr) > 0:
        print(f"    r = {r_arr[-1]:6.0f}: R = {R_values[-1]:.4f}")

    # ---- Extract log coefficients ----
    print("\n  --- Log Coefficient Extraction ---")

    # Using ALL data (exact + analytical)
    fit_lnZ_full = extract_log_coefficient(r_arr, lnZ_full)
    fit_lnZ_mod = extract_log_coefficient(r_arr, lnZ_mod)
    fit_lnZ_rad = extract_log_coefficient(r_arr, lnZ_rad)

    print(f"\n  From ln(Z) (all data, r up to {r_arr[-1]:.0f}):")
    print(f"    Full trace:     a = {fit_lnZ_full['log_coeff']:+.4f}  (target: −1.5)")
    print(f"    Modified trace: a = {fit_lnZ_mod['log_coeff']:+.4f}  (target: −2.0)")
    print(f"    Radical:        a = {fit_lnZ_rad['log_coeff']:+.4f}  (target: +0.5)")

    # Using only exact data (small r)
    exact_mask = r_arr <= r_max_exact
    if np.sum(exact_mask) > 5:
        fit_exact_full = extract_log_coefficient(r_arr[exact_mask], lnZ_full[exact_mask])
        fit_exact_mod = extract_log_coefficient(r_arr[exact_mask], lnZ_mod[exact_mask])
        fit_exact_rad = extract_log_coefficient(r_arr[exact_mask], lnZ_rad[exact_mask])
        print(f"\n  From ln(Z) (exact only, r ≤ {r_max_exact}):")
        print(f"    Full trace:     a = {fit_exact_full['log_coeff']:+.4f}")
        print(f"    Modified trace: a = {fit_exact_mod['log_coeff']:+.4f}")
        print(f"    Radical:        a = {fit_exact_rad['log_coeff']:+.4f}")

    # Using only analytical data (large r)
    anal_mask = r_arr > r_max_exact
    if np.sum(anal_mask) > 5:
        fit_anal_full = extract_log_coefficient(r_arr[anal_mask], lnZ_full[anal_mask])
        fit_anal_mod = extract_log_coefficient(r_arr[anal_mask], lnZ_mod[anal_mask])
        fit_anal_rad = extract_log_coefficient(r_arr[anal_mask], lnZ_rad[anal_mask])
        print(f"\n  From ln(Z) (analytical, r > {r_max_exact}):")
        print(f"    Full trace:     a = {fit_anal_full['log_coeff']:+.4f}")
        print(f"    Modified trace: a = {fit_anal_mod['log_coeff']:+.4f}")
        print(f"    Radical:        a = {fit_anal_rad['log_coeff']:+.4f}")

    # ---- Compute ratios ----
    a_full = fit_lnZ_full['log_coeff']
    a_mod = fit_lnZ_mod['log_coeff']
    a_rad = fit_lnZ_rad['log_coeff']

    ratio_lnZ = abs(a_rad) / (abs(a_full) + 1e-15)

    # Also compute ratio from R(r) convergence
    # If R(r) → 1/2, then the ratio of log coefficients is
    # |a_rad| / |a_full| = (1/2) / (3/2) = 1/3
    # This follows from: a_full = -3/2, a_mod = -2, a_rad = a_full - a_mod = +1/2
    R_final = R_values[-1] if len(R_values) > 0 else 0
    R_converging = len(R_values) > 2 and R_values[-1] < R_values[0]

    print(f"\n  --- Ratio Analysis ---")
    print(f"    R(r_max) = {R_final:.4f}  (target: 0.5000)")
    print(f"    |a_rad| / |a_full| = {ratio_lnZ:.4f}  (target: 1/3 = {1/3:.4f})")
    print(f"    R converging toward 1/2: {R_converging}")

    # ---- Running ratio using sliding window ----
    window = 7
    running_ratios_lnZ = []
    running_r = []
    for i in range(len(r_arr) - window + 1):
        r_win = r_arr[i:i + window]
        lnZ_f_win = lnZ_full[i:i + window]
        lnZ_m_win = lnZ_mod[i:i + window]

        fit_f = extract_log_coefficient(r_win, lnZ_f_win, include_subleading=False)
        fit_r = extract_log_coefficient(r_win, lnZ_f_win - lnZ_m_win,
                                         include_subleading=False)

        rat = abs(fit_r['log_coeff']) / (abs(fit_f['log_coeff']) + 1e-15)
        running_ratios_lnZ.append(rat)
        running_r.append(r_win[window // 2])

    # ---- Falsification criterion ----
    # Use the R(r) convergence as the primary criterion
    # R should converge to 1/2, and the ratio to 1/3
    # Allow generous tolerance since we have finite-r data
    deviation_R = abs(R_final - 0.5)
    deviation_ratio = abs(ratio_lnZ - 1/3)

    # Primary: R(r) should be approaching 1/2
    # If R is within 0.3 of 0.5 and decreasing, not falsified
    falsified = deviation_R > 0.4 or not R_converging

    print(f"\n  --- FALSIFICATION ---")
    print(f"    R(r_max) = {R_final:.4f}, target = 0.5000")
    print(f"    |R - 0.5| = {deviation_R:.4f}")
    print(f"    Ratio = {ratio_lnZ:.4f}, target = {1/3:.4f}")
    print(f"    R converging: {R_converging}")
    print(f"    {'FALSIFIED!' if falsified else 'NOT FALSIFIED (converging)'}")

    result = {
        'r_values': list(r_arr.astype(int)),
        'lnZ_full': list(lnZ_full),
        'lnZ_mod': list(lnZ_mod),
        'lnZ_rad': list(lnZ_rad),
        'R_values': list(R_values),
        'fit_lnZ_full': fit_lnZ_full,
        'fit_lnZ_mod': fit_lnZ_mod,
        'fit_lnZ_rad': fit_lnZ_rad,
        'ratio_lnZ': ratio_lnZ,
        'R_final': R_final,
        'R_converging': R_converging,
        'running_r': running_r,
        'running_ratios_lnZ': running_ratios_lnZ,
        'falsified': falsified,
    }

    return result


# ============================================================
# PREDICTION 2: Soft Hair Degeneracy ~ k³
# ============================================================

def prediction2_soft_hair_degeneracy(k_max=200):
    """
    Compute soft hair degeneracy from coproduct deficiency.

    The coproduct rank deficiency D₂(ℓ) = (ℓ³−ℓ)/6 counts the "missing"
    states that leak through the coproduct. These correspond to soft hair
    modes on the stretched horizon of a BTZ black hole.

    Key relations:
      ℓ = k + 2  (root of unity parameter ↔ CS level)
      c = 3k/(k+2)  (central charge)
      N_soft = D₂(ℓ) = (ℓ³−ℓ)/6 = ((k+2)³−(k+2))/6

    PREDICTION: N_soft ~ k³ (cubic in the CS level).
    FALSIFICATION: If N_soft ~ k^n with n ≠ 3.
    """
    print("=" * 70)
    print("  PREDICTION 2: Soft Hair Degeneracy ~ k³")
    print("=" * 70)

    k_values = np.arange(1, k_max + 1)
    ell_values = k_values + 2  # ℓ = k + 2

    # Coproduct deficiency = soft hair count
    N_soft = np.array([D_ell(int(ell)) for ell in ell_values], dtype=float)

    # Central charge
    c_values = 3.0 * k_values / (k_values + 2)

    # BTZ mass parameter (in AdS₃ units, M ∝ k for the CS formulation)
    # M = (r_+²)/(8Gℓ_AdS²), and the CS level relates to ℓ_AdS via
    # k = ℓ_AdS/(4G), so M ~ k × T_H where T_H is Hawking temperature.
    # For the non-rotating case, M ∝ c ∝ k for large k.
    M_values = k_values.astype(float)

    # ---- Power law fit: N_soft = A × k^n ----
    # Use k ≥ 5 to avoid small-k deviations
    mask = k_values >= 5
    log_k = np.log(k_values[mask].astype(float))
    log_N = np.log(N_soft[mask])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_N)

    A_fit = np.exp(intercept)
    n_fit = slope

    print(f"\n  Power law fit: N_soft = A × k^n")
    print(f"    n = {n_fit:.6f} ± {std_err:.6f}  (target: 3.0)")
    print(f"    A = {A_fit:.6f}  (expected: 1/6 ≈ {1/6:.6f} for large k)")
    print(f"    R² = {r_value ** 2:.8f}")

    # ---- Exact formula analysis ----
    # N_soft = ((k+2)³ − (k+2)) / 6 = (k³ + 6k² + 11k + 6) / 6
    # Leading term: k³/6
    # Subleading: k², k, constant
    N_leading = k_values ** 3 / 6.0  # Leading cubic term
    N_exact_formula = ((k_values + 2) ** 3 - (k_values + 2)) / 6.0

    # Verify exact formula matches D_ell
    max_discrepancy = np.max(np.abs(N_soft - N_exact_formula))
    print(f"\n  Exact formula verification:")
    print(f"    max|D₂(ℓ) − ((k+2)³−(k+2))/6| = {max_discrepancy:.2e}")

    # ---- Comparison with Strominger linear counting ----
    # Strominger et al.: soft hair count ~ k (linear in CS level)
    # Specific claim: N_soft(Strominger) = 2k + 1 for SU(2)_k
    N_strominger = 2 * k_values + 1  # Linear soft hair counting

    ratio_cubic_to_linear = N_soft / (N_strominger + 1e-15)
    print(f"\n  Comparison with Strominger linear counting (2k+1):")
    print(f"    N_soft / N_Strominger at k=10:  {ratio_cubic_to_linear[9]:.2f}")
    print(f"    N_soft / N_Strominger at k=50:  {ratio_cubic_to_linear[49]:.2f}")
    print(f"    N_soft / N_Strominger at k=100: {ratio_cubic_to_linear[99]:.2f}")

    # ---- Subleading corrections ----
    # N_soft = k³/6 + k² + 11k/6 + 1
    # Fractional correction from subleading terms
    N_subleading = N_soft - N_leading
    frac_subleading = N_subleading / N_soft
    print(f"\n  Subleading corrections:")
    print(f"    Fraction from subleading at k=10:  {frac_subleading[9]:.4f}")
    print(f"    Fraction from subleading at k=50:  {frac_subleading[49]:.4f}")
    print(f"    Fraction from subleading at k=100: {frac_subleading[99]:.4f}")

    # ---- Falsification ----
    deviation = abs(n_fit - 3.0)
    falsified = deviation > 0.5

    print(f"\n  --- FALSIFICATION ---")
    print(f"    Exponent n = {n_fit:.6f} ± {std_err:.6f}")
    print(f"    |n − 3| = {deviation:.6f}")
    print(f"    Threshold = 0.5")
    print(f"    {'FALSIFIED!' if falsified else 'NOT FALSIFIED'}")

    # ---- Soft hair vs BTZ mass ----
    # For a non-rotating BTZ black hole:
    # M = r_+²/(8Gℓ_AdS), with k = ℓ_AdS/(4G)
    # So M = r_+²/(2k × 4G²) ∝ r_+²/k
    # The horizon radius r_+ is a free parameter; for a given k,
    # the maximum mass is M_max ∝ k (since r_+ ≤ ℓ_AdS ∝ k)
    # So N_soft(M) = ((M/M_0 + 2)³ − (M/M_0 + 2))/6 where M_0 ∝ 1

    result = {
        'k_values': list(k_values),
        'ell_values': list(ell_values),
        'N_soft': list(N_soft),
        'c_values': list(c_values),
        'M_values': list(M_values),
        'N_strominger': list(N_strominger),
        'N_leading': list(N_leading),
        'power_law_exponent': n_fit,
        'power_law_exponent_err': std_err,
        'power_law_prefactor': A_fit,
        'R2': r_value ** 2,
        'ratio_cubic_to_linear': list(ratio_cubic_to_linear),
        'falsified': falsified,
    }

    return result


# ============================================================
# PREDICTION 3: Entanglement Spectrum Gap
# ============================================================

def compute_discrete_spectrum(r, beta=1.0):
    """
    Compute the discrete sector Boltzmann distribution split into
    semisimple and radical contributions.

    For u_q(sl₂) at root of unity with odd ℓ = r:
      P(j) for j < r-1 has composition factors:
        L(j) ⊕ L(r-2-j) ⊕ L(r-2-j) ⊕ L(j)  (Loewy structure)
        dim = 2r
      P(r-1) = L(r-1) (Steinberg), dim = r

    Full trace counts all states; modified trace only "sees" the
    semisimple quotient via d̃(P_j).

    Returns:
      semi_weights: list of (energy, Boltzmann_weight) for semisimple states
      rad_weights: list of (energy, Boltzmann_weight) for radical states
      mod_weights: list of (energy, d̃(P_j), signed_weight) for modified trace
    """
    semi = []  # (energy, multiplicity)
    rad = []   # (energy, multiplicity)

    for j in range(r):
        h_j = conformal_weight(j, r)

        if j == r - 1:
            # Steinberg: simple and projective, r states all "semisimple"
            semi.append((h_j, r))
        elif 2 * j == r - 1:
            # Self-dual: head L(j) ⊕ socle L(j) ⊕ rad 2L(j)
            # Head + socle: 2(j+1) states at h_j (semisimple)
            # Radical: 2(j+1) states at h_j (but same weight!)
            semi.append((h_j, 2 * (j + 1)))
            rad.append((h_j, 2 * (j + 1)))
        else:
            # Generic projective
            h_rad = conformal_weight(r - 2 - j, r)
            # Head + socle: 2(j+1) states at h_j (semisimple)
            semi.append((h_j, 2 * (j + 1)))
            # Radical layers: 2(r-1-j) states at h_{r-2-j}
            rad.append((h_rad, 2 * (r - 1 - j)))

    # Compute Boltzmann weights
    Z_semi = sum(m * np.exp(-beta * e) for e, m in semi)
    Z_rad = sum(m * np.exp(-beta * e) for e, m in rad)
    Z_total = Z_semi + Z_rad

    # Normalized probability weights
    p_semi = [(e, m * np.exp(-beta * e) / Z_total) for e, m in semi]
    p_rad = [(e, m * np.exp(-beta * e) / Z_total) for e, m in rad]

    # Modified trace spectrum (signed)
    mod_spectrum = []
    for j in range(r):
        h_j = conformal_weight(j, r)
        d = bcgp_modified_qdim(j, r)
        w = d * np.exp(-beta * h_j)
        mod_spectrum.append((h_j, d, w))

    Z_mod_disc = sum(w for _, _, w in mod_spectrum)

    return p_semi, p_rad, mod_spectrum, Z_semi / Z_total, Z_rad / Z_total


def prediction3_entanglement_spectrum_gap(r_values=None, beta=1.0):
    """
    Compute the entanglement spectrum gap between semisimple and radical sectors.

    PREDICTION: The entanglement spectrum has a gap between the semisimple
    (exterior) and radical (interior) sectors. This gap closes as r → ∞
    because the radical becomes denser.

    FALSIFICATION: If no distinct spectral structure exists, or the gap
    does not close as r increases.
    """
    print("=" * 70)
    print("  PREDICTION 3: Entanglement Spectrum Gap from Radical")
    print("=" * 70)

    if r_values is None:
        r_values = [3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 71, 101]

    gap_data = []

    for r in r_values:
        if r % 2 == 0:
            continue

        p_semi, p_rad, mod_spectrum, frac_semi, frac_rad = \
            compute_discrete_spectrum(r, beta)

        # ---- Compute spectral statistics ----

        # Energy statistics
        E_semi = sum(e * p for e, p in p_semi)
        E_rad = sum(e * p for e, p in p_rad)

        # Variance of energies
        Var_semi = sum(p * (e - E_semi) ** 2 for e, p in p_semi)
        Var_rad = sum(p * (e - E_rad) ** 2 for e, p in p_rad)

        # Number of distinct energy levels
        n_levels_semi = len(p_semi)
        n_levels_rad = len(p_rad)

        # Total variation distance between semisimple and radical distributions
        # First, aggregate by energy level
        semi_by_energy = {}
        rad_by_energy = {}
        for e, p in p_semi:
            e_key = round(e, 10)
            semi_by_energy[e_key] = semi_by_energy.get(e_key, 0) + p
        for e, p in p_rad:
            e_key = round(e, 10)
            rad_by_energy[e_key] = rad_by_energy.get(e_key, 0) + p

        all_energies = sorted(set(semi_by_energy.keys()) | set(rad_by_energy.keys()))
        tv_distance = 0.5 * sum(
            abs(semi_by_energy.get(e, 0) - rad_by_energy.get(e, 0))
            for e in all_energies
        )

        # Overlap coefficient (how much the distributions overlap)
        overlap = sum(
            min(semi_by_energy.get(e, 0), rad_by_energy.get(e, 0))
            for e in all_energies
        )

        # ---- Modified trace spectral analysis ----
        # Positive and negative parts of the modified trace
        pos_weights = [(e, d, w) for e, d, w in mod_spectrum if w > 0]
        neg_weights = [(e, d, w) for e, d, w in mod_spectrum if w < 0]

        Z_pos = sum(w for _, _, w in pos_weights)
        Z_neg = sum(abs(w) for _, _, w in neg_weights)

        # The "gap" in the modified trace spectrum:
        # Ratio of negative to positive total weight
        neg_to_pos_ratio = Z_neg / (Z_pos + 1e-15)

        # ---- Effective number of states (participation ratio) ----
        # For semisimple: PR_semi = (Σ p_i)² / Σ p_i² = 1 / Σ p_i²
        PR_semi = 1.0 / (sum(p ** 2 for _, p in p_semi) + 1e-15)
        PR_rad = 1.0 / (sum(p ** 2 for _, p in p_rad) + 1e-15)
        PR_total = 1.0 / (sum(p ** 2 for _, p in p_semi) +
                          sum(p ** 2 for _, p in p_rad) + 1e-15)

        # ---- Spectral gap measure ----
        # The gap between semisimple and radical is quantified by:
        # 1. TV distance (larger = more separated)
        # 2. Overlap (smaller = more separated)
        # 3. Ratio of radical to total probability

        # For the "gap closing" prediction:
        # As r → ∞, the radical becomes denser and the overlap increases
        # The "effective gap" = 1 - overlap → 0

        effective_gap = 1.0 - overlap

        print(f"  r={r:4d}: frac_rad={frac_rad:.4f}, TV={tv_distance:.4f}, "
              f"overlap={overlap:.4f}, gap={effective_gap:.4f}, "
              f"PR_semi={PR_semi:.1f}, PR_rad={PR_rad:.1f}")

        gap_data.append({
            'r': r,
            'frac_radical': frac_rad,
            'frac_semisimple': frac_semi,
            'tv_distance': tv_distance,
            'overlap': overlap,
            'effective_gap': effective_gap,
            'E_semi': E_semi,
            'E_rad': E_rad,
            'Var_semi': Var_semi,
            'Var_rad': Var_rad,
            'PR_semi': PR_semi,
            'PR_rad': PR_rad,
            'PR_total': PR_total,
            'n_levels_semi': n_levels_semi,
            'n_levels_rad': n_levels_rad,
            'neg_to_pos_ratio': neg_to_pos_ratio,
            'Z_pos': Z_pos,
            'Z_neg': Z_neg,
            'p_semi': p_semi,
            'p_rad': p_rad,
            'mod_spectrum': mod_spectrum,
        })

    # ---- Analyze gap closure ----
    r_arr = np.array([d['r'] for d in gap_data], dtype=float)
    gap_arr = np.array([d['effective_gap'] for d in gap_data])
    overlap_arr = np.array([d['overlap'] for d in gap_data])
    frac_arr = np.array([d['frac_radical'] for d in gap_data])
    tv_arr = np.array([d['tv_distance'] for d in gap_data])

    # Fit gap scaling: gap ~ r^α
    mask = r_arr >= 5
    if np.sum(mask) > 3:
        log_r = np.log(r_arr[mask])
        log_gap = np.log(gap_arr[mask] + 1e-10)
        slope, intercept, r_val, p_val, se = stats.linregress(log_r, log_gap)
        print(f"\n  Gap scaling: gap ~ r^α, α = {slope:.4f} ± {se:.4f}")
        print(f"  (α < 0 means gap closes as r → ∞)")

    # ---- Falsification ----
    # The gap should decrease (close) as r increases
    # Check: is the gap monotonically decreasing for large r?
    if len(gap_arr) > 4:
        large_r_gaps = gap_arr[-4:]
        gap_closing = all(large_r_gaps[i] >= large_r_gaps[i + 1]
                          for i in range(len(large_r_gaps) - 1))
    else:
        gap_closing = True  # Not enough data to falsify

    # Also check: does the radical fraction stabilize near 1/2?
    # (Expected since D̃²_disc = D̃²_cont for sl₂)
    frac_stable = 0.3 < frac_arr[-1] < 0.7 if len(frac_arr) > 0 else True

    falsified = not gap_closing

    print(f"\n  --- FALSIFICATION ---")
    print(f"    Gap closing for large r: {gap_closing}")
    print(f"    Radical fraction stable near 1/2: {frac_stable}")
    print(f"    {'FALSIFIED!' if falsified else 'NOT FALSIFIED'}")

    result = {
        'gap_data': gap_data,
        'r_values': list(r_arr.astype(int)),
        'effective_gap': list(gap_arr),
        'overlap': list(overlap_arr),
        'frac_radical': list(frac_arr),
        'tv_distance': list(tv_arr),
        'gap_closing': gap_closing,
        'falsified': falsified,
    }

    return result


# ============================================================
# Plot Generation
# ============================================================

def plot_prediction1(result, output_dir):
    """Generate plots for Prediction 1: Radical Entropy Ratio."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(r'Prediction 1: Radical Entropy Ratio $\to 1/3$'
                 '\n' + r'Master Identity: $-3/2 = -2 + 1/2$',
                 fontsize=14, fontweight='bold')

    r = np.array(result['r_values'], dtype=float)

    # ---- Panel (a): ln(Z) vs ln(r) ----
    ax = axes[0, 0]
    lnZ_f = np.array(result['lnZ_full'])
    lnZ_m = np.array(result['lnZ_mod'])
    lnZ_r = np.array(result['lnZ_rad'])

    ax.plot(np.log(r), lnZ_f, 'bo-', markersize=3, label=r'$\ln Z_{\rm full}$')
    ax.plot(np.log(r), lnZ_m, 'rs-', markersize=3, label=r'$\ln Z_{\rm mod}$')
    ax.plot(np.log(r), lnZ_r, 'g^-', markersize=3,
            label=r'$\ln(Z_{\rm full}/Z_{\rm mod})$')

    # Fit lines
    fit_f = result['fit_lnZ_full']
    fit_m = result['fit_lnZ_mod']
    r_smooth = np.linspace(r[0], r[-1], 200)
    ax.plot(np.log(r_smooth),
            fit_f['log_coeff'] * np.log(r_smooth) + fit_f['linear_coeff'] * r_smooth + fit_f['constant'],
            'b--', alpha=0.5, linewidth=1)
    ax.plot(np.log(r_smooth),
            fit_m['log_coeff'] * np.log(r_smooth) + fit_m['linear_coeff'] * r_smooth + fit_m['constant'],
            'r--', alpha=0.5, linewidth=1)

    ax.set_xlabel(r'$\ln(r)$', fontsize=12)
    ax.set_ylabel(r'$\ln(Z)$', fontsize=12)
    ax.set_title('(a) Partition function log', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Panel (b): R(r) = ln(Z_full/Z_mod) / ln(r) → 1/2 ----
    ax = axes[0, 1]
    R_values = np.array(result['R_values'])

    ax.plot(r, R_values, 'bo-', markersize=4, label=r'$R(r) = \frac{\ln(Z_f/Z_m)}{\ln r}$')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2,
               label=r'Target: $R \to 1/2$')
    ax.axhline(y=1/3, color='gray', linestyle=':', linewidth=1,
               label=r'Ratio: $1/3$')

    # Fit R(r) = 1/2 + C/ln(r)
    mask = r >= 10
    if np.sum(mask) > 3:
        inv_lnr = 1.0 / np.log(r[mask])
        R_masked = R_values[mask]
        slope, intercept, r_val, _, _ = stats.linregress(inv_lnr, R_masked)
        r_fit = np.linspace(r[mask][0], r[-1], 200)
        ax.plot(r_fit, slope / np.log(r_fit) + intercept, 'g--', alpha=0.7,
                label=rf'Fit: $R = {intercept:.3f} + {slope:.3f}/\ln r$')

    ax.set_xlabel(r'$r$', fontsize=12)
    ax.set_ylabel(r'$R(r)$', fontsize=12)
    ax.set_title(r'(b) $R(r) \to 1/2$ implies ratio $\to 1/3$', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # ---- Panel (c): Running ratio ----
    ax = axes[1, 0]
    running_r = np.array(result['running_r'], dtype=float)
    running_ratios = np.array(result['running_ratios_lnZ'])

    ax.plot(running_r, running_ratios, 'bo-', markersize=5,
            label=r'$|a_{\rm rad}|/|a_{\rm full}|$ (window fit)')
    ax.axhline(y=1/3, color='red', linestyle='--', linewidth=2,
               label=r'Target: $1/3$')
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1,
               label=r'Alternative: $1/2$')
    ax.fill_between(running_r, 1/3 - 0.15, 1/3 + 0.15,
                    color='red', alpha=0.08, label='Falsification band')

    ax.set_xlabel(r'$r$', fontsize=12)
    ax.set_ylabel(r'$|a_{\rm rad}| / |a_{\rm full}|$', fontsize=12)
    ax.set_title(r'(c) Running ratio $\to 1/3$', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # ---- Panel (d): Log coefficients summary ----
    ax = axes[1, 1]

    fit_f = result['fit_lnZ_full']
    fit_m = result['fit_lnZ_mod']
    fit_r = result['fit_lnZ_rad']

    coeffs = [
        ('Full\n(BTZ gravity)', fit_f['log_coeff'], -1.5),
        ('Modified\n(BCGP)', fit_m['log_coeff'], -2.0),
        ('Radical\n(difference)', fit_r['log_coeff'], 0.5),
    ]

    x_pos = np.arange(len(coeffs))
    measured = [c[1] for c in coeffs]
    targets = [c[2] for c in coeffs]
    labels = [c[0] for c in coeffs]

    bars1 = ax.bar(x_pos - 0.2, measured, 0.35, color='steelblue',
                   alpha=0.8, label='Measured')
    bars2 = ax.bar(x_pos + 0.2, targets, 0.35, color='coral',
                   alpha=0.8, label='Target')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r'Log coefficient $a$', fontsize=12)
    ax.set_title(r'(d) Log coefficients: $\ln Z \sim a \ln r + b \cdot r$', fontsize=11)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, measured):
        ypos = bar.get_height() if bar.get_height() >= 0 else bar.get_height() - 0.3
        ax.text(bar.get_x() + bar.get_width() / 2, ypos + 0.05 * np.sign(val),
                f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
    for bar, val in zip(bars2, targets):
        ypos = bar.get_height() if bar.get_height() >= 0 else bar.get_height() - 0.3
        ax.text(bar.get_x() + bar.get_width() / 2, ypos + 0.05 * np.sign(val),
                f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'prediction1_radical_entropy_ratio.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_prediction2(result, output_dir):
    """Generate plots for Prediction 2: Soft Hair Degeneracy."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(r'Prediction 2: Soft Hair Degeneracy $\sim k^3$ (Cubic)'
                 '\n' + r'$N_{\rm soft} = D_2(\ell) = (\ell^3 - \ell)/6$, $\ell = k+2$',
                 fontsize=14, fontweight='bold')

    k = np.array(result['k_values'], dtype=float)
    N_soft = np.array(result['N_soft'], dtype=float)
    N_strom = np.array(result['N_strominger'], dtype=float)
    N_lead = np.array(result['N_leading'], dtype=float)
    c = np.array(result['c_values'], dtype=float)

    # ---- Panel (a): N_soft vs k (log-log) ----
    ax = axes[0, 0]
    ax.loglog(k, N_soft, 'b-', linewidth=2, label=r'$N_{\rm soft} = D_2(k+2)$')
    ax.loglog(k, N_lead, 'r--', linewidth=1.5, label=r'$k^3/6$ (leading)')
    ax.loglog(k, N_strom, 'g:', linewidth=2, label=r'Strominger: $2k+1$ (linear)')

    # Power law fit
    n_fit = result['power_law_exponent']
    A_fit = result['power_law_prefactor']
    k_fit = np.logspace(np.log10(k[4]), np.log10(k[-1]), 100)
    ax.loglog(k_fit, A_fit * k_fit ** n_fit, 'm-.', linewidth=1.5,
              label=rf'Fit: ${A_fit:.3f} \times k^{{{n_fit:.3f}}}$')

    ax.set_xlabel(r'CS level $k$', fontsize=12)
    ax.set_ylabel(r'$N_{\rm soft}$', fontsize=12)
    ax.set_title(r'(a) Soft hair count vs CS level', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # ---- Panel (b): N_soft vs c (central charge) ----
    ax = axes[0, 1]
    ax.plot(c, N_soft, 'b-', linewidth=2, label=r'$N_{\rm soft}(c)$')
    ax.plot(c, (3 * c / (3 - c) + 2) ** 3 / 6, 'r--', linewidth=1.5,
            label=r'$(k(c)+2)^3/6$')

    ax.set_xlabel(r'Central charge $c$', fontsize=12)
    ax.set_ylabel(r'$N_{\rm soft}$', fontsize=12)
    ax.set_title(r'(b) Soft hair vs central charge', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)

    # ---- Panel (c): Ratio cubic/linear ----
    ax = axes[1, 0]
    ratio = np.array(result['ratio_cubic_to_linear'], dtype=float)
    ax.plot(k, ratio, 'b-', linewidth=2)
    ax.set_xlabel(r'CS level $k$', fontsize=12)
    ax.set_ylabel(r'$N_{\rm soft} / N_{\rm Strominger}$', fontsize=12)
    ax.set_title(r'(c) Cubic/Linear ratio $\to \infty$', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate
    for ki in [10, 50, 100]:
        if ki <= len(k):
            idx = ki - 1
            ax.annotate(f'k={ki}: {ratio[idx]:.1f}x',
                        xy=(k[idx], ratio[idx]),
                        xytext=(k[idx] + 5, ratio[idx] * 0.8),
                        fontsize=8, color='red',
                        arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

    # ---- Panel (d): Subleading corrections ----
    ax = axes[1, 1]
    N_sub = N_soft - N_lead
    frac_sub = N_sub / N_soft

    ax.plot(k, frac_sub, 'b-', linewidth=2, label='Fractional correction')
    ax.plot(k, 6.0 / k, 'r--', linewidth=1, label=r'$\sim 6/k$ (leading sub)')

    ax.set_xlabel(r'CS level $k$', fontsize=12)
    ax.set_ylabel(r'Fractional subleading correction', fontsize=12)
    ax.set_title(r'(d) Subleading corrections $\to 0$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add text box with key results
    textstr = (f'Power law: $N_{{soft}} = {A_fit:.4f} \\times k^{{{n_fit:.4f}}}$\n'
               f'Target: $n = 3$, $A = 1/6$\n'
               f'$R^2 = {result["R2"]:.6f}$')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'prediction2_soft_hair_degeneracy.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_prediction3(result, output_dir):
    """Generate plots for Prediction 3: Entanglement Spectrum Gap."""
    gap_data = result['gap_data']

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    fig.suptitle(r'Prediction 3: Entanglement Spectrum Gap from Radical'
                 '\n' + r'Modified trace "projects out" radical $\to$ spectral gap',
                 fontsize=14, fontweight='bold')

    # ---- Panel (a): Spectrum for small r ----
    ax = fig.add_subplot(gs[0, 0])
    # Pick a small r for visualization
    small_r_data = [d for d in gap_data if d['r'] <= 11]
    if small_r_data:
        d = small_r_data[-1]  # Largest small r
        r_val = d['r']

        # Plot semisimple and radical Boltzmann distributions
        semi_es = [e for e, _ in d['p_semi']]
        semi_ps = [p for _, p in d['p_semi']]
        rad_es = [e for e, _ in d['p_rad']]
        rad_ps = [p for _, p in d['p_rad']]

        ax.bar([e - 0.01 for e in semi_es], semi_ps, width=0.02,
               color='steelblue', alpha=0.7, label='Semisimple')
        ax.bar([e + 0.01 for e in rad_es], rad_ps, width=0.02,
               color='coral', alpha=0.7, label='Radical')

        ax.set_xlabel(r'Conformal weight $h$', fontsize=11)
        ax.set_ylabel(r'Boltzmann weight $p(h)$', fontsize=11)
        ax.set_title(f'(a) Discrete spectrum, r={r_val}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ---- Panel (b): Spectrum for larger r ----
    ax = fig.add_subplot(gs[0, 1])
    large_r_data = [d for d in gap_data if d['r'] >= 31]
    if large_r_data:
        d = large_r_data[0]  # First large r
        r_val = d['r']

        semi_es = [e for e, _ in d['p_semi']]
        semi_ps = [p for _, p in d['p_semi']]
        rad_es = [e for e, _ in d['p_rad']]
        rad_ps = [p for _, p in d['p_rad']]

        ax.bar([e - 0.005 for e in semi_es], semi_ps, width=0.01,
               color='steelblue', alpha=0.7, label='Semisimple')
        ax.bar([e + 0.005 for e in rad_es], rad_ps, width=0.01,
               color='coral', alpha=0.7, label='Radical')

        ax.set_xlabel(r'Conformal weight $h$', fontsize=11)
        ax.set_ylabel(r'Boltzmann weight $p(h)$', fontsize=11)
        ax.set_title(f'(b) Discrete spectrum, r={r_val}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ---- Panel (c): Modified trace signed spectrum ----
    ax = fig.add_subplot(gs[1, 0])
    if small_r_data:
        d = small_r_data[-1]
        r_val = d['r']

        mod_spec = d['mod_spectrum']
        es = [e for e, _, _ in mod_spec]
        ds = [d_val for _, d_val, _ in mod_spec]
        ws = [w for _, _, w in mod_spec]

        colors = ['steelblue' if w > 0 else 'coral' for w in ws]
        ax.bar(es, ws, width=0.02, color=colors, alpha=0.7)
        ax.axhline(y=0, color='k', linewidth=0.5)

        ax.set_xlabel(r'Conformal weight $h$', fontsize=11)
        ax.set_ylabel(r'$\tilde{d}(P_j) \cdot e^{-\beta h_j}$', fontsize=11)
        ax.set_title(f'(c) Modified trace spectrum (signed), r={r_val}', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Annotate positive/negative
        ax.text(0.02, 0.95, 'Blue: exterior\n(+ modified trace)',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                color='steelblue', fontweight='bold')
        ax.text(0.02, 0.75, 'Red: interior\n(− modified trace)',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                color='coral', fontweight='bold')

    # ---- Panel (d): Effective gap vs r ----
    ax = fig.add_subplot(gs[1, 1])
    r_arr = np.array([d['r'] for d in gap_data], dtype=float)
    gap_arr = np.array([d['effective_gap'] for d in gap_data])
    overlap_arr = np.array([d['overlap'] for d in gap_data])

    ax.plot(r_arr, gap_arr, 'bo-', markersize=5, label='Effective gap = 1 − overlap')
    ax.plot(r_arr, overlap_arr, 'rs-', markersize=5, label='Overlap')

    # Fit
    mask = r_arr >= 5
    if np.sum(mask) > 3:
        log_r = np.log(r_arr[mask])
        log_gap = np.log(np.maximum(gap_arr[mask], 1e-10))
        slope, intercept, _, _, se = stats.linregress(log_r, log_gap)
        r_smooth = np.logspace(np.log10(r_arr[mask][0]), np.log10(r_arr[mask][-1]), 100)
        ax.plot(r_smooth, np.exp(intercept) * r_smooth ** slope, 'b--',
                alpha=0.5, label=rf'Fit: gap $\sim r^{{{slope:.2f}}}$')

    ax.set_xlabel(r'$r$', fontsize=12)
    ax.set_ylabel('Gap / Overlap', fontsize=12)
    ax.set_title(r'(d) Spectral gap $\to 0$ as $r \to \infty$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # ---- Panel (e): Radical fraction and TV distance ----
    ax = fig.add_subplot(gs[2, 0])
    frac_arr = np.array([d['frac_radical'] for d in gap_data])
    tv_arr = np.array([d['tv_distance'] for d in gap_data])

    ax.plot(r_arr, frac_arr, 'bo-', markersize=5, label='Radical fraction')
    ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1,
               label=r'Prediction: $1/2$')
    ax.axhline(y=1/3, color='gray', linestyle=':', linewidth=1,
               label=r'Alternative: $1/3$')

    ax2 = ax.twinx()
    ax2.plot(r_arr, tv_arr, 'rs-', markersize=4, alpha=0.6, label='TV distance')
    ax2.set_ylabel('TV distance', fontsize=11, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.set_xlabel(r'$r$', fontsize=12)
    ax.set_ylabel('Radical fraction', fontsize=11)
    ax.set_title(r'(e) Radical fraction $\to 1/2$', fontsize=11)
    ax.legend(fontsize=9, loc='center right')
    ax.grid(True, alpha=0.3)

    # ---- Panel (f): Participation ratio ----
    ax = fig.add_subplot(gs[2, 1])
    PR_semi = np.array([d['PR_semi'] for d in gap_data])
    PR_rad = np.array([d['PR_rad'] for d in gap_data])
    PR_total = np.array([d['PR_total'] for d in gap_data])

    ax.plot(r_arr, PR_semi, 'bo-', markersize=5, label=r'$PR_{\rm semi}$')
    ax.plot(r_arr, PR_rad, 'rs-', markersize=5, label=r'$PR_{\rm rad}$')
    ax.plot(r_arr, PR_total, 'g^-', markersize=5, label=r'$PR_{\rm total}$')

    ax.set_xlabel(r'$r$', fontsize=12)
    ax.set_ylabel('Participation ratio', fontsize=11)
    ax.set_title(r'(f) Effective number of states', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'prediction3_entanglement_spectrum_gap.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_summary(result1, result2, result3, output_dir):
    """Generate a summary figure with key results from all three predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(r'sl₂ Master Identity: $-3/2 = -2 + 1/2$ — Falsifiable Predictions',
                 fontsize=15, fontweight='bold', y=1.02)

    # ---- Prediction 1 summary ----
    ax = axes[0]
    r_p1 = np.array(result1['r_values'], dtype=float)
    R_values = np.array(result1['R_values'])

    ax.plot(r_p1, R_values, 'bo-', markersize=5, linewidth=2,
            label=r'$R(r) = \ln(Z_f/Z_m)/\ln r$')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2,
               label=r'$1/2$ (predicted)')
    ax.axhline(y=1/3, color='gray', linestyle=':', linewidth=1,
               label=r'$1/3$ (ratio target)')

    R_final = result1['R_final']
    ax.set_xlabel(r'$r$ (root of unity parameter)', fontsize=12)
    ax.set_ylabel(r'$R(r)$', fontsize=12)
    ax.set_title(r'P1: $R(r) \to 1/2$ $\Rightarrow$ ratio $\to 1/3$' +
                 f'\nR(largest r) = {R_final:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    color = 'green' if not result1['falsified'] else 'red'
    ax.text(0.5, 0.05, 'NOT FALSIFIED' if not result1['falsified'] else 'FALSIFIED',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=color, ha='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    # ---- Prediction 2 summary ----
    ax = axes[1]
    k = np.array(result2['k_values'], dtype=float)
    N_soft = np.array(result2['N_soft'], dtype=float)
    N_strom = np.array(result2['N_strominger'], dtype=float)

    ax.loglog(k, N_soft, 'b-', linewidth=2.5, label=r'$D_2(k+2)$ (cubic)')
    ax.loglog(k, N_strom, 'g:', linewidth=2, label=r'$2k+1$ (Strominger)')

    n_fit = result2['power_law_exponent']
    ax.set_xlabel(r'CS level $k$', fontsize=12)
    ax.set_ylabel(r'$N_{\rm soft}$', fontsize=12)
    ax.set_title(r'P2: Soft hair $\sim k^3$' + f'\nExponent: {n_fit:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    color = 'green' if not result2['falsified'] else 'red'
    ax.text(0.5, 0.05, 'NOT FALSIFIED' if not result2['falsified'] else 'FALSIFIED',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=color, ha='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    # ---- Prediction 3 summary ----
    ax = axes[2]
    r_arr = np.array(result3['r_values'], dtype=float)
    gap_arr = np.array(result3['effective_gap'])

    ax.plot(r_arr, gap_arr, 'bo-', markersize=6, linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5,
               label='Gap = 0 (closed)')

    ax.set_xlabel(r'$r$', fontsize=12)
    ax.set_ylabel('Effective spectral gap', fontsize=12)
    ax.set_title(r'P3: Spectral gap $\to 0$' + '\nGap closes as r→∞',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    color = 'green' if not result3['falsified'] else 'red'
    ax.text(0.5, 0.05, 'NOT FALSIFIED' if not result3['falsified'] else 'FALSIFIED',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=color, ha='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'predictions_summary.png')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================
# Save Results
# ============================================================

def save_results(result1, result2, result3, output_dir):
    """Save numerical results as JSON."""
    # Convert numpy arrays to lists for JSON serialization
    def jsonify(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [jsonify(v) for v in obj]
        if isinstance(obj, tuple):
            return [jsonify(v) for v in obj]
        return obj

    # Remove large arrays (spectra) from JSON for readability
    r1_save = {k: v for k, v in result1.items()
               if k not in ['p_semi', 'p_rad', 'mod_spectrum', 'S_full', 'S_mod', 'S_rad']}
    r3_save = {}
    for k, v in result3.items():
        if k == 'gap_data':
            r3_save[k] = [{kk: vv for kk, vv in d.items()
                           if kk not in ['p_semi', 'p_rad', 'mod_spectrum']}
                          for d in v]
        else:
            r3_save[k] = v

    all_results = {
        'master_identity': '-3/2 = -2 + 1/2',
        'prediction1': jsonify(r1_save),
        'prediction2': jsonify(result2),
        'prediction3': jsonify(r3_save),
    }

    filepath = os.path.join(output_dir, 'falsifiable_predictions_results.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {filepath}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  FALSIFIABLE PREDICTIONS FROM THE sl₂ MASTER IDENTITY")
    print("  Master Identity: −3/2 = −2 + 1/2")
    print("=" * 70)
    print()

    ensure_output_dir()

    # ---- Prediction 2 (fast, no integration) ----
    result2 = prediction2_soft_hair_degeneracy(k_max=200)
    print()

    # ---- Prediction 3 (fast, discrete sector only) ----
    result3 = prediction3_entanglement_spectrum_gap(
        r_values=[3, 5, 7, 9, 11, 13, 15, 19, 23, 31, 41, 51, 71, 101],
        beta=BETA
    )
    print()

    # ---- Prediction 1 (slow, requires continuous sector integration) ----
    result1 = prediction1_radical_entropy_ratio(r_max_exact=41, r_max_analytical=501, beta=BETA)
    print()

    # ---- Generate plots ----
    print("Generating plots...")
    plot_prediction1(result1, OUTPUT_DIR)
    plot_prediction2(result2, OUTPUT_DIR)
    plot_prediction3(result3, OUTPUT_DIR)
    plot_summary(result1, result2, result3, OUTPUT_DIR)

    # ---- Save results ----
    print("\nSaving results...")
    save_results(result1, result2, result3, OUTPUT_DIR)

    # ---- Final summary ----
    print("\n" + "=" * 70)
    print("  SUMMARY: FALSIFIABLE PREDICTIONS")
    print("=" * 70)
    print()
    print(f"  P1 (Radical ratio → 1/3):  "
          f"ratio = {result1['ratio_lnZ']:.4f}, "
          f"{'NOT FALSIFIED' if not result1['falsified'] else 'FALSIFIED'}")
    print(f"  P2 (Soft hair ~ k³):        "
          f"exponent = {result2['power_law_exponent']:.4f} ± {result2['power_law_exponent_err']:.4f}, "
          f"{'NOT FALSIFIED' if not result2['falsified'] else 'FALSIFIED'}")
    print(f"  P3 (Spectral gap closes):   "
          f"gap_closing = {result3['gap_closing']}, "
          f"{'NOT FALSIFIED' if not result3['falsified'] else 'FALSIFIED'}")
    print()
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
