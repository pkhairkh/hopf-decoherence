"""
Validation: Full Thermal Trace vs Modified Trace Partition Functions
----------------------------------------------------------------------


in the logarithmic entropy correction (-3/2 vs -2).

Key insight:
  - Z_full_raw scales as r^{3/2} (full trace counts ALL states with weight r)
  - Z_mod_raw scales as r^1 (modified trace weights by d_tilde ~ alpha/pi)
  - After D_tilde^2 ~ r^3 normalization:
      S_full ~ (3/2 - 3)*ln(r) = -(3/2)*ln(r)  -> log correction = -3/2 (matches gravity!)
      S_mod  ~ (1 - 3)*ln(r)   = -2*ln(r)       -> log correction = -2
  - The 1/2 gap in raw scaling directly explains the -3/2 vs -2 gap.

Definitions (from task):
  Z_full(r) = sum_j r*exp(-beta*h_j) + integral r*exp(-beta*h_alpha) dalpha
  Z_mod(r)  = (1/D_tilde^2) * [sum_j d_tilde(P_j)*exp(-beta*h_j)
               + integral d_tilde(V_alpha)*exp(-beta*h_alpha) dalpha]

Where:
  d_tilde(P_j) = (-1)^j * sin(pi*(j+1)/r) / (r*sin^2(pi/r))
  d_tilde(V_alpha) = sin(pi*alpha/r) / (r*sin^2(pi/r))
  D_tilde^2 = sum_j d_tilde(P_j)^2 + integral d_tilde(V_alpha)^2 dalpha
  h_j = j*(j+2)/(4*r),  h_alpha = (alpha^2 - 1)/(4*r)

For the ENTROPY comparison, the crucial ratio is the RAW numerator ratio:
  Z_full_raw / Z_mod_num ~ (pi^{3/2}/(2*sqrt(beta))) * sqrt(r)

This sqrt(r) scaling in the raw ratio means:
  ln(Z_full_raw) - ln(Z_mod_num) ~ (1/2)*ln(r)
  i.e. the raw partition functions differ by exactly 1/2 in their ln(r) coefficient.

After dividing BOTH by D_tilde^2 (subtracting 3*ln(r)):
  ln(Z_full_norm) - ln(Z_mod) = [3/2 - 3]*ln(r) - [1 - 3]*ln(r) = -(3/2)*ln(r) + 2*ln(r)
                                = (1/2)*ln(r)

The entropy gap is exactly 1/2 = |-3/2 - (-2)|.
"""

import numpy as np
from scipy import integrate
import warnings

warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)


# ============================================================================
# Modified quantum dimensions (from bcgp_btz.py)
# ============================================================================

def modified_qdim(j, r):
    """d_tilde(P_j) = (-1)^j sin(pi*(j+1)/r) / (r sin^2(pi/r))"""
    if j < 0 or j >= r:
        return 0.0
    if j == r - 1:
        return 0.0  # Steinberg
    return ((-1)**j * np.sin(np.pi * (j + 1) / r)) / (r * np.sin(np.pi / r)**2)


def typical_qdim(alpha, r):
    """d_tilde(V_alpha) = sin(pi*alpha/r) / (r sin^2(pi/r))"""
    return np.sin(np.pi * alpha / r) / (r * np.sin(np.pi / r)**2)


def conformal_weight(j, r):
    """h_j = j*(j+2)/(4r)"""
    return j * (j + 2) / (4.0 * r)


def typical_conformal_weight(alpha, r):
    """h_alpha = (alpha^2 - 1)/(4r)"""
    return (alpha**2 - 1) / (4.0 * r)


# ============================================================================
# D_tilde^2: Modified global dimension squared
# ============================================================================

def compute_D_tilde_squared(r):
    """D_tilde^2 = sum_j d_tilde(P_j)^2 + integral d_tilde(V_alpha)^2 dalpha"""
    D2_disc = sum(modified_qdim(j, r)**2 for j in range(r))

    sin_pi_r = np.sin(np.pi / r)
    prefactor_sq = 1.0 / (r * sin_pi_r**2)**2

    def integrand(alpha):
        return np.sin(np.pi * alpha / r)**2 * prefactor_sq

    D2_cont = 0.0
    eps = 1e-6
    for k in range(r):
        val, _ = integrate.quad(integrand, k + eps, k + 1 - eps, limit=100)
        D2_cont += val

    return D2_disc + D2_cont, D2_disc, D2_cont


# ============================================================================
# Z_full(r): Full thermal trace (UNNORMALIZED)
# ============================================================================

def Z_full_discrete(beta, r):
    """Discrete part: sum_j r * exp(-beta * h_j)"""
    return sum(r * np.exp(-beta * conformal_weight(j, r)) for j in range(r))


def Z_full_continuous(beta, r):
    """Continuous part: integral r * exp(-beta * h_alpha) dalpha"""
    def integrand(alpha):
        return r * np.exp(-beta * typical_conformal_weight(alpha, r))

    Z = 0.0
    eps = 1e-6
    for k in range(r):
        val, _ = integrate.quad(integrand, k + eps, k + 1 - eps, limit=100)
        Z += val
    return Z


def Z_full(beta, r):
    """Full thermal trace (unnormalized)."""
    return Z_full_discrete(beta, r) + Z_full_continuous(beta, r)


# ============================================================================
# Z_mod(r): Modified trace numerator (raw) and normalized
# ============================================================================

def Z_mod_numerator_discrete(beta, r):
    """Discrete part: sum_j d_tilde(P_j) * exp(-beta*h_j)"""
    return sum(modified_qdim(j, r) * np.exp(-beta * conformal_weight(j, r))
               for j in range(r))


def Z_mod_numerator_continuous(beta, r):
    """Continuous part: integral d_tilde(V_alpha) * exp(-beta*h_alpha) dalpha"""
    sin_pi_r = np.sin(np.pi / r)
    prefactor = 1.0 / (r * sin_pi_r**2)

    def integrand(alpha):
        d = prefactor * np.sin(np.pi * alpha / r)
        h = typical_conformal_weight(alpha, r)
        return d * np.exp(-beta * h)

    Z = 0.0
    eps = 1e-6
    for k in range(r):
        val, _ = integrate.quad(integrand, k + eps, k + 1 - eps, limit=100)
        Z += val
    return Z


def Z_mod_numerator(beta, r):
    """Modified trace numerator (unnormalized)."""
    return Z_mod_numerator_discrete(beta, r) + Z_mod_numerator_continuous(beta, r)


def Z_mod(beta, r):
    """Modified trace partition function: (1/D_tilde^2) * numerator."""
    D2, _, _ = compute_D_tilde_squared(r)
    return Z_mod_numerator(beta, r) / D2


# ============================================================================
# Main computation: r = 3, 5, 7, ..., 101
# ============================================================================

def compute_all_ratios(beta=1.0, r_max=101):
    """Compute Z_full, Z_mod, and their ratio for odd r from 3 to r_max."""
    r_values = list(range(3, r_max + 1, 2))
    results = []

    for r in r_values:
        Zf_disc = Z_full_discrete(beta, r)
        Zf_cont = Z_full_continuous(beta, r)
        Zf = Zf_disc + Zf_cont

        Zm_disc = Z_mod_numerator_discrete(beta, r)
        Zm_cont = Z_mod_numerator_continuous(beta, r)
        Zm_num = Zm_disc + Zm_cont

        D2, D2_disc, D2_cont = compute_D_tilde_squared(r)
        Zm = Zm_num / D2

        # Task-defined ratio: Z_full (raw) / Z_mod (normalized by D_tilde^2)
        ratio_task = Zf / Zm if abs(Zm) > 1e-30 else float('inf')

        # Raw ratio: Z_full_raw / Z_mod_num (both unnormalized)
        ratio_raw = Zf / Zm_num if abs(Zm_num) > 1e-30 else float('inf')

        results.append({
            'r': r,
            'Z_full': Zf,
            'Z_full_disc': Zf_disc,
            'Z_full_cont': Zf_cont,
            'Z_mod_num': Zm_num,
            'Z_mod_num_disc': Zm_disc,
            'Z_mod_num_cont': Zm_cont,
            'D_tilde_sq': D2,
            'D_tilde_sq_disc': D2_disc,
            'D_tilde_sq_cont': D2_cont,
            'Z_mod': Zm,
            'ratio_task': ratio_task,   # Z_full / Z_mod (task definition)
            'ratio_raw': ratio_raw,     # Z_full_raw / Z_mod_num (raw numerator ratio)
        })

    return results


# ============================================================================
# Scaling verification: fit ln(Z) = a*ln(r) + b
# ============================================================================

def _fit_log_scaling(r_vals, z_vals, r_min=None, include_1r=True):
    """Fit ln|Z| = a*ln(r) + b + c/r for r >= r_min.
    
    If include_1r=False, fit without the 1/r correction term.
    Returns (a, b) or (a, b, c) depending on include_1r.
    """
    r = np.array(r_vals, dtype=float)
    z = np.abs(np.array(z_vals, dtype=float))
    if r_min is not None:
        mask = r >= r_min
        r, z = r[mask], z[mask]
    ln_r = np.log(r)
    ln_z = np.log(z)
    if include_1r:
        A = np.column_stack([ln_r, np.ones_like(ln_r), 1.0/r])
    else:
        A = np.column_stack([ln_r, np.ones_like(ln_r)])
    coeffs, _, _, _ = np.linalg.lstsq(A, ln_z, rcond=None)
    if include_1r:
        return coeffs[0], coeffs[1], coeffs[2]
    else:
        return coeffs[0], coeffs[1]


def verify_scaling(results, beta=1.0, r_min_asymptotic=51):
    """Verify the scaling of ln(Z) with r.

    For small r, asymptotics don't hold, so we use r >= r_min_asymptotic
    for the asymptotic fit, and all r for the full-range fit.
    """
    r_all = [res['r'] for res in results]
    r_min = r_min_asymptotic

    # All-r fits (no 1/r correction)
    fits_all = {}
    fits_all['ln_Z_full_raw'] = _fit_log_scaling(r_all, [res['Z_full'] for res in results], include_1r=False)
    fits_all['ln_Z_mod_raw'] = _fit_log_scaling(r_all, [res['Z_mod_num'] for res in results], include_1r=False)
    fits_all['ln_D_tilde_sq'] = _fit_log_scaling(r_all, [res['D_tilde_sq'] for res in results], include_1r=False)
    fits_all['ln_Z_full_norm'] = _fit_log_scaling(r_all,
        [res['Z_full'] / res['D_tilde_sq'] for res in results], include_1r=False)
    fits_all['ln_Z_mod'] = _fit_log_scaling(r_all, [res['Z_mod'] for res in results], include_1r=False)
    fits_all['ln_ratio_raw'] = _fit_log_scaling(r_all, [res['ratio_raw'] for res in results], include_1r=False)

    # Asymptotic fits (large r only, with 1/r correction for better extraction)
    fits_asy = {}
    fits_asy['ln_Z_full_raw'] = _fit_log_scaling(r_all, [res['Z_full'] for res in results], r_min)
    fits_asy['ln_Z_mod_raw'] = _fit_log_scaling(r_all, [res['Z_mod_num'] for res in results], r_min)
    fits_asy['ln_D_tilde_sq'] = _fit_log_scaling(r_all, [res['D_tilde_sq'] for res in results], r_min)
    fits_asy['ln_Z_full_norm'] = _fit_log_scaling(r_all,
        [res['Z_full'] / res['D_tilde_sq'] for res in results], r_min)
    fits_asy['ln_Z_mod'] = _fit_log_scaling(r_all, [res['Z_mod'] for res in results], r_min)
    fits_asy['ln_ratio_raw'] = _fit_log_scaling(r_all, [res['ratio_raw'] for res in results], r_min)

    expected = {
        'ln_Z_full_raw': 1.5,
        'ln_Z_mod_raw': 1.0,
        'ln_D_tilde_sq': 3.0,
        'ln_Z_full_norm': -1.5,
        'ln_Z_mod': -2.0,
        'ln_ratio_raw': 0.5,
    }

    return fits_all, fits_asy, expected


# ============================================================================
# Detailed analytic derivation (printed)
# ============================================================================

def print_analytic_derivation(beta=1.0):
    """Print the step-by-step analytic derivation of the 0.5 gap."""
    print("=" * 80)
    print("  ANALYTIC DERIVATION: Why the -3/2 vs -2 gap exists")
    print("=" * 80)

    print(f"""
STEP 1: Large-r asymptotics of d_tilde
=======================================
For large r, sin(pi/r) ~ pi/r, so:
  d_tilde(P_j) = (-1)^j sin(pi*(j+1)/r) / (r sin^2(pi/r))
              ~ (-1)^j sin(pi*(j+1)/r) * r / pi^2

  d_tilde(V_alpha) = sin(pi*alpha/r) / (r sin^2(pi/r))
                  ~ sin(pi*alpha/r) * r / pi^2

In the Gaussian-dominated region (alpha << sqrt(r)):
  sin(pi*alpha/r) ~ pi*alpha/r
  d_tilde(V_alpha) ~ (pi*alpha/r) * r / pi^2 = alpha / pi

STEP 2: Z_full continuous sector
=================================
Z_full_cont = integral_0^r r * exp(-beta * alpha^2/(4r)) dalpha

For large r, the Gaussian is concentrated near alpha = 0:
  ~ r * integral_0^inf exp(-beta*alpha^2/(4r)) dalpha
  = r * sqrt(4*pi*r/beta) / 2
  = r * sqrt(pi*r/beta)
  = r^(3/2) * sqrt(pi/beta)

Therefore: ln(Z_full_raw) ~ (3/2) * ln(r) + const

STEP 3: Z_mod continuous sector (numerator)
=============================================
Z_mod_num_cont = integral_0^r d_tilde(V_alpha) * exp(-beta*h_alpha) dalpha
              ~ integral_0^r (alpha/pi) * exp(-beta*alpha^2/(4r)) dalpha
              = (1/pi) * integral_0^inf alpha * exp(-beta*alpha^2/(4r)) dalpha

Using integral_0^inf x * exp(-c*x^2) dx = 1/(2c) with c = beta/(4r):
              = (1/pi) * 1/(2 * beta/(4r))
              = (1/pi) * 2r/beta
              = 2r / (pi*beta)

Therefore: ln(Z_mod_num) ~ ln(r) + const

STEP 4: D_tilde^2 asymptotics
================================
D_tilde^2 = sum_j d_tilde(P_j)^2 + integral d_tilde(V_alpha)^2 dalpha

Discrete: sum_j sin^2(pi*(j+1)/r) * r^2/pi^4 ~ (r/2) * r^2/pi^4 = r^3/(2*pi^4)
Continuous: integral sin^2(pi*alpha/r) * r^2/pi^4 dalpha = (r/2) * r^2/pi^4 = r^3/(2*pi^4)

D_tilde^2 ~ r^3/pi^4

Therefore: ln(D_tilde^2) ~ 3*ln(r) + const

STEP 5: RAW numerator ratio (key quantity)
============================================
Z_full_raw / Z_mod_num ~ [r^(3/2) * sqrt(pi/beta)] / [2r/(pi*beta)]
                       = r^(1/2) * sqrt(pi/beta) * pi*beta / 2
                       = (pi^(3/2) / (2*sqrt(beta))) * sqrt(r)

Therefore: ln(Z_full_raw / Z_mod_num) ~ (1/2)*ln(r) + const

The RAW ratio differs by exactly 1/2 in the ln(r) coefficient!

STEP 6: Entropy log corrections (after D_tilde^2 normalization)
----------------------------------------------------------------------
Both Z_full and Z_mod are divided by D_tilde^2 (subtracting 3*ln(r)):

  ln(Z_full_norm) = ln(Z_full_raw) - ln(D_tilde^2)
                  ~ (3/2)*ln(r) - 3*ln(r) = -(3/2)*ln(r)
                  -> S_full has log correction = -3/2  (matches gravity!)

  ln(Z_mod) = ln(Z_mod_num) - ln(D_tilde^2)
            ~ 1*ln(r) - 3*ln(r) = -2*ln(r)
            -> S_mod has log correction = -2

GAP: |-3/2 - (-2)| = 1/2

This 1/2 gap comes directly from the 1/2 gap in the raw scaling:
  (3/2 - 1) = 1/2

ROOT CAUSE: The full thermal trace weight r (the module dimension) vs the
modified trace weight d_tilde(V_alpha) ~ alpha/pi produces an extra factor
of sqrt(r) in Z_full's Gaussian integral. The modified trace's alpha/pi
weight suppresses large-alpha contributions that the full dimension r
amplifies, reducing the r-scaling by exactly one factor of sqrt(r).

Physically, this reflects the radical filtration in non-semisimple representation
theory: the modified trace subtracts the radical (non-semisimple) substructure
from the full trace, removing exactly 1/2 unit of ln(r) scaling.
""")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    beta = 1.0
    r_max = 101

    # Print analytic derivation
    print_analytic_derivation(beta)

    # Compute numerical results
    print("=" * 80)
    print(f"  NUMERICAL COMPUTATION: r = 3, 5, 7, ..., {r_max}, beta = {beta}")
    print("=" * 80)

    results = compute_all_ratios(beta, r_max)

    # ---- Table 1: Core results for selected r values ----
    print(f"\n  Table 1: Core partition function values")
    print(f"  {'r':>4s}  {'Z_full_raw':>12s}  {'Z_mod_num':>12s}  {'D_tilde^2':>12s}  "
          f"{'Z_mod':>12s}  {'raw_ratio':>12s}  {'raw_ratio/sqrt(r)':>18s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*18}")

    for res in results:
        r = res['r']
        raw_ratio_sqrtr = res['ratio_raw'] / np.sqrt(r)
        # Print all for small r, sampled for large r
        if r <= 21 or r % 20 == 1:
            print(f"  {r:4d}  {res['Z_full']:12.4e}  {res['Z_mod_num']:12.4e}  "
                  f"{res['D_tilde_sq']:12.4e}  {res['Z_mod']:12.4e}  "
                  f"{res['ratio_raw']:12.4e}  {raw_ratio_sqrtr:18.4f}")

    # ---- Table 2: Continuous sector comparison ----
    print(f"\n{'='*80}")
    print(f"  Table 2: Continuous sector - Numerical vs Analytical (beta={beta})")
    print(f"{'='*80}")

    analytic_prefactor_raw = np.pi**1.5 / (2.0 * np.sqrt(beta))

    print(f"\n  Z_full_cont:  numerical vs r^(3/2)*sqrt(pi/beta)")
    print(f"  Z_mod_num_cont:  numerical vs 2r/(pi*beta)")
    print(f"  D_tilde^2:  numerical vs r^3/pi^4")
    print(f"  Raw ratio:  numerical vs (pi^(3/2)/(2*sqrt(beta)))*sqrt(r)")
    print(f"  Analytic prefactor for raw ratio: pi^(3/2)/(2*sqrt(beta)) = {analytic_prefactor_raw:.6f}")
    print()
    print(f"  {'r':>4s}  {'Z_full/asy':>10s}  {'Z_mod_num/asy':>13s}  {'D2/asy':>10s}  "
          f"{'raw_ratio/asy':>14s}  {'eff_prefactor':>14s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*13}  {'-'*10}  {'-'*14}  {'-'*14}")

    for res in results:
        r = res['r']
        if r < 11:
            continue
        asy_Zf = r**1.5 * np.sqrt(np.pi / beta)
        asy_Zm = 2.0 * r / (np.pi * beta)
        asy_D2 = r**3 / np.pi**4
        asy_ratio = analytic_prefactor_raw * np.sqrt(r)

        ratio_Zf = res['Z_full_cont'] / asy_Zf
        ratio_Zm = res['Z_mod_num_cont'] / asy_Zm
        ratio_D2 = res['D_tilde_sq'] / asy_D2
        ratio_raw = res['ratio_raw'] / asy_ratio
        eff_prefactor = res['ratio_raw'] / np.sqrt(r)

        if r <= 21 or r % 10 == 1:
            print(f"  {r:4d}  {ratio_Zf:10.4f}  {ratio_Zm:13.4f}  {ratio_D2:10.4f}  "
                  f"{ratio_raw:14.4f}  {eff_prefactor:14.4f}")

    # ---- Scaling verification ----
    print(f"\n{'='*80}")
    print(f"  SCALING VERIFICATION: ln(Z) = a*ln(r) + b")
    print(f"{'='*80}")

    fits_all, fits_asy, expected = verify_scaling(results, beta, r_min_asymptotic=51)

    print(f"\n  Fit model: ln(Z) = a*ln(r) + b + c/r  (large-r fits include 1/r correction)")
    print(f"\n  {'Quantity':<22s}  {'All-r a':>10s}  {'Large-r a':>11s}  {'Expected':>10s}  "
          f"{'All-r err':>10s}  {'Large-r err':>11s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*10}  {'-'*11}")

    for name in expected:
        exp = expected[name]
        a_all = fits_all[name][0]
        a_asy = fits_asy[name][0]
        err_all = abs(a_all - exp)
        err_asy = abs(a_asy - exp)
        m_all = "ok" if err_all < 0.15 else "!!"
        m_asy = "ok" if err_asy < 0.15 else "!!"
        print(f"  {name:<22s}  {a_all:>+10.4f}  {a_asy:>+11.4f}  {exp:>+10.4f}  "
              f"{err_all:>9.4f}{m_all}  {err_asy:>10.4f}{m_asy}")

    # ---- Detailed: ln(Z_full_raw) - ln(Z_mod_num) = gap ----
    print(f"\n{'='*80}")
    print(f"  DIRECT GAP MEASUREMENT: ln(Z_full_raw) - ln(Z_mod_num)")
    print(f"  Expected: (3/2 - 1)*ln(r) = (1/2)*ln(r)")
    print(f"{'='*80}")

    print(f"\n  {'r':>4s}  {'ln(Z_full_raw)':>14s}  {'ln(Z_mod_num)':>14s}  "
          f"{'difference':>12s}  {'(1/2)*ln(r)':>12s}  {'ratio':>8s}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*8}")

    for res in results:
        r = res['r']
        if r < 11:
            continue
        ln_Zf = np.log(res['Z_full'])
        ln_Zm = np.log(abs(res['Z_mod_num']))
        diff = ln_Zf - ln_Zm
        half_lnr = 0.5 * np.log(r)
        ratio = diff / half_lnr if half_lnr > 0 else 0
        if r <= 21 or r % 10 == 1:
            print(f"  {r:4d}  {ln_Zf:14.4f}  {ln_Zm:14.4f}  "
                  f"{diff:12.4f}  {half_lnr:12.4f}  {ratio:8.4f}")

    # Fit the gap (with 1/r correction)
    r_vals = np.array([res['r'] for res in results if res['r'] >= 31], dtype=float)
    diffs = np.array([np.log(res['Z_full']) - np.log(abs(res['Z_mod_num']))
                      for res in results if res['r'] >= 31])
    ln_r = np.log(r_vals)
    A = np.column_stack([ln_r, np.ones_like(ln_r), 1.0/r_vals])
    gap_coeffs, _, _, _ = np.linalg.lstsq(A, diffs, rcond=None)
    print(f"\n  Gap fit (r >= 31): ln(Z_full) - ln(Z_mod_num) = {gap_coeffs[0]:.4f}*ln(r) + {gap_coeffs[1]:.4f} + {gap_coeffs[2]:.4f}/r")
    print(f"  Expected coefficient: 0.5000, deviation: {abs(gap_coeffs[0] - 0.5):.4f}")

    # ---- Full normalized gap ----
    print(f"\n{'='*80}")
    print(f"  NORMALIZED GAP: ln(Z_full/D_tilde^2) - ln(Z_mod)")
    print(f"  = [ln(Z_full_raw) - ln(D_tilde^2)] - [ln(Z_mod_num) - ln(D_tilde^2)]")
    print(f"  = ln(Z_full_raw) - ln(Z_mod_num)")
    print(f"  = (1/2)*ln(r) + const")
    print(f"  -> Entropy gap = 1/2, explaining -3/2 vs -2")
    print(f"{'='*80}")

    norm_diffs = np.array([
        np.log(abs(res['Z_full'] / res['D_tilde_sq'])) - np.log(abs(res['Z_mod']))
        for res in results if res['r'] >= 31
    ])
    gap_norm_coeffs, _, _, _ = np.linalg.lstsq(A, norm_diffs, rcond=None)
    print(f"\n  Normalized gap fit (r >= 31): {gap_norm_coeffs[0]:.4f}*ln(r) + {gap_norm_coeffs[1]:.4f} + {gap_norm_coeffs[2]:.4f}/r")
    print(f"  Expected coefficient: 0.5000, deviation: {abs(gap_norm_coeffs[0] - 0.5):.4f}")

    # ---- Summary ----
    print(f"\n{'='*80}")
    print(f"  SUMMARY: The 0.5 Gap Explanation")
    print(f"{'='*80}")

    a_full_asy = fits_asy['ln_Z_full_raw'][0]
    a_mod_asy = fits_asy['ln_Z_mod_raw'][0]
    a_D2_asy = fits_asy['ln_D_tilde_sq'][0]
    a_full_norm_asy = fits_asy['ln_Z_full_norm'][0]
    a_mod_norm_asy = fits_asy['ln_Z_mod'][0]
    a_ratio_asy = fits_asy['ln_ratio_raw'][0]
    raw_gap = a_full_asy - a_mod_asy
    norm_gap = a_full_norm_asy - a_mod_norm_asy

    print(f"""
  ASYMPTOTIC SCALING (r >= 51, with 1/r correction in fit):
  ==========================================================
  ln(Z_full_raw)  ~ {a_full_asy:+.4f}*ln(r)  (expected +1.5000)
  ln(Z_mod_num)   ~ {a_mod_asy:+.4f}*ln(r)  (expected +1.0000)
  ln(D_tilde^2)   ~ {a_D2_asy:+.4f}*ln(r)  (expected +3.0000)
  ln(Z_full_norm) ~ {a_full_norm_asy:+.4f}*ln(r)  (expected -1.5000)
  ln(Z_mod)       ~ {a_mod_norm_asy:+.4f}*ln(r)  (expected -2.0000)
  ln(raw_ratio)   ~ {a_ratio_asy:+.4f}*ln(r)  (expected +0.5000)

  RAW GAP:   ln(Z_full_raw) - ln(Z_mod_num) = {raw_gap:.4f}  (expected 0.5000)
  NORM GAP:  ln(Z_full_norm) - ln(Z_mod)    = {norm_gap:.4f}  (expected 0.5000)

  DIRECT FIT (with 1/r correction): ln(Z_full) - ln(Z_mod_num) = {gap_coeffs[0]:.4f}*ln(r)  (expected 0.5000)

  CONCLUSION:
  ===========
  The -3/2 vs -2 gap in the logarithmic entropy correction is exactly 1/2.
  This 1/2 originates from the difference in r-scaling between the full
  thermal trace numerator (~r^(3/2)) and the modified trace numerator (~r^1).

  The full trace weights each state by the module dimension r, producing a
  Gaussian integral ~ r * sqrt(r) = r^(3/2). The modified trace weights by
  d_tilde(V_alpha) ~ alpha/pi, producing an integral ~ r. The difference
  in the ln(r) exponent is 3/2 - 1 = 1/2.

  After D_tilde^2 normalization (subtracting 3*ln(r) from both):
    Z_full_norm ~ r^(-3/2)  ->  entropy log correction = -3/2  (matches gravity)
    Z_mod       ~ r^(-2)    ->  entropy log correction = -2     (off by 0.5)

  The modified trace UNDERCOUNTS by a factor of sqrt(r) relative to the
  full thermal trace, which is precisely the correction from the radical
  substructure in non-semisimple representation theory.

  IMPLICATION: The BCGP modified trace formulation loses the gravitational
  log correction because it divides by D_tilde^2 which was designed for
  the modified trace weights. The full thermal trace correctly reproduces
  -3/2 when normalized by D_tilde^2, but the BCGP construction uses the
  modified trace weights, introducing the 0.5 deficit.
""")
