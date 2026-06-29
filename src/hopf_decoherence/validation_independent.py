"""
Independent validation of BCGP BTZ partition function log correction.

Re-implements from scratch to confirm the -3/2 log correction.

Two partition function formulations:
  1) FULL THERMAL TRACE (ordinary trace): dim(P_j)=r, dim(V_alpha)=r
  2) MODIFIED TRACE: d_tilde(P_j), d_tilde(V_alpha) from BCGP

Both normalized by D_tilde^2 = sum d_tilde^2 + integral d_tilde^2.

Target: S_log = -3/2 (gravitational prediction for BTZ)
"""

import numpy as np
from scipy.integrate import quad
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# Core formulas (independent re-implementation, no imports from bcgp_btz)
# =============================================================================

def d_tilde_projective(j, r):
    """Modified quantum dimension of projective module P_j.

    d_tilde(P_j) = (-1)^j * sin(pi*(j+1)/r) / (r * sin^2(pi/r))

    Note: P_{r-1} (Steinberg) has d_tilde = 0 since sin(pi*r/r) = sin(pi) = 0.
    """
    return ((-1.0)**j * np.sin(np.pi * (j + 1) / r)) / (r * np.sin(np.pi / r)**2)


def d_tilde_typical(alpha, r):
    """Modified quantum dimension of typical module V_alpha.

    d_tilde(V_alpha) = sin(pi*alpha/r) / (r * sin^2(pi/r))
    """
    return np.sin(np.pi * alpha / r) / (r * np.sin(np.pi / r)**2)


def h_projective(j, r):
    """Conformal weight of projective module P_j.

    h_j = j*(j+2) / (4*r)
    """
    return j * (j + 2) / (4.0 * r)


def h_typical(alpha, r):
    """Conformal weight of typical module V_alpha.

    h_alpha = (alpha^2 - 1) / (4*r)
    """
    return (alpha**2 - 1.0) / (4.0 * r)


# =============================================================================
# Modified global dimension D_tilde^2
# =============================================================================

def D_tilde_squared(r):
    """Compute D_tilde^2 = sum_{j=0}^{r-2} d_tilde(P_j)^2 + integral_0^r d_tilde(V_alpha)^2 d_alpha.

    The integral is computed segment by segment between integer alpha values
    to avoid singularities.
    """
    # Discrete part
    D2_disc = 0.0
    for j in range(r - 1):  # j = 0, ..., r-2
        d = d_tilde_projective(j, r)
        D2_disc += d**2

    # Continuous part
    sin2 = np.sin(np.pi / r)**2
    prefactor = 1.0 / (r * sin2)**2  # = 1 / (r^2 * sin^4(pi/r))

    def integrand(alpha):
        return np.sin(np.pi * alpha / r)**2 * prefactor

    D2_cont = 0.0
    eps = 1e-8
    for k in range(r):
        a = k + eps
        b = k + 1 - eps
        val, _ = quad(integrand, a, b, limit=200)
        D2_cont += val

    return D2_disc + D2_cont


# =============================================================================
# Partition functions
# =============================================================================

def Z_full_thermal_trace(beta, r):
    """Partition function with full (ordinary) thermal trace.

    Z = (1/D_tilde^2) * [sum_{j=0}^{r-2} r * exp(-beta*h_j)
                         + integral_0^r r * exp(-beta*h_alpha) d_alpha]

    Here dim(P_j) = r for all j, dim(V_alpha) = r for all alpha.
    """
    D2 = D_tilde_squared(r)

    # Discrete sector
    Z_disc = 0.0
    for j in range(r - 1):  # j = 0, ..., r-2
        Z_disc += r * np.exp(-beta * h_projective(j, r))

    # Continuous sector: integrate r * exp(-beta * h_alpha) over [0, r]
    def integrand(alpha):
        return r * np.exp(-beta * h_typical(alpha, r))

    Z_cont = 0.0
    eps = 1e-8
    for k in range(r):
        a = k + eps
        b = k + 1 - eps
        val, _ = quad(integrand, a, b, limit=200)
        Z_cont += val

    return (Z_disc + Z_cont) / D2


def Z_modified_trace(beta, r):
    """Partition function with modified trace (BCGP).

    Z = (1/D_tilde^2) * [sum_{j=0}^{r-2} d_tilde(P_j) * exp(-beta*h_j)
                         + integral_0^r d_tilde(V_alpha) * exp(-beta*h_alpha) d_alpha]
    """
    D2 = D_tilde_squared(r)

    # Discrete sector
    Z_disc = 0.0
    for j in range(r - 1):  # j = 0, ..., r-2
        d = d_tilde_projective(j, r)
        Z_disc += d * np.exp(-beta * h_projective(j, r))

    # Continuous sector
    sin2 = np.sin(np.pi / r)**2
    prefactor = 1.0 / (r * sin2)

    def integrand(alpha):
        d = prefactor * np.sin(np.pi * alpha / r)
        return d * np.exp(-beta * h_typical(alpha, r))

    Z_cont = 0.0
    eps = 1e-8
    for k in range(r):
        a = k + eps
        b = k + 1 - eps
        val, _ = quad(integrand, a, b, limit=200)
        Z_cont += val

    return (Z_disc + Z_cont) / D2


# =============================================================================
# Entropy computation
# =============================================================================

def compute_entropy(Z_func, beta, r, dbeta=1e-6):
    """Compute entropy S = ln(Z) + beta * (d/d beta) ln(Z).

    Uses central finite difference for the beta derivative.
    """
    Z0 = Z_func(beta, r)
    Zp = Z_func(beta + dbeta, r)
    Zm = Z_func(beta - dbeta, r)

    if abs(Z0) < 1e-30:
        return float('nan')

    dlnZ_dbeta = (Zp - Zm) / (2.0 * dbeta * Z0)
    S = np.log(Z0) + beta * dlnZ_dbeta
    return S


# =============================================================================
# Log correction extraction
# =============================================================================

def extract_log_coefficient(Z_func, r_values, beta=1.0):
    """Extract log coefficient a from S(r) = a*ln(r) + b*r + c.

    Returns dict with fitted coefficients and diagnostics.
    """
    r_list = []
    S_list = []

    for r in r_values:
        if r % 2 == 0:
            continue
        try:
            S = compute_entropy(Z_func, beta, r)
            if np.isfinite(S):
                r_list.append(float(r))
                S_list.append(S)
        except Exception as e:
            print(f"  [WARN] r={r} failed: {e}")
            continue

    r_arr = np.array(r_list)
    S_arr = np.array(S_list)

    if len(r_list) < 5:
        return {"log_coefficient": float('nan'), "error": "insufficient data"}

    # Fit S = a*ln(r) + b*r + c
    A = np.column_stack([np.log(r_arr), r_arr, np.ones_like(r_arr)])
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, S_arr, rcond=None)

    # Also fit with 1/r correction: S = a*ln(r) + b*r + c + d/r
    A2 = np.column_stack([np.log(r_arr), r_arr, np.ones_like(r_arr), 1.0 / r_arr])
    coeffs2, _, _, _ = np.linalg.lstsq(A2, S_arr, rcond=None)

    # Compute R^2 for the 3-param fit
    S_pred = A @ coeffs
    SS_res = np.sum((S_arr - S_pred)**2)
    SS_tot = np.sum((S_arr - np.mean(S_arr))**2)
    R2 = 1.0 - SS_res / SS_tot if SS_tot > 0 else 0.0

    return {
        "log_coefficient": coeffs[0],
        "linear_coefficient": coeffs[1],
        "constant": coeffs[2],
        "log_coefficient_4param": coeffs2[0],
        "R_squared": R2,
        "n_points": len(r_list),
        "r_min": int(r_arr[0]),
        "r_max": int(r_arr[-1]),
        "r_values": r_list,
        "entropies": S_list,
    }


# =============================================================================
# MAIN: Run full validation
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  INDEPENDENT VALIDATION: BCGP BTZ Partition Function")
    print("  Target: S_log = -3/2 = -1.5000")
    print("=" * 80)

    # r = 3, 5, 7, ..., 101 (odd values)
    r_values = list(range(3, 102, 2))
    beta = 1.0

    # ------------------------------------------------------------------
    # Part 1: Verify basic quantities for small r
    # ------------------------------------------------------------------
    print("\n--- Part 1: Sanity checks for small r ---")
    for r in [3, 5, 7, 11]:
        print(f"\n  r = {r}:")
        print(f"    sin(pi/r) = {np.sin(np.pi/r):.8f}")
        print(f"    sin^2(pi/r) = {np.sin(np.pi/r)**2:.8f}")

        # Check d_tilde values
        d_vals = [d_tilde_projective(j, r) for j in range(r)]
        print(f"    d_tilde(P_j) = {[f'{d:+.6f}' for d in d_vals]}")

        # D_tilde^2
        D2 = D_tilde_squared(r)
        print(f"    D_tilde^2 = {D2:.8f}")

        # Quick Z values
        Z_full = Z_full_thermal_trace(beta, r)
        Z_mod = Z_modified_trace(beta, r)
        print(f"    Z_full(beta=1, r={r}) = {Z_full:.8e}")
        print(f"    Z_mod(beta=1, r={r}) = {Z_mod:.8e}")

    # ------------------------------------------------------------------
    # Part 2: Full thermal trace log correction
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 2: FULL THERMAL TRACE - Log correction extraction")
    print("=" * 80)

    print(f"  Computing entropy for r = 3, 5, 7, ..., 101 with beta = {beta}")
    print()

    result_full = extract_log_coefficient(Z_full_thermal_trace, r_values, beta=beta)

    print(f"  Number of data points: {result_full['n_points']}")
    print(f"  r range: [{result_full['r_min']}, {result_full['r_max']}]")
    print(f"  R^2 of fit: {result_full['R_squared']:.8f}")
    print()
    print(f"  3-param fit: S = a*ln(r) + b*r + c")
    print(f"    a (log coefficient) = {result_full['log_coefficient']:+.6f}")
    print(f"    b (linear coeff)    = {result_full['linear_coefficient']:+.6f}")
    print(f"    c (constant)        = {result_full['constant']:+.6f}")
    print(f"    Deviation from -3/2 = {abs(result_full['log_coefficient'] - (-1.5)):.6f}")
    print()
    print(f"  4-param fit: S = a*ln(r) + b*r + c + d/r")
    print(f"    a (log coefficient) = {result_full['log_coefficient_4param']:+.6f}")
    print(f"    Deviation from -3/2 = {abs(result_full['log_coefficient_4param'] - (-1.5)):.6f}")

    # ------------------------------------------------------------------
    # Part 3: Modified trace log correction
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 3: MODIFIED TRACE - Log correction extraction")
    print("=" * 80)

    print(f"  Computing entropy for r = 3, 5, 7, ..., 101 with beta = {beta}")
    print()

    result_mod = extract_log_coefficient(Z_modified_trace, r_values, beta=beta)

    print(f"  Number of data points: {result_mod['n_points']}")
    print(f"  r range: [{result_mod['r_min']}, {result_mod['r_max']}]")
    print(f"  R^2 of fit: {result_mod['R_squared']:.8f}")
    print()
    print(f"  3-param fit: S = a*ln(r) + b*r + c")
    print(f"    a (log coefficient) = {result_mod['log_coefficient']:+.6f}")
    print(f"    b (linear coeff)    = {result_mod['linear_coefficient']:+.6f}")
    print(f"    c (constant)        = {result_mod['constant']:+.6f}")
    print(f"    Deviation from -3/2 = {abs(result_mod['log_coefficient'] - (-1.5)):.6f}")
    print()
    print(f"  4-param fit: S = a*ln(r) + b*r + c + d/r")
    print(f"    a (log coefficient) = {result_mod['log_coefficient_4param']:+.6f}")
    print(f"    Deviation from -3/2 = {abs(result_mod['log_coefficient_4param'] - (-1.5)):.6f}")

    # ------------------------------------------------------------------
    # Part 4: Detailed entropy table (subset)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 4: Entropy values (selected r)")
    print("=" * 80)
    print(f"  {'r':>5s}  {'S_full':>12s}  {'S_mod':>12s}  {'ln(r)':>8s}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*8}")

    for r in [3, 5, 7, 11, 15, 21, 31, 51, 71, 101]:
        S_full = compute_entropy(Z_full_thermal_trace, beta, r)
        S_mod = compute_entropy(Z_modified_trace, beta, r)
        print(f"  {r:5d}  {S_full:12.6f}  {S_mod:12.6f}  {np.log(r):8.4f}")

    # ------------------------------------------------------------------
    # Part 5: Scaling analysis with different beta values
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 5: Beta dependence of log coefficient")
    print("=" * 80)

    r_scan = list(range(3, 62, 2))  # Faster scan with fewer r values

    for b in [0.5, 1.0, 2.0, 5.0]:
        res_full = extract_log_coefficient(Z_full_thermal_trace, r_scan, beta=b)
        res_mod = extract_log_coefficient(Z_modified_trace, r_scan, beta=b)
        lc_f = res_full['log_coefficient']
        lc_m = res_mod['log_coefficient']
        print(f"  beta={b:4.1f}: full_trace a={lc_f:+.4f} (dev={abs(lc_f+1.5):.4f}), "
              f"mod_trace a={lc_m:+.4f} (dev={abs(lc_m+1.5):.4f})")

    # ------------------------------------------------------------------
    # Part 6: Convergence analysis (truncate r range)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 6: Convergence of log coefficient with r_max")
    print("=" * 80)

    for r_max in [21, 41, 61, 81, 101]:
        r_sub = [r for r in r_values if r <= r_max]
        res_full = extract_log_coefficient(Z_full_thermal_trace, r_sub, beta=1.0)
        res_mod = extract_log_coefficient(Z_modified_trace, r_sub, beta=1.0)
        lc_f = res_full['log_coefficient']
        lc_m = res_mod['log_coefficient']
        print(f"  r_max={r_max:3d}: full_trace a={lc_f:+.6f} (dev={abs(lc_f+1.5):.6f}), "
              f"mod_trace a={lc_m:+.6f} (dev={abs(lc_m+1.5):.6f})")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    a_full = result_full['log_coefficient']
    a_mod = result_mod['log_coefficient']
    a_full_4 = result_full['log_coefficient_4param']
    a_mod_4 = result_mod['log_coefficient_4param']

    print(f"  Full thermal trace log coefficient (3-param): {a_full:+.6f}")
    print(f"  Modified trace log coefficient (3-param):     {a_mod:+.6f}")
    print(f"  Full thermal trace log coefficient (4-param): {a_full_4:+.6f}")
    print(f"  Modified trace log coefficient (4-param):     {a_mod_4:+.6f}")
    print(f"  Target: -1.5000")
    print(f"  Deviation (full, 3-param): {abs(a_full + 1.5):.6f}")
    print(f"  Deviation (mod, 3-param):  {abs(a_mod + 1.5):.6f}")
    print(f"  Deviation (full, 4-param): {abs(a_full_4 + 1.5):.6f}")
    print(f"  Deviation (mod, 4-param):  {abs(a_mod_4 + 1.5):.6f}")

    if abs(a_full + 1.5) < 0.5:
        print(f"\n  *** FULL THERMAL TRACE MATCHES -3/2 within 0.5 ***")
    else:
        print(f"\n  *** FULL THERMAL TRACE does NOT match -3/2 ***")

    if abs(a_mod + 1.5) < 0.5:
        print(f"  *** MODIFIED TRACE MATCHES -3/2 within 0.5 ***")
    else:
        print(f"  *** MODIFIED TRACE does NOT match -3/2 ***")

    # ------------------------------------------------------------------
    # Part 7: Finite-difference log coefficient extraction
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 7: Finite-difference log coefficient extraction")
    print("=" * 80)
    print("  Computing dS/d(ln r) = r * (S(r+2) - S(r-2)) / 4 for large r")
    print()

    for label, Z_func in [("Full thermal", Z_full_thermal_trace),
                           ("Modified trace", Z_modified_trace)]:
        print(f"  {label}:")
        print(f"  {'r':>5s}  {'S(r)':>12s}  {'dS/dlnr':>12s}  {'local_a':>12s}")
        print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*12}")

        local_a_vals = []
        r_large = list(range(51, 102, 2))  # Only large r
        S_cache = {}
        for r in r_large:
            S_cache[r] = compute_entropy(Z_func, beta, r)

        for i in range(1, len(r_large) - 1):
            r = r_large[i]
            Sp = S_cache[r_large[i+1]]
            Sm = S_cache[r_large[i-1]]
            S0 = S_cache[r]
            dS_dlnr = r * (Sp - Sm) / (r_large[i+1] - r_large[i-1])
            # dS/dlnr = a + b*r, so local_a = dS/dlnr - b*r
            # But we don't know b exactly. Just report dS/dlnr as the
            # "instantaneous" log coefficient (which includes b*r contamination)
            print(f"  {r:5d}  {S0:12.6f}  {dS_dlnr:12.6f}")
            local_a_vals.append(dS_dlnr)

        # The dS/dlnr values should approach a as r -> infinity if b is small
        # Since b ≈ -0.004 and r ≈ 75, the contamination is about -0.3
        # Let's fit dS/dlnr = a + b*r to extract a
        r_mid = np.array(r_large[1:-1], dtype=float)
        ds_arr = np.array(local_a_vals)
        A_fd = np.column_stack([np.ones_like(r_mid), r_mid])
        c_fd, _, _, _ = np.linalg.lstsq(A_fd, ds_arr, rcond=None)
        print(f"  Fit dS/dlnr = a + b*r: a = {c_fd[0]:+.6f}, b = {c_fd[1]:+.6f}")
        print(f"  Deviation from -3/2: {abs(c_fd[0] + 1.5):.6f}")
        print()

    # ------------------------------------------------------------------
    # Part 8: 5-parameter fit with additional subleading terms
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Part 8: Extended fits with subleading terms")
    print("=" * 80)

    for label, Z_func in [("Full thermal", Z_full_thermal_trace),
                           ("Modified trace", Z_modified_trace)]:
        print(f"\n  {label}:")

        r_list = []
        S_list = []
        for r in r_values:
            if r % 2 == 0:
                continue
            try:
                S = compute_entropy(Z_func, beta, r)
                if np.isfinite(S):
                    r_list.append(float(r))
                    S_list.append(S)
            except Exception:
                continue

        r_arr = np.array(r_list)
        S_arr = np.array(S_list)

        # 5-param: a*ln(r) + b*r + c + d/r + e/r^2
        A5 = np.column_stack([np.log(r_arr), r_arr, np.ones_like(r_arr),
                               1.0/r_arr, 1.0/r_arr**2])
        c5, _, _, _ = np.linalg.lstsq(A5, S_arr, rcond=None)
        print(f"    5-param (a*ln + b*r + c + d/r + e/r^2): a = {c5[0]:+.6f}, dev = {abs(c5[0]+1.5):.6f}")

        # 6-param: a*ln(r) + b*r + c + d/r + e*ln(r)/r + f/r^2
        A6 = np.column_stack([np.log(r_arr), r_arr, np.ones_like(r_arr),
                               1.0/r_arr, np.log(r_arr)/r_arr, 1.0/r_arr**2])
        c6, _, _, _ = np.linalg.lstsq(A6, S_arr, rcond=None)
        print(f"    6-param (a*ln + b*r + c + d/r + e*ln/r + f/r^2): a = {c6[0]:+.6f}, dev = {abs(c6[0]+1.5):.6f}")

        # Fit using only r >= 31 to reduce finite-size effects
        mask = r_arr >= 31
        r_sub = r_arr[mask]
        S_sub = S_arr[mask]
        A3_large = np.column_stack([np.log(r_sub), r_sub, np.ones_like(r_sub)])
        c3l, _, _, _ = np.linalg.lstsq(A3_large, S_sub, rcond=None)
        print(f"    3-param (r>=31 only): a = {c3l[0]:+.6f}, dev = {abs(c3l[0]+1.5):.6f}")

        A4_large = np.column_stack([np.log(r_sub), r_sub, np.ones_like(r_sub), 1.0/r_sub])
        c4l, _, _, _ = np.linalg.lstsq(A4_large, S_sub, rcond=None)
        print(f"    4-param (r>=31 only): a = {c4l[0]:+.6f}, dev = {abs(c4l[0]+1.5):.6f}")

        # Fit using only r >= 51
        mask2 = r_arr >= 51
        r_sub2 = r_arr[mask2]
        S_sub2 = S_arr[mask2]
        A3_vl = np.column_stack([np.log(r_sub2), r_sub2, np.ones_like(r_sub2)])
        c3vl, _, _, _ = np.linalg.lstsq(A3_vl, S_sub2, rcond=None)
        print(f"    3-param (r>=51 only): a = {c3vl[0]:+.6f}, dev = {abs(c3vl[0]+1.5):.6f}")

        A4_vl = np.column_stack([np.log(r_sub2), r_sub2, np.ones_like(r_sub2), 1.0/r_sub2])
        c4vl, _, _, _ = np.linalg.lstsq(A4_vl, S_sub2, rcond=None)
        print(f"    4-param (r>=51 only): a = {c4vl[0]:+.6f}, dev = {abs(c4vl[0]+1.5):.6f}")
