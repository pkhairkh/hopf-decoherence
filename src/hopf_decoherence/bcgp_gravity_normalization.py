"""
BCGP → CS/WRT Gravitational Normalization Factor
==================================================

Derives the universal normalization factor N(g,r) that maps BCGP TQFT
partition functions to Chern-Simons / Witten-Reshetikhin-Turaev (CS/WRT)
gravitational partition functions.

KEY RESULTS:
  For sl_N at root of unity parameter r:

  1. D̃² POWER LAW: D̃² ~ r^{(N-1)(2N-1)}
     - sl₂: r³  (matches dim = 3)
     - sl₃: r¹⁰ (exceeds dim = 8 by 2)
     - sl₄: r²¹ (exceeds dim = 15 by 6)

  2. D̃² EXCESS OVER dim(G): (N-1)(N-2)
     - sl₂: 0  (no excess — D̃² matches dim)
     - sl₃: 2
     - sl₄: 6

  3. Z_raw POWER: Z_raw ~ r^{3(N-1)/2}
     - sl₂: r^{3/2}
     - sl₃: r³
     - sl₄: r^{9/2}

  4. Z_raw DEFICIT from CS/WRT requirement: (N-1)(N-2)/2
     - sl₂: 0
     - sl₃: 1

  5. NORMALIZATION FACTOR: N(sl_N, r) = r^{3(N-1)(N-2)/2}
     - sl₂: r⁰ = 1 (trivial — BCGP already agrees with gravity)
     - sl₃: r³
     - sl₄: r⁹

ANALYTICAL DERIVATION OF D̃² POWER LAW:
  d̃(P(λ)) = (-1)^|λ| · ∏_{α∈Φ⁺} sin(π⟨λ+ρ,α∨⟩/r)
             / [r^{N-1} · ∏_{α∈Φ⁺} sin²(π⟨ρ,α∨⟩/r)]

  For d̃², the denominator becomes:
    r^{2(N-1)} · ∏_{α∈Φ⁺} sin⁴(π⟨ρ,α∨⟩/r)
    ≈ r^{2(N-1)} · ∏_{α∈Φ⁺} (π⟨ρ,α∨⟩/r)⁴
    = const · r^{2(N-1) - 4|Φ⁺|}
    = const · r^{2(N-1) - 2N(N-1)}

  The numerator sum/integral scales as r^{N-1} (from integrating over
  the (N-1)-dimensional alcove/continuous domain).

  Therefore: D̃² ~ r^{N-1} / r^{2(N-1) - 2N(N-1)}
                  = r^{N-1 - 2(N-1) + 2N(N-1)}
                  = r^{2N(N-1) - (N-1)}
                  = r^{(N-1)(2N-1)}

  Verification:
    sl₂: (1)(3) = 3  ✓
    sl₃: (2)(5) = 10 ✓
    sl₄: (3)(7) = 21

WHY D̃² ≠ dim(G) FOR N ≥ 3:
  D̃² power - dim(G) = (N-1)(2N-1) - (N²-1)
                     = 2N² - 3N + 1 - N² + 1
                     = N² - 3N + 2
                     = (N-1)(N-2)

  This excess comes from the Weyl denominator structure:
  the product ∏_{α∈Φ⁺} sin⁴(π⟨ρ,α∨⟩/r) introduces 4|Φ⁺| powers
  of (1/r) in the denominator, but |Φ⁺| = N(N-1)/2 grows faster
  than (N-1)/2 (the growth of the numerator sum), creating an excess.

  For sl₂: |Φ⁺| = 1, and the numerator sum grows as r, so:
    denominator: r²·(1/r)⁴ = 1/r² → D̃² ~ r·r² = r³ = dim(G)
  For sl₃: |Φ⁺| = 3, and the numerator sum grows as r², so:
    denominator: r⁴·(1/r)⁸·(1/r)⁴ = 1/r⁸ → D̃² ~ r²·r⁸ = r¹⁰ ≠ dim(G)=8

  The key difference: for sl₂, the number of positive roots (1) equals
  the rank (1), so the denominator and numerator powers balance to give
  dim(G). For N ≥ 3, |Φ⁺| > rank, creating the excess.

NORMALIZATION FACTOR DERIVATION:
  BCGP log coeff = Z_raw_power - D̃²_power
                 = 3(N-1)/2 - (N-1)(2N-1)

  CS/WRT log coeff = -dim(G)/2 = -(N²-1)/2

  Required: N(g,r) × Z_BCGP = Z_CS
  → N_power = CS_log - BCGP_log
            = -(N²-1)/2 - [3(N-1)/2 - (N-1)(2N-1)]
            = 3(N-1)(N-2)/2

  For sl₂: N = r⁰ = 1 (trivial) ✓
  For sl₃: N = r³ ✓
  For sl₄: N = r⁹

References:
  [1] BCGP: Blanchet-Costantino-Geer-Patureau-Mirand, arXiv:1605.07941
  [2] Sen (2012), arXiv:1205.0971 — Log corrections from Euclidean gravity
  [3] Giombi-Maloney-Yin (2008), arXiv:0803.2195 — 1-loop AdS3 gravity
"""

import numpy as np
from scipy import integrate
import json
import os
import time
import warnings

warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================================
# PART 1: sl₂ FORMULAS (exact)
# ============================================================================

def D_tilde_squared_sl2(r):
    """Modified global dimension D̃² for sl₂.
    
    Exact formula: D̃² = 1/(r sin⁴(π/r)) ~ r³/π⁴
    """
    return 1.0 / (r * np.sin(np.pi / r) ** 4)


def D_tilde_squared_sl2_asymp(r):
    """Asymptotic approximation D̃² ≈ r³/π⁴ for large r."""
    return r ** 3 / np.pi ** 4


# ============================================================================
# PART 2: sl₃ FORMULAS (numerical)
# ============================================================================

def enumerate_alcove_sl3(r):
    """Enumerate the alcove C_r = {(a,b) : a >= 0, b >= 0, a+b <= r-2}."""
    alcove = []
    for a in range(r - 1):
        for b in range(r - 1 - a):
            alcove.append((a, b))
    return alcove


def modified_qdim_disc_sl3(a, b, r):
    """Modified quantum dimension d̃(P(a,b)) for sl₃ discrete sector.
    
    d̃(P(a,b)) = (-1)^{a+b} sin(π(a+1)/r) sin(π(b+1)/r) sin(π(a+b+2)/r)
                 / (r² sin⁴(π/r) sin²(2π/r))
    """
    if a + b > r - 2:
        return 0.0
    sign = (-1) ** (a + b)
    num = (np.sin(np.pi * (a + 1) / r) *
           np.sin(np.pi * (b + 1) / r) *
           np.sin(np.pi * (a + b + 2) / r))
    den = r ** 2 * np.sin(np.pi / r) ** 4 * np.sin(2 * np.pi / r) ** 2
    return sign * num / den


def modified_qdim_cont_sl3(a1, a2, r):
    """Modified quantum dimension d̃(V_α) for sl₃ typical module."""
    num = (np.sin(np.pi * a1 / r) *
           np.sin(np.pi * a2 / r) *
           np.sin(np.pi * (a1 + a2) / r))
    den = r ** 2 * np.sin(np.pi / r) ** 4 * np.sin(2 * np.pi / r) ** 2
    return num / den


def D_tilde_squared_disc_sl3(r):
    """Discrete part of D̃² for sl₃: Σ d̃(P(a,b))²."""
    total = 0.0
    for a, b in enumerate_alcove_sl3(r):
        d = modified_qdim_disc_sl3(a, b, r)
        total += d * d
    return total


def D_tilde_squared_cont_sl3(r):
    """Continuous part of D̃² for sl₃: ∫∫ d̃(V_α)² dα over [0,r]²."""
    def integrand(a2, a1):
        d = modified_qdim_cont_sl3(a1, a2, r)
        return d * d

    val, err = integrate.dblquad(
        integrand, 0, r, 0, r, epsabs=1e-10, epsrel=1e-8
    )
    return val


def D_tilde_squared_sl3(r):
    """Total D̃² for sl₃ = discrete + continuous."""
    d2d = D_tilde_squared_disc_sl3(r)
    d2c = D_tilde_squared_cont_sl3(r)
    return d2d + d2c, d2d, d2c


# ============================================================================
# PART 3: sl₄ FORMULAS (numerical, for hypothesis testing)
# ============================================================================

def enumerate_alcove_sl4(r):
    """Enumerate the alcove for sl₄: {(a,b,c) : a,b,c >= 0, a+b+c <= r-2}."""
    alcove = []
    for a in range(r - 1):
        for b in range(r - 1 - a):
            for c in range(r - 1 - a - b):
                alcove.append((a, b, c))
    return alcove


def modified_qdim_disc_sl4(a, b, c, r):
    """Modified quantum dimension d̃(P(a,b,c)) for sl₄.
    
    For sl₄, the positive roots are:
      α₁ = ε₁-ε₂, α₂ = ε₂-ε₃, α₃ = ε₃-ε₄,
      α₁+α₂ = ε₁-ε₃, α₂+α₃ = ε₂-ε₄, α₁+α₂+α₃ = ε₁-ε₄
    
    The inner products ⟨λ+ρ, α∨⟩ for λ = (a,b,c):
      ⟨λ+ρ, α₁∨⟩ = a + 1
      ⟨λ+ρ, α₂∨⟩ = b + 1
      ⟨λ+ρ, α₃∨⟩ = c + 1
      ⟨λ+ρ, (α₁+α₂)∨⟩ = a + b + 2
      ⟨λ+ρ, (α₂+α₃)∨⟩ = b + c + 2
      ⟨λ+ρ, (α₁+α₂+α₃)∨⟩ = a + b + c + 3
    
    The denominator involves:
      r³ · ∏ sin²(π⟨ρ,α∨⟩/r)
    where ⟨ρ,α∨⟩ = ht(α):
      ht(α₁) = 1, ht(α₂) = 1, ht(α₃) = 1
      ht(α₁+α₂) = 2, ht(α₂+α₃) = 2, ht(α₁+α₂+α₃) = 3
    
    So denominator = r³ · sin²(π/r)³ · sin²(2π/r)² · sin²(3π/r)
    """
    if a + b + c > r - 2:
        return 0.0
    
    sign = (-1) ** (a + b + c)
    
    # Numerator: product of sines over 6 positive roots
    num = (np.sin(np.pi * (a + 1) / r) *
           np.sin(np.pi * (b + 1) / r) *
           np.sin(np.pi * (c + 1) / r) *
           np.sin(np.pi * (a + b + 2) / r) *
           np.sin(np.pi * (b + c + 2) / r) *
           np.sin(np.pi * (a + b + c + 3) / r))
    
    # Denominator: r^{N-1} × ∏ sin²(π⟨ρ,α∨⟩/r)
    den = (r ** 3 *
           np.sin(np.pi / r) ** 6 *      # 3 simple roots, each sin²
           np.sin(2 * np.pi / r) ** 4 *   # 2 height-2 roots, each sin²
           np.sin(3 * np.pi / r) ** 2)    # 1 height-3 root, sin²
    
    return sign * num / den


def D_tilde_squared_disc_sl4(r):
    """Discrete part of D̃² for sl₄: Σ d̃(P(a,b,c))²."""
    total = 0.0
    for a, b, c in enumerate_alcove_sl4(r):
        d = modified_qdim_disc_sl4(a, b, c, r)
        total += d * d
    return total


# ============================================================================
# PART 4: POWER LAW EXTRACTION
# ============================================================================

def extract_power_law(r_values, values, label=""):
    """Extract power law from (r, value) data.
    
    Fits ln(value) = p · ln(r) + c + d/r using least squares.
    Also computes finite-difference estimate.
    """
    r_arr = np.array(r_values, dtype=float)
    v_arr = np.array(values, dtype=float)
    
    # Filter valid data
    mask = (v_arr > 0) & np.isfinite(v_arr)
    r_arr = r_arr[mask]
    v_arr = v_arr[mask]
    
    if len(r_arr) < 3:
        return {'power': float('nan'), 'method': 'insufficient_data'}
    
    ln_r = np.log(r_arr)
    ln_v = np.log(v_arr)
    
    # 3-param fit: ln(v) = p·ln(r) + c + d/r
    A3 = np.column_stack([ln_r, np.ones_like(r_arr), 1.0 / r_arr])
    c3, _, _, _ = np.linalg.lstsq(A3, ln_v, rcond=None)
    power_3p = c3[0]
    
    # 4-param fit: ln(v) = p·ln(r) + c + d/r + e/r²
    if len(r_arr) >= 5:
        A4 = np.column_stack([ln_r, np.ones_like(r_arr), 1.0 / r_arr, 1.0 / r_arr ** 2])
        c4, _, _, _ = np.linalg.lstsq(A4, ln_v, rcond=None)
        power_4p = c4[0]
    else:
        power_4p = float('nan')
    
    # Finite difference: d(ln v)/d(ln r)
    fd = np.diff(ln_v) / np.diff(ln_r)
    r_mid = np.exp(0.5 * (ln_r[:-1] + ln_r[1:]))
    
    # Fit finite differences to asymptotic value
    if len(r_mid) >= 3:
        A_fd = np.column_stack([np.ones_like(r_mid), 1.0 / r_mid, 1.0 / r_mid ** 2])
        c_fd, _, _, _ = np.linalg.lstsq(A_fd, fd, rcond=None)
        power_fd = c_fd[0]
    else:
        power_fd = fd[-1] if len(fd) > 0 else float('nan')
    
    return {
        'power_3param': power_3p,
        'power_4param': power_4p,
        'power_fd': power_fd,
        'fd_values': list(zip(r_mid.tolist(), fd.tolist())),
        'n_points': len(r_arr),
        'label': label,
    }


# ============================================================================
# PART 5: PARTITION FUNCTION SCALING
# ============================================================================

def casimir_sl3(a, b):
    """Quadratic Casimir C₂(a,b) = (a² + b² + ab + 2a + 2b)/3."""
    return (a ** 2 + b ** 2 + a * b + 2 * a + 2 * b) / 3.0


def weyl_dim_sl3(a, b):
    """Weyl dimension formula: dim L(a,b) = (a+1)(b+1)(a+b+2)/2."""
    return (a + 1) * (b + 1) * (a + b + 2) // 2


def Z_full_sl3(beta, r):
    """Full thermal trace for sl₃ at fixed β."""
    # Discrete sector (using Weyl dimensions)
    Z_disc = 0.0
    for a, b in enumerate_alcove_sl3(r):
        d_L = weyl_dim_sl3(a, b)
        h = casimir_sl3(a, b) / r
        Z_disc += d_L * np.exp(-beta * h)
    
    # Continuous sector: ∫∫ r² exp(-β h) dα₁ dα₂
    alpha_max = min(r, 8.0 * np.sqrt(3.0 * r / max(beta, 0.01)))
    
    def integrand(a2, a1):
        h = casimir_sl3(a1, a2) / r
        return r ** 2 * np.exp(-beta * h)
    
    Z_cont, _ = integrate.dblquad(
        integrand, 0, alpha_max, 0, alpha_max, epsabs=1e-10, epsrel=1e-8
    )
    
    return Z_disc + Z_cont


# ============================================================================
# PART 6: ANALYTICAL FORMULAS
# ============================================================================

def D2_power_formula(N):
    """Predicted D̃² power law exponent for sl_N.
    
    D̃² ~ r^{(N-1)(2N-1)}
    """
    return (N - 1) * (2 * N - 1)


def D2_excess_over_dim(N):
    """D̃² power minus dim(G) = (N-1)(N-2)."""
    return (N - 1) * (N - 2)


def Z_raw_power(N):
    """Z_raw power for sl_N: 3(N-1)/2."""
    return 3.0 * (N - 1) / 2.0


def Z_raw_deficit(N):
    """Z_raw deficit from CS/WRT requirement: (N-1)(N-2)/2."""
    return (N - 1) * (N - 2) / 2.0


def cs_wrt_log_coeff(N):
    """CS/WRT log correction coefficient: -dim(G)/2 = -(N²-1)/2."""
    return -(N ** 2 - 1) / 2.0


def bcgp_log_coeff(N):
    """BCGP full trace log correction: Z_raw_power - D̃²_power."""
    return Z_raw_power(N) - D2_power_formula(N)


def normalization_power(N):
    """Normalization factor power: N(sl_N, r) = r^{3(N-1)(N-2)/2}."""
    return 3.0 * (N - 1) * (N - 2) / 2.0


def analytical_summary(N_values=None):
    """Generate analytical summary table for all sl_N."""
    if N_values is None:
        N_values = [2, 3, 4, 5, 6]
    
    results = []
    for N in N_values:
        results.append({
            'N': N,
            'algebra': f'sl_{N}',
            'rank': N - 1,
            'dim_G': N ** 2 - 1,
            'n_pos_roots': N * (N - 1) // 2,
            'D2_power': D2_power_formula(N),
            'D2_excess': D2_excess_over_dim(N),
            'Z_raw_power': Z_raw_power(N),
            'Z_raw_deficit': Z_raw_deficit(N),
            'bcgp_log_coeff': bcgp_log_coeff(N),
            'cs_wrt_log_coeff': cs_wrt_log_coeff(N),
            'normalization_power': normalization_power(N),
            'total_discrepancy': D2_excess_over_dim(N) + Z_raw_deficit(N),
        })
    
    return results


# ============================================================================
# PART 7: COMPREHENSIVE NUMERICAL VERIFICATION
# ============================================================================

def verify_D2_power_law_sl2(r_values=None):
    """Verify D̃² ~ r³ for sl₂."""
    if r_values is None:
        r_values = list(range(5, 201, 2))
    
    print("=" * 80)
    print("  VERIFICATION: D̃² ~ r³ for sl₂")
    print("=" * 80)
    
    r_valid = [r for r in r_values if r >= 3]
    D2_values = [D_tilde_squared_sl2(r) for r in r_valid]
    
    result = extract_power_law(r_valid, D2_values, "sl₂ D̃²")
    
    print(f"\n  Exact formula: D̃² = 1/(r sin⁴(π/r)) ≈ r³/π⁴")
    print(f"  Numerical power law exponent:")
    print(f"    3-param fit: {result['power_3param']:.6f} (target: 3.0)")
    print(f"    4-param fit: {result['power_4param']:.6f} (target: 3.0)")
    print(f"    Finite-diff: {result['power_fd']:.6f} (target: 3.0)")
    
    # Verify asymptotic constant
    r_large = r_valid[-1]
    D2_exact = D_tilde_squared_sl2(r_large)
    D2_asymp = D_tilde_squared_sl2_asymp(r_large)
    print(f"\n  At r={r_large}: D̃²_exact = {D2_exact:.6f}, D̃²_asymp = r³/π⁴ = {D2_asymp:.6f}")
    print(f"  Ratio: {D2_exact / D2_asymp:.6f} (target: 1.0)")
    
    return {
        'algebra': 'sl₂',
        'D2_values': dict(zip(r_valid, D2_values)),
        'power_3param': result['power_3param'],
        'power_4param': result['power_4param'],
        'power_fd': result['power_fd'],
        'target': 3.0,
        'match': abs(result['power_fd'] - 3.0) < 0.1,
    }


def verify_D2_power_law_sl3(r_values=None):
    """Verify D̃² ~ r¹⁰ for sl₃."""
    if r_values is None:
        r_values = [4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 21, 25, 31]
    
    print("\n" + "=" * 80)
    print("  VERIFICATION: D̃² ~ r¹⁰ for sl₃")
    print("=" * 80)
    
    D2_total = []
    D2_disc = []
    D2_cont = []
    r_valid = []
    
    print(f"\n  {'r':>4s}  {'D̃²_total':>14s}  {'D̃²_disc':>14s}  {'D̃²_cont':>14s}  "
          f"{'disc/cont':>10s}  {'D̃²·π⁶/r¹⁰':>12s}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*12}")
    
    for r in r_values:
        try:
            total, disc, cont = D_tilde_squared_sl3(r)
            D2_total.append(total)
            D2_disc.append(disc)
            D2_cont.append(cont)
            r_valid.append(r)
            
            ratio = disc / cont if cont > 0 else float('nan')
            # Check asymptotic constant: D̃² ≈ C · r¹⁰/π⁶
            asymp_ratio = total * np.pi ** 6 / r ** 10
            print(f"  {r:4d}  {total:14.4f}  {disc:14.4f}  {cont:14.4f}  "
                  f"{ratio:10.4f}  {asymp_ratio:12.6f}")
        except Exception as e:
            print(f"  {r:4d}  FAILED: {e}")
    
    # Power law extraction
    result_total = extract_power_law(r_valid, D2_total, "sl₃ D̃² total")
    result_disc = extract_power_law(r_valid, D2_disc, "sl₃ D̃² disc")
    result_cont = extract_power_law(r_valid, D2_cont, "sl₃ D̃² cont")
    
    print(f"\n  Power law exponents:")
    print(f"    D̃²_total: 3-param={result_total['power_3param']:.4f}, "
          f"4-param={result_total['power_4param']:.4f}, "
          f"FD={result_total['power_fd']:.4f} (target: 10.0)")
    print(f"    D̃²_disc:  3-param={result_disc['power_3param']:.4f}, "
          f"FD={result_disc['power_fd']:.4f}")
    print(f"    D̃²_cont:  3-param={result_cont['power_3param']:.4f}, "
          f"FD={result_cont['power_fd']:.4f}")
    
    # Verify disc/cont = 0.5 exactly
    ratios = [d / c for d, c in zip(D2_disc, D2_cont) if c > 0]
    print(f"\n  D̃²_disc / D̃²_cont ratio: {ratios[0]:.6f} "
          f"(max deviation from 0.5: {max(abs(r - 0.5) for r in ratios):.2e})")
    
    return {
        'algebra': 'sl₃',
        'D2_total': dict(zip(r_valid, D2_total)),
        'D2_disc': dict(zip(r_valid, D2_disc)),
        'D2_cont': dict(zip(r_valid, D2_cont)),
        'power_total_3p': result_total['power_3param'],
        'power_total_4p': result_total['power_4param'],
        'power_total_fd': result_total['power_fd'],
        'power_disc_fd': result_disc['power_fd'],
        'power_cont_fd': result_cont['power_fd'],
        'target': 10.0,
        'match': abs(result_total['power_fd'] - 10.0) < 0.5,
        'disc_cont_ratio': ratios,
    }


def verify_D2_power_law_sl4(r_values=None):
    """Verify D̃² ~ r²¹ for sl₄ (discrete sector only, small r)."""
    if r_values is None:
        r_values = [5, 6, 7, 8, 9, 10, 11, 13]
    
    print("\n" + "=" * 80)
    print("  VERIFICATION: D̃² ~ r²¹ for sl₄ (discrete sector only)")
    print("=" * 80)
    
    D2_disc = []
    r_valid = []
    
    print(f"\n  {'r':>4s}  {'D̃²_disc':>14s}  {'n_alcove':>10s}  {'D̃²_disc·π¹²/r²¹':>16s}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*10}  {'─'*16}")
    
    for r in r_values:
        try:
            d2 = D_tilde_squared_disc_sl4(r)
            n_alc = len(enumerate_alcove_sl4(r))
            D2_disc.append(d2)
            r_valid.append(r)
            
            # Asymptotic constant check: D̃²_disc ≈ C · r²¹/π¹²
            asymp_ratio = d2 * np.pi ** 12 / r ** 21
            print(f"  {r:4d}  {d2:14.4f}  {n_alc:10d}  {asymp_ratio:16.6f}")
        except Exception as e:
            print(f"  {r:4d}  FAILED: {e}")
    
    # Power law extraction
    result = extract_power_law(r_valid, D2_disc, "sl₄ D̃² disc")
    
    print(f"\n  Discrete sector power law:")
    print(f"    3-param: {result['power_3param']:.4f}")
    print(f"    4-param: {result['power_4param']:.4f}")
    print(f"    FD:      {result['power_fd']:.4f} (target: 21.0)")
    print(f"    NOTE: Only discrete sector; continuous sector adds ~same power.")
    print(f"    Total D̃² predicted power: 21 (= (4-1)(2·4-1) = 3×7)")
    
    return {
        'algebra': 'sl₄',
        'D2_disc': dict(zip(r_valid, D2_disc)),
        'power_disc_3p': result['power_3param'],
        'power_disc_4p': result['power_4param'],
        'power_disc_fd': result['power_fd'],
        'target_disc': 21.0,
        'note': 'Discrete sector only; continuous sector not computed for sl₄',
    }


def verify_Z_raw_scaling_sl3(r_values=None, beta=1.0):
    """Verify Z_raw ~ r³ for sl₃ at fixed β."""
    if r_values is None:
        r_values = [5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31]
    
    print("\n" + "=" * 80)
    print(f"  VERIFICATION: Z_raw ~ r³ for sl₃ (β = {beta})")
    print("=" * 80)
    
    Z_values = []
    r_valid = []
    
    for r in r_values:
        try:
            Z = Z_full_sl3(beta, r)
            if Z > 0 and np.isfinite(Z):
                Z_values.append(Z)
                r_valid.append(r)
                print(f"  r={r:3d}: Z_raw = {Z:.6e}")
        except Exception as e:
            print(f"  r={r:3d}: FAILED: {e}")
    
    result = extract_power_law(r_valid, Z_values, "sl₃ Z_raw")
    
    print(f"\n  Z_raw power law exponent:")
    print(f"    3-param: {result['power_3param']:.4f} (target: 3.0)")
    print(f"    FD:      {result['power_fd']:.4f} (target: 3.0)")
    
    return {
        'algebra': 'sl₃',
        'Z_raw_values': dict(zip(r_valid, Z_values)),
        'power_3p': result['power_3param'],
        'power_fd': result['power_fd'],
        'target': 3.0,
        'beta': beta,
    }


# ============================================================================
# PART 8: NORMALIZATION FACTOR VERIFICATION
# ============================================================================

def verify_normalization_factor():
    """Comprehensive verification of the normalization factor N(sl_N, r)."""
    print("\n" + "=" * 80)
    print("  NORMALIZATION FACTOR: N(sl_N, r) = r^{3(N-1)(N-2)/2}")
    print("=" * 80)
    
    # Analytical summary
    summary = analytical_summary()
    
    print(f"\n  {'N':>3s}  {'alg':>6s}  {'dim':>4s}  {'D̃²_pwr':>7s}  {'excess':>6s}  "
          f"{'Z_raw':>6s}  {'deficit':>7s}  {'BCGP_lc':>8s}  {'CS_lc':>6s}  "
          f"{'N_power':>8s}  {'N':>8s}")
    print(f"  {'─'*3}  {'─'*6}  {'─'*4}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*7}  "
          f"{'─'*8}  {'─'*6}  {'─'*8}  {'─'*8}")
    
    for row in summary:
        N = row['N']
        N_str = f"r^{int(normalization_power(N))}" if normalization_power(N) > 0 else "1"
        print(f"  {N:3d}  sl_{N:>3d}  {row['dim_G']:4d}  {row['D2_power']:7d}  "
              f"{row['D2_excess']:6d}  {row['Z_raw_power']:6.1f}  {row['Z_raw_deficit']:7.1f}  "
              f"{row['bcgp_log_coeff']:8.1f}  {row['cs_wrt_log_coeff']:6.1f}  "
              f"{row['normalization_power']:8.1f}  {N_str:>8s}")
    
    print(f"\n  KEY FORMULAS:")
    print(f"    D̃² power = (N-1)(2N-1)")
    print(f"    D̃² excess = (N-1)(N-2)")
    print(f"    Z_raw power = 3(N-1)/2")
    print(f"    Z_raw deficit = (N-1)(N-2)/2")
    print(f"    Total discrepancy = 3(N-1)(N-2)/2 = D̃²_excess + Z_raw_deficit")
    print(f"    Normalization: N(sl_N, r) = r^{{3(N-1)(N-2)/2}}")
    
    # Verify: BCGP + normalization = CS/WRT
    print(f"\n  VERIFICATION: BCGP_log + N_power = CS/WRT_log")
    for row in summary:
        N = row['N']
        check = row['bcgp_log_coeff'] + row['normalization_power']
        target = row['cs_wrt_log_coeff']
        print(f"    sl_{N}: {row['bcgp_log_coeff']:.1f} + {row['normalization_power']:.1f} "
              f"= {check:.1f} {'=' if abs(check - target) < 0.01 else '≠'} "
              f"{target:.1f} {'✓' if abs(check - target) < 0.01 else '✗'}")
    
    return summary


# ============================================================================
# PART 9: DECOMPOSITION OF DISCREPANCY
# ============================================================================

def decompose_discrepancy():
    """Decompose the BCGP-CS/WRT discrepancy into D̃² excess and Z_raw deficit."""
    print("\n" + "=" * 80)
    print("  DISCREPANCY DECOMPOSITION: BCGP vs CS/WRT")
    print("=" * 80)
    
    for N in [2, 3, 4, 5]:
        bcgp = bcgp_log_coeff(N)
        cs = cs_wrt_log_coeff(N)
        disc = bcgp - cs
        d2_excess = D2_excess_over_dim(N)
        z_deficit = Z_raw_deficit(N)
        
        print(f"\n  sl_{N}:")
        print(f"    BCGP log coeff = {bcgp:.1f}")
        print(f"    CS/WRT log coeff = {cs:.1f}")
        print(f"    Total discrepancy = {disc:.1f}")
        print(f"      D̃² excess contribution = -{d2_excess} "
              f"(D̃² ~ r^{{{D2_power_formula(N)}}} instead of r^{{{N**2-1}}})")
        print(f"      Z_raw deficit contribution = -{z_deficit:.1f} "
              f"(Z_raw ~ r^{{{Z_raw_power(N):.1f}}} instead of r^{{{(N**2-1)/2:.1f}}})")
        print(f"      Sum = -{d2_excess + z_deficit:.1f} = total discrepancy ✓" 
              if abs(disc - (-(d2_excess + z_deficit))) < 0.01 else
              f"      Sum = -{d2_excess + z_deficit:.1f} ≠ total discrepancy ✗")


# ============================================================================
# PART 10: COMPLETE COMPUTATION AND SAVE
# ============================================================================

def run_complete_analysis():
    """Run the complete analysis and save results."""
    t0 = time.time()
    
    print("=" * 80)
    print("  BCGP → CS/WRT GRAVITATIONAL NORMALIZATION FACTOR")
    print("  Complete Analysis")
    print("=" * 80)
    
    # Step 1: Verify D̃² ~ r³ for sl₂
    sl2_result = verify_D2_power_law_sl2()
    
    # Step 2: Verify D̃² ~ r¹⁰ for sl₃
    sl3_result = verify_D2_power_law_sl3()
    
    # Step 3: Verify Z_raw ~ r³ for sl₃
    zraw_result = verify_Z_raw_scaling_sl3()
    
    # Step 4: Test hypothesis with sl₄ D̃²
    sl4_result = verify_D2_power_law_sl4()
    
    # Step 5: Decompose discrepancy
    decompose_discrepancy()
    
    # Step 6: Verify normalization factor
    norm_result = verify_normalization_factor()
    
    elapsed = time.time() - t0
    print(f"\n  Total computation time: {elapsed:.1f} seconds")
    
    # Compile results
    results = {
        'sl2_D2_verification': {
            'power_fd': sl2_result['power_fd'],
            'power_3p': sl2_result['power_3param'],
            'target': 3.0,
            'match': sl2_result['match'],
        },
        'sl3_D2_verification': {
            'power_fd': sl3_result['power_total_fd'],
            'power_3p': sl3_result['power_total_3p'],
            'power_4p': sl3_result['power_total_4p'],
            'target': 10.0,
            'match': sl3_result['match'],
            'disc_cont_ratio': sl3_result['disc_cont_ratio'],
        },
        'sl3_Z_raw_verification': {
            'power_fd': zraw_result['power_fd'],
            'power_3p': zraw_result['power_3p'],
            'target': 3.0,
            'beta': zraw_result['beta'],
        },
        'sl4_D2_verification': {
            'power_fd': sl4_result.get('power_disc_fd', float('nan')),
            'power_3p': sl4_result.get('power_disc_3p', float('nan')),
            'target': 21.0,
            'note': 'Discrete sector only',
        },
        'analytical_formulas': {
            'D2_power': '(N-1)(2N-1)',
            'D2_excess': '(N-1)(N-2)',
            'Z_raw_power': '3(N-1)/2',
            'Z_raw_deficit': '(N-1)(N-2)/2',
            'normalization_power': '3(N-1)(N-2)/2',
            'cs_wrt_log_coeff': '-(N²-1)/2',
            'bcgp_log_coeff': '3(N-1)/2 - (N-1)(2N-1)',
        },
        'normalization_table': [],
        'computation_time': elapsed,
    }
    
    for row in norm_result:
        results['normalization_table'].append({
            'N': row['N'],
            'algebra': row['algebra'],
            'dim_G': row['dim_G'],
            'D2_power': row['D2_power'],
            'D2_excess': row['D2_excess'],
            'Z_raw_power': row['Z_raw_power'],
            'Z_raw_deficit': row['Z_raw_deficit'],
            'bcgp_log_coeff': row['bcgp_log_coeff'],
            'cs_wrt_log_coeff': row['cs_wrt_log_coeff'],
            'normalization_power': row['normalization_power'],
        })
    
    # Save results
    output_dir = "/home/z/my-project/download"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'bcgp_gravity_normalization.json')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")
    
    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_complete_analysis()
