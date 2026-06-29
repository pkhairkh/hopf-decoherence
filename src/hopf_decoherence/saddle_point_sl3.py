"""
Saddle Point Analysis for sl₃ BCGP Partition Function
=======================================================

Analyzes the saddle point structure of the sl₃ BCGP partition function to
understand WHY the continuous sector gives different scaling from sl₂.

KEY QUESTION:
  For sl₂: d̃(V_α) ≈ α/π (small near origin)
    → Z_bcgp_cont ~ ∫₀ʳ α e^{-β(α²-1)/(4r)} dα ~ r^{3/2}
    → Z_full_cont  ~ ∫₀ʳ r e^{-β(α²-1)/(4r)} dα ~ r^{3/2}
    → Ratio Z_bcgp/Z_full ~ 1 (same order, BCGP only slightly smaller due to α vs r prefactor)

  For sl₃: d̃(V_α) ~ sin(πα₁/r) sin(πα₂/r) sin(π(α₁+α₂)/r) / (r² sin⁴(π/r) sin²(2π/r))
    → At saddle point: d̃(V_α*) scales as a POWER of r (grows with r!)
    → This makes Z_bcgp_cont LARGER relative to Z_full_cont than in sl₂

WHY THE SHIFT CHANGES SIGN:
  sl₂: shift = +1/2 (full trace > |modified trace|)
  sl₃: shift = -1/2 (|modified trace| > full trace, because d̃(V_α*) >> r² at the saddle)

CORE FORMULAS:
  C₂(α₁,α₂) = (α₁² + α₂² + α₁α₂ + 2α₁ + 2α₂)/3
  h(α₁,α₂) = C₂(α₁,α₂)/r
  d̃(V_α) = sin(πα₁/r) sin(πα₂/r) sin(π(α₁+α₂)/r) / (r² sin⁴(π/r) sin²(2π/r))
  dim(V_α) = r²

TASKS:
  1. Compute Z_bcgp_cont and Z_full_cont for sl₃ at multiple r
  2. Find saddle point of integrand
  3. Evaluate d̃(V_α*) at saddle and determine r-scaling
  4. Compare with sl₂
  5. Compute ratio Z_bcgp_cont / Z_full_cont and verify scaling
"""

import numpy as np
from scipy import integrate, optimize
import json
import os
import warnings

warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================================
# CORE FORMULAS
# ============================================================================

def casimir_sl3(a, b):
    """Quadratic Casimir C₂(a,b) = (a² + b² + ab + 2a + 2b)/3."""
    return (a**2 + b**2 + a*b + 2*a + 2*b) / 3.0


def casimir_sl3_no_shift(a, b):
    """Casimir without ρ-shift: C₂(a,b) = (a² + b² + ab)/3.

    For typical modules V_α, the conformal weight uses this form
    in some conventions.
    """
    return (a**2 + b**2 + a*b) / 3.0


def conformal_weight_sl3(a, b, r):
    """Conformal weight h(a,b) = C₂(a,b)/r."""
    return casimir_sl3(a, b) / r


def modified_qdim_cont(a1, a2, r):
    """Modified quantum dimension d̃(V_α) for typical module V_(α₁,α₂).

    d̃(V_α) = sin(πα₁/r) sin(πα₂/r) sin(π(α₁+α₂)/r)
              / (r² sin⁴(π/r) sin²(2π/r))
    """
    num = (np.sin(np.pi * a1 / r) *
           np.sin(np.pi * a2 / r) *
           np.sin(np.pi * (a1 + a2) / r))
    den = r**2 * np.sin(np.pi / r)**4 * np.sin(2 * np.pi / r)**2
    return num / den


def modified_qdim_cont_abs(a1, a2, r):
    """Absolute value of modified quantum dimension |d̃(V_α)|."""
    return abs(modified_qdim_cont(a1, a2, r))


# ============================================================================
# sl₂ COMPARISON FORMULAS
# ============================================================================

def casimir_sl2(alpha):
    """Casimir for sl₂ continuous module V_α: C₂ = (α²-1)/4."""
    return (alpha**2 - 1) / 4.0


def conformal_weight_sl2(alpha, r):
    """Conformal weight for sl₂ typical module: h_α = (α²-1)/(4r)."""
    return casimir_sl2(alpha) / r


def modified_qdim_sl2(alpha, r):
    """Modified quantum dimension for sl₂ typical module V_α.

    d̃(V_α) = sin(πα/r) / (r sin²(π/r))
    """
    num = np.sin(np.pi * alpha / r)
    den = r * np.sin(np.pi / r)**2
    return num / den


# ============================================================================
# PARTITION FUNCTION COMPUTATION (sl₃)
# ============================================================================

def Z_full_cont_sl3(beta, r, use_triangular=True):
    """Full thermal trace — continuous sector for sl₃.

    Z_full_cont = ∫∫ r² × e^{-β C₂(α)/r} dα₁ dα₂

    Integration domain: triangular {α₁>0, α₂>0, α₁+α₂<r} (physically correct)
    or square [0,r]² for comparison.
    """
    def integrand(a2, a1):
        h = casimir_sl3(a1, a2) / r
        return r**2 * np.exp(-beta * h)

    if use_triangular:
        val, err = integrate.dblquad(
            integrand, 0, r, 0, lambda a1: r - a1,
            epsabs=1e-10, epsrel=1e-8
        )
    else:
        val, err = integrate.dblquad(
            integrand, 0, r, 0, r,
            epsabs=1e-10, epsrel=1e-8
        )
    return val


def Z_bcgp_cont_sl3(beta, r, use_triangular=True):
    """BCGP modified trace — continuous sector for sl₃.

    Z_bcgp_cont = ∫∫ d̃(V_α) × e^{-β C₂(α)/r} dα₁ dα₂

    Integration domain: triangular (where d̃(V_α) > 0).
    """
    def integrand(a2, a1):
        h = casimir_sl3(a1, a2) / r
        d = modified_qdim_cont(a1, a2, r)
        return d * np.exp(-beta * h)

    if use_triangular:
        val, err = integrate.dblquad(
            integrand, 0, r, 0, lambda a1: r - a1,
            epsabs=1e-10, epsrel=1e-8
        )
    else:
        val, err = integrate.dblquad(
            integrand, 0, r, 0, r,
            epsabs=1e-10, epsrel=1e-8
        )
    return val


# ============================================================================
# PARTITION FUNCTION COMPUTATION (sl₂)
# ============================================================================

def Z_full_cont_sl2(beta, r):
    """Full thermal trace — continuous sector for sl₂.

    Z_full_cont = ∫₀ʳ r × e^{-β(α²-1)/(4r)} dα
    """
    def integrand(alpha):
        h = conformal_weight_sl2(alpha, r)
        return r * np.exp(-beta * h)

    val, err = integrate.quad(integrand, 0, r, epsabs=1e-10, epsrel=1e-8)
    return val


def Z_bcgp_cont_sl2(beta, r):
    """BCGP modified trace — continuous sector for sl₂.

    Z_bcgp_cont = ∫₀ʳ d̃(V_α) × e^{-β(α²-1)/(4r)} dα
    """
    def integrand(alpha):
        h = conformal_weight_sl2(alpha, r)
        d = modified_qdim_sl2(alpha, r)
        return d * np.exp(-beta * h)

    val, err = integrate.quad(integrand, 0, r, epsabs=1e-10, epsrel=1e-8)
    return val


# ============================================================================
# SADDLE POINT ANALYSIS
# ============================================================================

def log_integrand_bcgp_sl3(x, r, beta):
    """Log of the BCGP integrand: ln[d̃(V_α) × e^{-β C₂/r}].

    x = [α₁, α₂]
    We maximize this to find the saddle point.

    Returns -log_integrand for minimization.
    """
    a1, a2 = x
    # Check domain
    if a1 <= 0 or a2 <= 0 or a1 + a2 >= r:
        return 1e30

    h = casimir_sl3(a1, a2) / r
    d = modified_qdim_cont(a1, a2, r)

    if abs(d) < 1e-30:
        return 1e30

    return -(np.log(abs(d)) - beta * h)


def log_integrand_bcgp_sq_sl3(x, r, beta):
    """Log of the BCGP integrand with d̃²: ln[d̃(V_α)² × e^{-β C₂/r}].

    x = [α₁, α₂]
    """
    a1, a2 = x
    if a1 <= 0 or a2 <= 0 or a1 + a2 >= r:
        return 1e30

    h = casimir_sl3(a1, a2) / r
    d = modified_qdim_cont(a1, a2, r)

    if abs(d) < 1e-30:
        return 1e30

    return -(2 * np.log(abs(d)) - beta * h)


def find_saddle_point_sl3(r, beta, use_dtilde_sq=False):
    """Find the saddle point of the BCGP integrand for sl₃.

    Uses multiple starting points to find the global maximum.

    Parameters
    ----------
    r : int
        Root of unity parameter.
    beta : float
        Inverse temperature.
    use_dtilde_sq : bool
        If True, find saddle of d̃² × e^{-βC₂/r}; otherwise d̃ × e^{-βC₂/r}.

    Returns
    -------
    dict
        Saddle point data including (α₁*, α₂*), d̃(V_α*), C₂(α*), etc.
    """
    func = log_integrand_bcgp_sq_sl3 if use_dtilde_sq else log_integrand_bcgp_sl3

    # Try multiple starting points in the triangular domain
    starts = [
        [r/6, r/6],
        [r/4, r/4],
        [r/3, r/3],
        [r/3, r/6],
        [r/6, r/3],
        [r/5, r/5],
        [0.1*r, 0.1*r],
        [0.4*r, 0.1*r],
        [0.1*r, 0.4*r],
    ]

    best_result = None
    best_val = float('inf')

    for x0 in starts:
        try:
            # Bounds: α₁ ∈ (ε, r-ε), α₂ ∈ (ε, r-ε), α₁+α₂ < r-ε
            eps = 1e-6 * r
            bounds = [(eps, r - 2*eps), (eps, r - 2*eps)]

            result = optimize.minimize(
                func, x0, args=(r, beta),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )

            # Check constraint α₁ + α₂ < r
            if result.x[0] + result.x[1] < r - eps and result.fun < best_val:
                best_val = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return None

    a1_star, a2_star = best_result.x
    d_star = modified_qdim_cont(a1_star, a2_star, r)
    c2_star = casimir_sl3(a1_star, a2_star)
    h_star = c2_star / r

    return {
        'alpha1_star': a1_star,
        'alpha2_star': a2_star,
        'alpha_star_sum': a1_star + a2_star,
        'alpha1_over_r': a1_star / r,
        'alpha2_over_r': a2_star / r,
        'd_tilde_star': d_star,
        'd_tilde_star_abs': abs(d_star),
        'ln_d_tilde_star': np.log(abs(d_star)),
        'c2_star': c2_star,
        'h_star': h_star,
        'log_integrand': -best_val,
        'success': best_result.success,
    }


def find_saddle_point_sl2(r, beta):
    """Find the saddle point of the BCGP integrand for sl₂.

    Maximizes ln[d̃(V_α) × e^{-β(α²-1)/(4r)}] over α ∈ (0, r).

    Returns
    -------
    dict
        Saddle point data.
    """
    def neg_log_integrand(alpha):
        if alpha <= 0 or alpha >= r:
            return 1e30
        h = conformal_weight_sl2(alpha, r)
        d = modified_qdim_sl2(alpha, r)
        if abs(d) < 1e-30:
            return 1e30
        return -(np.log(abs(d)) - beta * h)

    # Try multiple starting points
    starts = [0.1*r, 0.2*r, 0.3*r, 0.5*r, r/3, r/4, r/6]
    best_result = None
    best_val = float('inf')

    for x0 in starts:
        try:
            eps = 1e-6 * r
            result = optimize.minimize_scalar(
                neg_log_integrand, bounds=(eps, r - eps), method='bounded'
            )
            if result.fun < best_val:
                best_val = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return None

    alpha_star = best_result.x
    d_star = modified_qdim_sl2(alpha_star, r)
    h_star = conformal_weight_sl2(alpha_star, r)
    c2_star = casimir_sl2(alpha_star)

    return {
        'alpha_star': alpha_star,
        'alpha_over_r': alpha_star / r,
        'd_tilde_star': d_star,
        'ln_d_tilde_star': np.log(abs(d_star)),
        'c2_star': c2_star,
        'h_star': h_star,
        'log_integrand': -best_val,
    }


# ============================================================================
# ANALYTICAL ESTIMATES
# ============================================================================

def d_tilde_near_origin_sl3(a1, a2, r):
    """Approximate d̃(V_α) near origin using small-argument expansion.

    sin(πα₁/r) ~ πα₁/r, etc.
    sin(π/r) ~ π/r, sin(2π/r) ~ 2π/r

    d̃(V_α) ~ π³α₁α₂(α₁+α₂)/r³ / (r² × π⁴/r⁴ × 4π²/r²)
            = π³α₁α₂(α₁+α₂)/r³ / (4π⁶/r⁴)
            = α₁α₂(α₁+α₂) × r / (4π³)

    So d̃(V_α) ~ r × α₁α₂(α₁+α₂) / (4π³) near origin.
    This GROWS with r (unlike sl₂ where d̃ ~ α/π is r-independent).
    """
    return a1 * a2 * (a1 + a2) * r / (4 * np.pi**3)


def d_tilde_at_symmetric_point(r):
    """d̃(V_α) at the symmetric point α = (r/3, r/3).

    sin(π/3) = √3/2, sin(2π/3) = √3/2

    d̃ = (√3/2)³ / (r² × sin⁴(π/r) × sin²(2π/r))

    For large r: ~ (3√3/8) / (4π⁶/r⁴) = (3√3/8) × r⁴/(4π⁶) ~ r⁴
    """
    a1 = a2 = r / 3.0
    return modified_qdim_cont(a1, a2, r)


def d_tilde_at_symmetric_point_asymp(r):
    """Asymptotic formula for d̃ at α = (r/3, r/3) for large r."""
    return (3 * np.sqrt(3) / 8) * r**4 / (4 * np.pi**6)


# ============================================================================
# D̃² CONTINUOUS (for normalization)
# ============================================================================

def D_tilde_squared_cont_sl3(r):
    """Continuous part of modified global dimension: ∫∫ d̃(V_α)² dα₁ dα₂."""
    def integrand(a2, a1):
        d = modified_qdim_cont(a1, a2, r)
        return d * d

    val, err = integrate.dblquad(
        integrand, 0, r, 0, lambda a1: r - a1,
        epsabs=1e-10, epsrel=1e-8
    )
    return val


# ============================================================================
# COMPREHENSIVE ANALYSIS
# ============================================================================

def compute_saddle_point_analysis(
    r_values=None,
    beta=1.0,
    output_dir="/home/z/my-project/download"
):
    """Run the full saddle point analysis for sl₃ (and sl₂ comparison).

    Parameters
    ----------
    r_values : list of int
        Root of unity parameters.
    beta : float
        Inverse temperature.
    output_dir : str
        Where to save results.
    """
    if r_values is None:
        r_values = [5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31]

    print("=" * 110)
    print("  SADDLE POINT ANALYSIS FOR sl₃ BCGP PARTITION FUNCTION")
    print("  Understanding WHY the continuous sector gives different scaling from sl₂")
    print(f"  β = {beta}")
    print("=" * 110)

    # ======================================================================
    # STEP 1: Compute partition functions and saddle points for sl₃
    # ======================================================================
    print(f"\n{'─' * 110}")
    print(f"  STEP 1: PARTITION FUNCTIONS AND SADDLE POINTS (sl₃)")
    print(f"{'─' * 110}")

    sl3_data = []
    for r in r_values:
        print(f"  Computing r={r}...", end=" ", flush=True)
        try:
            # Partition functions
            Zfc = Z_full_cont_sl3(beta, r)
            Zmc = Z_bcgp_cont_sl3(beta, r)

            # Saddle points
            saddle_lin = find_saddle_point_sl3(r, beta, use_dtilde_sq=False)
            saddle_sq = find_saddle_point_sl3(r, beta, use_dtilde_sq=True)

            # d̃ at symmetric point
            d_sym = d_tilde_at_symmetric_point(r)

            # d̃ at origin (small α)
            a1_test, a2_test = 0.01 * r, 0.01 * r
            d_near_origin = modified_qdim_cont(a1_test, a2_test, r)
            d_near_origin_approx = d_tilde_near_origin_sl3(a1_test, a2_test, r)

            res = {
                'r': r,
                'beta': beta,
                'Z_full_cont': Zfc,
                'Z_bcgp_cont': Zmc,
                'ratio_Zbcgp_Zfull': abs(Zmc) / Zfc if Zfc > 0 else float('nan'),
                'ln_Z_full_cont': np.log(Zfc) if Zfc > 0 else float('-inf'),
                'ln_Z_bcgp_cont': np.log(abs(Zmc)) if abs(Zmc) > 1e-300 else float('-inf'),
                # Saddle point (linear integrand)
                'saddle_a1': saddle_lin['alpha1_star'] if saddle_lin else float('nan'),
                'saddle_a2': saddle_lin['alpha2_star'] if saddle_lin else float('nan'),
                'saddle_a1_over_r': saddle_lin['alpha1_over_r'] if saddle_lin else float('nan'),
                'saddle_a2_over_r': saddle_lin['alpha2_over_r'] if saddle_lin else float('nan'),
                'saddle_d_tilde': saddle_lin['d_tilde_star_abs'] if saddle_lin else float('nan'),
                'saddle_ln_d_tilde': saddle_lin['ln_d_tilde_star'] if saddle_lin else float('nan'),
                'saddle_c2': saddle_lin['c2_star'] if saddle_lin else float('nan'),
                'saddle_h': saddle_lin['h_star'] if saddle_lin else float('nan'),
                # Saddle point (d̃² integrand)
                'saddle_sq_a1': saddle_sq['alpha1_star'] if saddle_sq else float('nan'),
                'saddle_sq_a2': saddle_sq['alpha2_star'] if saddle_sq else float('nan'),
                'saddle_sq_a1_over_r': saddle_sq['alpha1_over_r'] if saddle_sq else float('nan'),
                'saddle_sq_a2_over_r': saddle_sq['alpha2_over_r'] if saddle_sq else float('nan'),
                'saddle_sq_d_tilde': saddle_sq['d_tilde_star_abs'] if saddle_sq else float('nan'),
                'saddle_sq_ln_d_tilde': saddle_sq['ln_d_tilde_star'] if saddle_sq else float('nan'),
                # Symmetric point
                'd_tilde_symmetric': d_sym,
                'ln_d_tilde_symmetric': np.log(abs(d_sym)) if abs(d_sym) > 1e-300 else float('-inf'),
                # Near origin
                'd_tilde_near_origin': d_near_origin,
                'd_tilde_near_origin_approx': d_near_origin_approx,
            }
            sl3_data.append(res)
            print(f"Z_full={Zfc:.4e}, Z_bcgp={Zmc:.4e}, ratio={res['ratio_Zbcgp_Zfull']:.6f}, "
                  f"saddle=({res['saddle_a1_over_r']:.3f}, {res['saddle_a2_over_r']:.3f})r, "
                  f"d̃*={res['saddle_d_tilde']:.4e}")
        except Exception as e:
            print(f"FAILED: {e}")

    # ======================================================================
    # STEP 2: Compute partition functions and saddle points for sl₂
    # ======================================================================
    print(f"\n{'─' * 110}")
    print(f"  STEP 2: PARTITION FUNCTIONS AND SADDLE POINTS (sl₂)")
    print(f"{'─' * 110}")

    sl2_data = []
    for r in r_values:
        print(f"  Computing r={r}...", end=" ", flush=True)
        try:
            Zfc = Z_full_cont_sl2(beta, r)
            Zmc = Z_bcgp_cont_sl2(beta, r)
            saddle = find_saddle_point_sl2(r, beta)

            # d̃ at a reference point (small α)
            alpha_test = 0.01 * r
            d_near_origin_sl2 = modified_qdim_sl2(alpha_test, r)
            # Near origin: d̃ ~ α/π (independent of r)
            d_near_origin_approx_sl2 = alpha_test / np.pi

            res = {
                'r': r,
                'beta': beta,
                'Z_full_cont': Zfc,
                'Z_bcgp_cont': Zmc,
                'ratio_Zbcgp_Zfull': abs(Zmc) / Zfc if Zfc > 0 else float('nan'),
                'ln_Z_full_cont': np.log(Zfc) if Zfc > 0 else float('-inf'),
                'ln_Z_bcgp_cont': np.log(abs(Zmc)) if abs(Zmc) > 1e-300 else float('-inf'),
                # Saddle point
                'saddle_alpha': saddle['alpha_star'] if saddle else float('nan'),
                'saddle_alpha_over_r': saddle['alpha_over_r'] if saddle else float('nan'),
                'saddle_d_tilde': saddle['d_tilde_star'] if saddle else float('nan'),
                'saddle_ln_d_tilde': saddle['ln_d_tilde_star'] if saddle else float('nan'),
                'saddle_c2': saddle['c2_star'] if saddle else float('nan'),
                'saddle_h': saddle['h_star'] if saddle else float('nan'),
                # Near origin
                'd_tilde_near_origin': d_near_origin_sl2,
                'd_tilde_near_origin_approx': d_near_origin_approx_sl2,
            }
            sl2_data.append(res)
            print(f"Z_full={Zfc:.4e}, Z_bcgp={Zmc:.4e}, ratio={res['ratio_Zbcgp_Zfull']:.6f}, "
                  f"saddle={res['saddle_alpha_over_r']:.3f}r, "
                  f"d̃*={res['saddle_d_tilde']:.4e}")
        except Exception as e:
            print(f"FAILED: {e}")

    # ======================================================================
    # STEP 3: d̃(V_α*) scaling analysis
    # ======================================================================
    print(f"\n{'─' * 110}")
    print(f"  STEP 3: d̃(V_α*) SCALING ANALYSIS")
    print(f"{'─' * 110}")

    if len(sl3_data) >= 3:
        r_arr = np.array([d['r'] for d in sl3_data], dtype=float)

        # Fit ln(d̃*) = p × ln(r) + c + d/r for sl₃ saddle (linear integrand)
        ln_dtilde_star = np.array([d['saddle_ln_d_tilde'] for d in sl3_data])
        valid = np.isfinite(ln_dtilde_star)
        if sum(valid) >= 3:
            A = np.column_stack([np.log(r_arr[valid]), np.ones(sum(valid)), 1.0/r_arr[valid]])
            c_sl3, _, _, _ = np.linalg.lstsq(A, ln_dtilde_star[valid], rcond=None)
            p_sl3 = c_sl3[0]
            print(f"\n  sl₃ saddle point d̃(V_α*) scaling:")
            print(f"    ln(d̃*) = {p_sl3:.4f} × ln(r) + {c_sl3[1]:.4f} + {c_sl3[2]:.4f}/r")
            print(f"    d̃(V_α*) ~ r^{p_sl3:.4f}")

        # Fit ln(d̃*) = p × ln(r) + c for sl₃ saddle (d̃² integrand)
        ln_dtilde_sq_star = np.array([d['saddle_sq_ln_d_tilde'] for d in sl3_data])
        valid_sq = np.isfinite(ln_dtilde_sq_star)
        if sum(valid_sq) >= 3:
            A = np.column_stack([np.log(r_arr[valid_sq]), np.ones(sum(valid_sq)), 1.0/r_arr[valid_sq]])
            c_sl3_sq, _, _, _ = np.linalg.lstsq(A, ln_dtilde_sq_star[valid_sq], rcond=None)
            p_sl3_sq = c_sl3_sq[0]
            print(f"\n  sl₃ saddle point d̃(V_α*) scaling (from d̃² integrand):")
            print(f"    ln(d̃*) = {p_sl3_sq:.4f} × ln(r) + {c_sl3_sq[1]:.4f}")
            print(f"    d̃(V_α*) ~ r^{p_sl3_sq:.4f}")

        # Fit ln(d̃_symmetric) scaling
        ln_d_sym = np.array([d['ln_d_tilde_symmetric'] for d in sl3_data])
        valid_sym = np.isfinite(ln_d_sym)
        if sum(valid_sym) >= 3:
            A = np.column_stack([np.log(r_arr[valid_sym]), np.ones(sum(valid_sym)), 1.0/r_arr[valid_sym]])
            c_sym, _, _, _ = np.linalg.lstsq(A, ln_d_sym[valid_sym], rcond=None)
            p_sym = c_sym[0]
            print(f"\n  sl₃ d̃ at symmetric point α=(r/3, r/3):")
            print(f"    ln(d̃_sym) = {p_sym:.4f} × ln(r) + {c_sym[1]:.4f}")
            print(f"    d̃_sym ~ r^{p_sym:.4f}")

        # Fit for sl₂
        if len(sl2_data) >= 3:
            r_arr_sl2 = np.array([d['r'] for d in sl2_data], dtype=float)
            ln_dtilde_star_sl2 = np.array([d['saddle_ln_d_tilde'] for d in sl2_data])
            valid_sl2 = np.isfinite(ln_dtilde_star_sl2)
            if sum(valid_sl2) >= 3:
                A = np.column_stack([np.log(r_arr_sl2[valid_sl2]), np.ones(sum(valid_sl2)), 1.0/r_arr_sl2[valid_sl2]])
                c_sl2, _, _, _ = np.linalg.lstsq(A, ln_dtilde_star_sl2[valid_sl2], rcond=None)
                p_sl2 = c_sl2[0]
                print(f"\n  sl₂ saddle point d̃(V_α*) scaling:")
                print(f"    ln(d̃*) = {p_sl2:.4f} × ln(r) + {c_sl2[1]:.4f}")
                print(f"    d̃(V_α*) ~ r^{p_sl2:.4f}")
                print(f"    (Expected: d̃ ~ α/π independent of r near origin, but at saddle α* ~ √r, so d̃* ~ √r)")

    # ======================================================================
    # STEP 4: Saddle point location analysis
    # ======================================================================
    print(f"\n{'─' * 110}")
    print(f"  STEP 4: SADDLE POINT LOCATION ANALYSIS")
    print(f"{'─' * 110}")

    print(f"\n  sl₃ saddle point positions (linear integrand):")
    print(f"  {'r':>4s}  {'α₁*/r':>8s}  {'α₂*/r':>8s}  {'(α₁*+α₂*)/r':>12s}  {'C₂(α*)':>10s}  {'h(α*)':>8s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*8}")
    for d in sl3_data:
        print(f"  {d['r']:4d}  {d['saddle_a1_over_r']:8.4f}  {d['saddle_a2_over_r']:8.4f}  "
              f"{d['saddle_a1_over_r']+d['saddle_a2_over_r']:12.4f}  "
              f"{d['saddle_c2']:10.4f}  {d['saddle_h']:8.4f}")

    print(f"\n  sl₃ saddle point positions (d̃² integrand):")
    print(f"  {'r':>4s}  {'α₁*/r':>8s}  {'α₂*/r':>8s}  {'(α₁*+α₂*)/r':>12s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*12}")
    for d in sl3_data:
        print(f"  {d['r']:4d}  {d['saddle_sq_a1_over_r']:8.4f}  {d['saddle_sq_a2_over_r']:8.4f}  "
              f"{d['saddle_sq_a1_over_r']+d['saddle_sq_a2_over_r']:12.4f}")

    print(f"\n  sl₂ saddle point positions:")
    print(f"  {'r':>4s}  {'α*/r':>8s}  {'C₂(α*)':>10s}  {'h(α*)':>8s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*8}")
    for d in sl2_data:
        print(f"  {d['r']:4d}  {d['saddle_alpha_over_r']:8.4f}  {d['saddle_c2']:10.4f}  {d['saddle_h']:8.4f}")

    # ======================================================================
    # STEP 5: Partition function ratio analysis (KEY RESULT)
    # ======================================================================
    print(f"\n{'─' * 110}")
    print(f"  STEP 5: PARTITION FUNCTION RATIO (KEY RESULT)")
    print(f"{'─' * 110}")

    print(f"\n  sl₃: Z_bcgp_cont / Z_full_cont ratio:")
    print(f"  {'r':>4s}  {'Z_full_cont':>14s}  {'Z_bcgp_cont':>14s}  {'ratio':>10s}  {'ln(ratio)':>10s}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*10}")
    for d in sl3_data:
        ln_ratio = np.log(d['ratio_Zbcgp_Zfull']) if d['ratio_Zbcgp_Zfull'] > 0 else float('-inf')
        print(f"  {d['r']:4d}  {d['Z_full_cont']:14.4e}  {d['Z_bcgp_cont']:14.4e}  "
              f"{d['ratio_Zbcgp_Zfull']:10.6f}  {ln_ratio:10.4f}")

    print(f"\n  sl₂: Z_bcgp_cont / Z_full_cont ratio:")
    print(f"  {'r':>4s}  {'Z_full_cont':>14s}  {'Z_bcgp_cont':>14s}  {'ratio':>10s}  {'ln(ratio)':>10s}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*10}")
    for d in sl2_data:
        ln_ratio = np.log(d['ratio_Zbcgp_Zfull']) if d['ratio_Zbcgp_Zfull'] > 0 else float('-inf')
        print(f"  {d['r']:4d}  {d['Z_full_cont']:14.4e}  {d['Z_bcgp_cont']:14.4e}  "
              f"{d['ratio_Zbcgp_Zfull']:10.6f}  {ln_ratio:10.4f}")

    # Fit the ratio scaling
    if len(sl3_data) >= 3:
        r_arr = np.array([d['r'] for d in sl3_data], dtype=float)
        ratios_sl3 = np.array([d['ratio_Zbcgp_Zfull'] for d in sl3_data])
        valid_r = ratios_sl3 > 0
        if sum(valid_r) >= 3:
            ln_ratios = np.log(ratios_sl3[valid_r])
            A = np.column_stack([np.log(r_arr[valid_r]), np.ones(sum(valid_r)), 1.0/r_arr[valid_r]])
            c_ratio, _, _, _ = np.linalg.lstsq(A, ln_ratios, rcond=None)
            p_ratio_sl3 = c_ratio[0]
            print(f"\n  sl₃ ratio scaling: Z_bcgp/Z_full ~ r^{p_ratio_sl3:.4f}")

    if len(sl2_data) >= 3:
        r_arr_sl2 = np.array([d['r'] for d in sl2_data], dtype=float)
        ratios_sl2 = np.array([d['ratio_Zbcgp_Zfull'] for d in sl2_data])
        valid_r2 = ratios_sl2 > 0
        if sum(valid_r2) >= 3:
            ln_ratios2 = np.log(ratios_sl2[valid_r2])
            A = np.column_stack([np.log(r_arr_sl2[valid_r2]), np.ones(sum(valid_r2)), 1.0/r_arr_sl2[valid_r2]])
            c_ratio2, _, _, _ = np.linalg.lstsq(A, ln_ratios2, rcond=None)
            p_ratio_sl2 = c_ratio2[0]
            print(f"  sl₂ ratio scaling: Z_bcgp/Z_full ~ r^{p_ratio_sl2:.4f}")

    # ======================================================================
    # STEP 6: Partition function scaling (unnormalized)
    # ======================================================================
    print(f"\n{'─' * 110}")
    print(f"  STEP 6: UNNORMALIZED PARTITION FUNCTION SCALING")
    print(f"{'─' * 110}")

    if len(sl3_data) >= 3:
        r_arr = np.array([d['r'] for d in sl3_data], dtype=float)

        # Z_full_cont scaling
        ln_Zfc = np.array([d['ln_Z_full_cont'] for d in sl3_data])
        valid = np.isfinite(ln_Zfc)
        if sum(valid) >= 3:
            A = np.column_stack([np.log(r_arr[valid]), np.ones(sum(valid)), 1.0/r_arr[valid]])
            c_zfc, _, _, _ = np.linalg.lstsq(A, ln_Zfc[valid], rcond=None)
            print(f"\n  sl₃ Z_full_cont ~ r^{c_zfc[0]:.4f}  (expected: r^3)")

        # Z_bcgp_cont scaling
        ln_Zmc = np.array([d['ln_Z_bcgp_cont'] for d in sl3_data])
        valid = np.isfinite(ln_Zmc)
        if sum(valid) >= 3:
            A = np.column_stack([np.log(r_arr[valid]), np.ones(sum(valid)), 1.0/r_arr[valid]])
            c_zmc, _, _, _ = np.linalg.lstsq(A, ln_Zmc[valid], rcond=None)
            print(f"  sl₃ Z_bcgp_cont ~ r^{c_zmc[0]:.4f}  (expected: r^{{7/2}} = r^3.5 if d̃*~r⁴)")

    if len(sl2_data) >= 3:
        r_arr_sl2 = np.array([d['r'] for d in sl2_data], dtype=float)

        ln_Zfc2 = np.array([d['ln_Z_full_cont'] for d in sl2_data])
        valid = np.isfinite(ln_Zfc2)
        if sum(valid) >= 3:
            A = np.column_stack([np.log(r_arr_sl2[valid]), np.ones(sum(valid)), 1.0/r_arr_sl2[valid]])
            c_zfc2, _, _, _ = np.linalg.lstsq(A, ln_Zfc2[valid], rcond=None)
            print(f"\n  sl₂ Z_full_cont ~ r^{c_zfc2[0]:.4f}  (expected: r^{{3/2}})")

        ln_Zmc2 = np.array([d['ln_Z_bcgp_cont'] for d in sl2_data])
        valid = np.isfinite(ln_Zmc2)
        if sum(valid) >= 3:
            A = np.column_stack([np.log(r_arr_sl2[valid]), np.ones(sum(valid)), 1.0/r_arr_sl2[valid]])
            c_zmc2, _, _, _ = np.linalg.lstsq(A, ln_Zmc2[valid], rcond=None)
            print(f"  sl₂ Z_bcgp_cont ~ r^{c_zmc2[0]:.4f}  (expected: r^{{3/2}})")

    # ======================================================================
    # STEP 7: d̃(V_α) landscape analysis
    # ======================================================================
    print(f"\n{'─' * 110}")
    print(f"  STEP 7: d̃(V_α) LANDSCAPE — COMPARISON AT KEY POINTS")
    print(f"{'─' * 110}")

    print(f"\n  sl₃ d̃(V_α) at various points:")
    print(f"  {'r':>4s}  {'d̃(origin)':>12s}  {'d̃(r/3,r/3)':>12s}  {'d̃(saddle)':>12s}  "
          f"{'r²':>8s}  {'d̃*/r²':>10s}  {'d̃_sym/r²':>10s}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*10}")
    for d in sl3_data:
        r2 = d['r']**2
        print(f"  {d['r']:4d}  {d['d_tilde_near_origin']:12.4e}  {d['d_tilde_symmetric']:12.4e}  "
              f"{d['saddle_d_tilde']:12.4e}  {r2:8d}  "
              f"{d['saddle_d_tilde']/r2:10.4e}  {d['d_tilde_symmetric']/r2:10.4e}")

    print(f"\n  sl₂ d̃(V_α) at various points:")
    print(f"  {'r':>4s}  {'d̃(0.01r)':>12s}  {'d̃(saddle)':>12s}  {'r':>8s}  {'d̃*/r':>10s}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*10}")
    for d in sl2_data:
        print(f"  {d['r']:4d}  {d['d_tilde_near_origin']:12.4e}  {d['saddle_d_tilde']:12.4e}  "
              f"{d['r']:8d}  {d['saddle_d_tilde']/d['r']:10.4e}")

    # ======================================================================
    # STEP 8: Gaussian approximation analysis
    # ======================================================================
    print(f"\n{'─' * 110}")
    print(f"  STEP 8: GAUSSIAN APPROXIMATION AT SADDLE POINT")
    print(f"{'─' * 110}")

    print("""
  For sl₂: The integrand d̃(V_α) × e^{-βC₂(α)/r} has:
    - d̃(V_α) ≈ α/π (linear near origin, SMALL compared to r)
    - Saddle at α* ~ O(√r) where Gaussian e^{-βC₂/r} peaks
    - d̃(V_α*) ~ √r/π, while dim(V_α*) = r
    - Ratio d̃*/r ~ 1/(π√r) → 0 as r → ∞
    - So Z_bcgp_cont < Z_full_cont (modified trace UNDERCOUNTS)

  For sl₃: The integrand d̃(V_α) × e^{-βC₂(α)/r} has:
    - d̃(V_α) grows with r (even near origin: d̃ ~ r × α₁α₂(α₁+α₂))
    - At symmetric point (r/3, r/3): d̃ ~ r⁴ while dim(V_α) = r²
    - Ratio d̃/r² ~ r² → ∞ as r → ∞
    - So Z_bcgp_cont > Z_full_cont (modified trace OVERCOUNTS)

  This is WHY the shift changes sign from +1/2 (sl₂) to -1/2 (sl₃)!
""")

    # ======================================================================
    # STEP 9: Numerical verification of the Gaussian width
    # ======================================================================
    print(f"\n{'─' * 110}")
    print(f"  STEP 9: GAUSSIAN WIDTH ANALYSIS")
    print(f"{'─' * 110}")

    # For the Gaussian e^{-βC₂(α)/r} around the saddle:
    # C₂(α) ≈ C₂(α*) + ½ ∂²C₂/∂αᵢ∂αⱼ (α-α*)ᵢ(α-α*)ⱼ
    # The Hessian of C₂/r at the saddle determines the Gaussian width

    print(f"\n  sl₃ Hessian of C₂/r at saddle point:")
    print(f"  {'r':>4s}  {'λ₁':>10s}  {'λ₂':>10s}  {'width₁':>10s}  {'width₂':>10s}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for d in sl3_data:
        r = d['r']
        a1, a2 = d['saddle_a1'], d['saddle_a2']
        if np.isnan(a1) or np.isnan(a2):
            continue

        # Hessian of C₂/r = (1/r) × Hessian(C₂)
        # C₂ = (α₁² + α₂² + α₁α₂ + 2α₁ + 2α₂)/3
        # ∂²C₂/∂α₁² = 2/3
        # ∂²C₂/∂α₂² = 2/3
        # ∂²C₂/∂α₁∂α₂ = 1/3
        H = np.array([[2/3, 1/3], [1/3, 2/3]]) / r

        # Eigenvalues of the Hessian
        eigvals = np.linalg.eigvalsh(H)

        # Gaussian widths (σ = 1/√(β × eigenvalue))
        widths = 1.0 / np.sqrt(beta * eigvals)

        print(f"  {r:4d}  {eigvals[0]:10.6f}  {eigvals[1]:10.6f}  "
              f"{widths[0]:10.4f}  {widths[1]:10.4f}")

    print(f"\n  Note: Hessian of C₂ is CONSTANT (independent of α)")
    print(f"  → C₂ is a QUADRATIC form, so the Gaussian approximation is EXACT")
    print(f"  → Eigenvalues: (2/3 ± 1/3)/r = 1/(3r) and 1/r")
    print(f"  → Widths: √(3r/β) and √(r/β)")

    # ======================================================================
    # STEP 10: Exact Gaussian integral comparison
    # ======================================================================
    print(f"\n{'─' * 110}")
    print(f"  STEP 10: EXACT GAUSSIAN INTEGRAL PREDICTIONS")
    print(f"{'─' * 110}")

    print("""
  For sl₃, C₂(α) is exactly quadratic, so the Gaussian integral can be
  computed analytically. The full integral over the triangular domain
  {α₁ > 0, α₂ > 0, α₁ + α₂ < r} can be approximated by extending
  to all of ℝ² when the width √r is much smaller than r.

  Full trace integrand: r² × e^{-βC₂(α)/r}
  → Z_full_cont ≈ r² × ∫∫_{ℝ²} e^{-βC₂(α)/r} dα₁ dα₂
  = r² × 2π / √(det(βH/r))  where H is the C₂ Hessian
  = r² × 2π / √(β²/r² × det(H))
  = r² × 2πr / (β × √(det(H)))

  det(H) = (2/3)² - (1/3)² = 4/9 - 1/9 = 3/9 = 1/3
  √(det(H)) = 1/√3

  Z_full_cont ≈ r² × 2πr / (β/√3) = 2π√3 r³/β

  For BCGP, the integrand is d̃(V_α) × e^{-βC₂(α)/r}. If d̃(V_α) is
  slowly varying near the saddle (which it is, since it's bounded by O(1)
  in the sin factors), then:
  Z_bcgp_cont ≈ d̃(α*) × ∫∫ e^{-βC₂/r} dα × (volume correction)

  But d̃ is NOT slowly varying — it has structure from the sin factors.
  The key question is whether d̃(V_α) at the saddle dominates the integral.
""")

    # Compute exact Gaussian integral for comparison
    det_H = 1.0 / 3.0  # det of C₂ Hessian
    sqrt_det_H = np.sqrt(det_H)

    print(f"  Exact Gaussian integrals (full ℝ²):")
    print(f"  {'r':>4s}  {'Z_full_num':>14s}  {'Z_full_Gauss':>14s}  {'ratio':>8s}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*8}")
    for d in sl3_data:
        r = d['r']
        Z_gauss = r**2 * 2 * np.pi * r / (beta * sqrt_det_H)
        ratio = d['Z_full_cont'] / Z_gauss if Z_gauss > 0 else float('nan')
        print(f"  {r:4d}  {d['Z_full_cont']:14.4e}  {Z_gauss:14.4e}  {ratio:8.4f}")

    # ======================================================================
    # FINAL SUMMARY
    # ======================================================================
    print(f"\n{'═' * 110}")
    print(f"  FINAL SUMMARY: SADDLE POINT ANALYSIS FOR sl₃")
    print(f"{'═' * 110}")

    # Collect key results
    p_dtilde_sl3 = p_sl3 if 'p_sl3' in dir() else float('nan')
    p_dtilde_sq_sl3 = p_sl3_sq if 'p_sl3_sq' in dir() else float('nan')
    p_dtilde_sym = p_sym if 'p_sym' in dir() else float('nan')
    p_dtilde_sl2 = p_sl2 if 'p_sl2' in dir() else float('nan')
    p_ratio_sl3_val = p_ratio_sl3 if 'p_ratio_sl3' in dir() else float('nan')
    p_ratio_sl2_val = p_ratio_sl2 if 'p_ratio_sl2' in dir() else float('nan')

    print(f"""
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  KEY FINDING: d̃(V_α*) SCALING AT THE SADDLE POINT                         │
  │                                                                              │
  │  sl₂: d̃(V_α*) ~ r^{p_dtilde_sl2:.2f}  (expected ~ r^0.5)                 │
  │        At saddle: d̃*/r → 0 (modified trace UNDERCOUNTS)                    │
  │                                                                              │
  │  sl₃: d̃(V_α*) ~ r^{p_dtilde_sl3:.2f}  (expected ~ r^4)                   │
  │        At saddle: d̃*/r² ~ r² → ∞ (modified trace OVERCOUNTS)              │
  │                                                                              │
  │  At symmetric point: d̃_sym ~ r^{p_dtilde_sym:.2f} (expected ~ r^4)        │
  │                                                                              │
  │  RATIO SCALING:                                                              │
  │  sl₃: Z_bcgp_cont/Z_full_cont ~ r^{p_ratio_sl3_val:.2f} (expected ~ r²)   │
  │  sl₂: Z_bcgp_cont/Z_full_cont ~ r^{p_ratio_sl2_val:.2f} (expected ~ r^0)  │
  │                                                                              │
  │  MECHANISM:                                                                  │
  │  • sl₂: d̃(V_α) ≈ α/π (small, O(1) near origin) → UNDERCOUNTS             │
  │  • sl₃: d̃(V_α) ~ r × α₁α₂(α₁+α₂) (grows WITH r) → OVERCOUNTS            │
  │                                                                              │
  │  The sin factors in d̃(V_α) are O(1) at α ~ r/3, while the denominator    │
  │  r² sin⁴(π/r) sin²(2π/r) ~ r² × (π/r)⁴ × (2π/r)² ~ 1/r⁴              │
  │  So d̃(V_α) ~ r⁴ × const at the symmetric point.                           │
  │                                                                              │
  │  This is WHY the shift changes sign:                                         │
  │  sl₂: shift = +1/2 (full trace > |modified trace|)                         │
  │  sl₃: shift = -1/2 (|modified trace| > full trace)                         │
  └──────────────────────────────────────────────────────────────────────────────┘
""")

    # ======================================================================
    # SAVE RESULTS
    # ======================================================================
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'beta': beta,
        'r_values': r_values,
        'sl3_data': sl3_data,
        'sl2_data': sl2_data,
        'scaling_exponents': {
            'sl3_d_tilde_at_saddle': float(p_dtilde_sl3) if np.isfinite(p_dtilde_sl3) else None,
            'sl3_d_tilde_at_saddle_sq': float(p_dtilde_sq_sl3) if np.isfinite(p_dtilde_sq_sl3) else None,
            'sl3_d_tilde_at_symmetric': float(p_dtilde_sym) if np.isfinite(p_dtilde_sym) else None,
            'sl2_d_tilde_at_saddle': float(p_dtilde_sl2) if np.isfinite(p_dtilde_sl2) else None,
            'sl3_ratio_bcgp_full': float(p_ratio_sl3_val) if np.isfinite(p_ratio_sl3_val) else None,
            'sl2_ratio_bcgp_full': float(p_ratio_sl2_val) if np.isfinite(p_ratio_sl2_val) else None,
        },
        'analytical_predictions': {
            'sl3_d_tilde_at_symmetric_point': 'r^4 (from sin(π/3)³/r²sin⁴(π/r)sin²(2π/r) ~ r⁴)',
            'sl2_d_tilde_at_saddle': 'r^{1/2} (from α*/π with α* ~ √r)',
            'sl3_Z_full_cont': 'r³ (from r² × Gaussian 2D integral)',
            'sl3_Z_bcgp_cont': 'r^{7/2} (from r⁴ × Gaussian 2D integral / r, or r⁴ × r^{-1/2})',
            'sl3_ratio_prediction': 'r² (from r^{7/2}/r³)',
        },
    }

    # Clean numpy types for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    results_clean = clean_for_json(results)

    output_path = os.path.join(output_dir, 'saddle_point_sl3.json')
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")

    return results_clean


# ============================================================================
# ADDITIONAL ANALYSIS: d̃(V_α) profile along α₁ = α₂
# ============================================================================

def d_tilde_profile_sl3(r, n_points=200, beta=1.0):
    """Compute d̃(V_α) and the BCGP integrand along the α₁ = α₂ line.

    This shows the structure of the integrand as a function of the
    single parameter t = α₁/r = α₂/r ∈ (0, 0.5) (since α₁+α₂ < r).
    """
    t_values = np.linspace(0.001, 0.499, n_points)
    d_tilde_vals = []
    integrand_vals = []
    full_integrand_vals = []

    for t in t_values:
        a1 = a2 = t * r
        d = modified_qdim_cont(a1, a2, r)
        h = casimir_sl3(a1, a2) / r

        d_tilde_vals.append(d)
        integrand_vals.append(d * np.exp(-beta * h))
        full_integrand_vals.append(r**2 * np.exp(-beta * h))

    return {
        'r': r,
        't_values': t_values.tolist(),
        'd_tilde': d_tilde_vals,
        'bcgp_integrand': integrand_vals,
        'full_integrand': full_integrand_vals,
        'max_d_tilde_t': t_values[np.argmax(np.abs(d_tilde_vals))],
        'max_bcgp_integrand_t': t_values[np.argmax(np.abs(integrand_vals))],
    }


# ============================================================================
# D̃² AND NORMALIZED PARTITION FUNCTION ANALYSIS
# ============================================================================

def normalized_partition_analysis(
    r_values=None,
    beta=1.0,
    output_dir="/home/z/my-project/download"
):
    """Compute normalized partition functions Z/D̃² and extract log coefficients.

    This is the analysis that directly shows the sign change of the shift.
    """
    if r_values is None:
        r_values = [5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31]

    print(f"\n{'═' * 110}")
    print(f"  NORMALIZED PARTITION FUNCTION ANALYSIS (Z_cont / D̃²)")
    print(f"{'═' * 110}")

    # Import D̃² from sl3_master_identity
    from .sl3_master_identity import (
        D_tilde_squared_disc,
        D_tilde_squared_cont,
        Z_full_disc,
        Z_bcgp_disc,
    )

    per_r = []
    for r in r_values:
        print(f"  Computing r={r}...", end=" ", flush=True)
        try:
            D2d = D_tilde_squared_disc(r)
            D2c = D_tilde_squared_cont(r)
            D2 = D2d + D2c

            Zfc = Z_full_cont_sl3(beta, r)
            Zmc = Z_bcgp_cont_sl3(beta, r)

            Zfd = Z_full_disc(beta, r)
            Zmd = Z_bcgp_disc(beta, r)

            Zf = (Zfd + Zfc) / D2
            Zm = (Zmd + Zmc) / D2

            # Continuous-only normalized
            Zf_cont_norm = Zfc / D2
            Zm_cont_norm = Zmc / D2

            per_r.append({
                'r': r,
                'D2': D2,
                'ln_D2': np.log(D2),
                'Z_full_cont': Zfc,
                'Z_bcgp_cont': Zmc,
                'Z_full': Zf,
                'Z_bcgp': Zm,
                'ln_Z_full': np.log(abs(Zf)) if abs(Zf) > 1e-300 else float('-inf'),
                'ln_Z_bcgp': np.log(abs(Zm)) if abs(Zm) > 1e-300 else float('-inf'),
                'ln_Z_full_cont': np.log(abs(Zf_cont_norm)) if abs(Zf_cont_norm) > 1e-300 else float('-inf'),
                'ln_Z_bcgp_cont': np.log(abs(Zm_cont_norm)) if abs(Zm_cont_norm) > 1e-300 else float('-inf'),
            })
            print(f"D̃²={D2:.2e}, lnZ_full={per_r[-1]['ln_Z_full']:.4f}, "
                  f"lnZ_bcgp={per_r[-1]['ln_Z_bcgp']:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")

    if len(per_r) >= 3:
        r_arr = np.array([d['r'] for d in per_r], dtype=float)

        # Extract log coefficients
        for label, key in [
            ('ln Z_full (total)', 'ln_Z_full'),
            ('ln Z_bcgp (total)', 'ln_Z_bcgp'),
            ('ln Z_full_cont/D̃²', 'ln_Z_full_cont'),
            ('ln Z_bcgp_cont/D̃²', 'ln_Z_bcgp_cont'),
            ('ln D̃²', 'ln_D2'),
        ]:
            vals = np.array([d[key] for d in per_r])
            valid = np.isfinite(vals)
            if sum(valid) >= 3:
                A = np.column_stack([np.log(r_arr[valid]), np.ones(sum(valid)), 1.0/r_arr[valid]])
                c, _, _, _ = np.linalg.lstsq(A, vals[valid], rcond=None)
                print(f"  {label:30s}: ~ r^{c[0]:+.4f}")

        # Shift analysis
        ln_Zf = np.array([d['ln_Z_full'] for d in per_r])
        ln_Zm = np.array([d['ln_Z_bcgp'] for d in per_r])
        delta = ln_Zf - ln_Zm
        valid = np.isfinite(delta)
        if sum(valid) >= 3:
            A = np.column_stack([np.log(r_arr[valid]), np.ones(sum(valid)), 1.0/r_arr[valid]])
            c_shift, _, _, _ = np.linalg.lstsq(A, delta[valid], rcond=None)
            print(f"\n  SHIFT (ln Z_full - ln Z_bcgp) ~ r^{c_shift[0]:+.4f}")
            print(f"  sl₂ shift: +1/2")
            print(f"  sl₃ shift: {c_shift[0]:+.4f}")
            if c_shift[0] < 0:
                print(f"  → NEGATIVE shift (BCGP > full trace) — CONFIRMS SIGN CHANGE")

    return per_r


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    r_values = [5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31]
    beta = 1.0

    # Main saddle point analysis
    results = compute_saddle_point_analysis(
        r_values=r_values,
        beta=beta,
    )

    # Normalized partition function analysis
    print("\n\n")
    norm_results = normalized_partition_analysis(
        r_values=r_values,
        beta=beta,
    )
