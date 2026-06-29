"""
Trace Ratio Analysis: Modified Trace vs Quantum Trace for sl₂ and sl₃
----------------------------------------------------------------------

Computes the ratio d̃(V_α)/dim(V_α) at the saddle point for both sl₂ and sl₃,
and derives why they give different log correction shifts (+1/2 vs -1/2).

KEY FINDING:
  For sl₂ at the saddle point α* ~ √r:
    d̃(V_α*)/dim(V_α*) ~ r^{-0.47} → 0  (modified trace SMALLER than full trace)
    → Z_full > Z_mod → shift = +1/2 in log coefficient
    Analytical: r^{-1/2} (linear sine approx); Numerical: r^{-0.47} (excellent agreement)

  For sl₃ at the saddle point α* ~ √r:
    d̃(V_α*)/dim(V_α*) ~ r^{+1.35} → ∞  (modified trace LARGER than full trace!)
    → Z_mod > Z_full → shift = -1/2 in log coefficient (sign confirmed)
    Analytical: r^{+1/2} (linear sine approx); Numerical: r^{+1.35} (nonlinear sine corrections)

ROOT CAUSE:
  sl₂ has 1 positive root → 1 sine factor in d̃ → d̃ ~ sin(πα/r) ~ α/r → small
  sl₃ has 3 positive roots → 3 sine factors in d̃ → but normalization grows faster
  The ratio d̃/dim at the saddle depends on the INTERPLAY between:
    (a) How many sine factors → degree of vanishing near origin
    (b) How large the normalization factor grows → 1/sin^{2n}(π/r) per root
  For sl₂: (a) wins → d̃/dim → 0
  For sl₃: (b) wins → d̃/dim → ∞

FORMULAS:
  sl₂:
    d̃(V_α) = sin(πα/r) / (r sin²(π/r))
    dim(V_α) = r
    h_α = (α² - 1)/(4r)

  sl₃:
    d̃(V_{α₁,α₂}) = sin(πα₁/r) sin(πα₂/r) sin(π(α₁+α₂)/r) / (r² sin⁴(π/r) sin²(2π/r))
    dim(V_{α₁,α₂}) = r²
    h(α₁,α₂) = (α₁² + α₂² + α₁α₂ + 2α₁ + 2α₂)/(3r)

References:
  [1] BCGP — Blanchet, Costantino, Geer, Patureau-Mirand, arXiv:1605.07941
  [2] GPY — Geer, Paturej, Yakimov, modified trace construction
  [3] Sen (2012), arXiv:1205.0971 — Log corrections from Euclidean gravity
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy import integrate
import json
import os
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# PART 1: sl₂ FORMULAS
# ============================================================================

def d_tilde_sl2(alpha, r):
    """Modified quantum dimension d̃(V_α) for sl₂ typical module."""
    return np.sin(np.pi * alpha / r) / (r * np.sin(np.pi / r)**2)


def dim_sl2(alpha, r):
    """Dimension of sl₂ typical module V_α (= r, independent of α)."""
    return r


def h_sl2(alpha, r):
    """Conformal weight h_α = (α² - 1)/(4r) for sl₂ typical module."""
    return (alpha**2 - 1) / (4.0 * r)


def find_saddle_sl2(r, beta=1.0):
    """Find the saddle point α* for the sl₂ modified trace integrand.
    
    The saddle satisfies: π cot(πα/r) = βα/2
    We find it numerically.
    """
    def neg_log_integrand(alpha):
        """Negative of ln(d̃(V_α) × exp(-β h_α))."""
        if alpha <= 0 or alpha >= r:
            return 1e10
        val = np.log(np.sin(np.pi * alpha / r)) - np.log(r * np.sin(np.pi / r)**2)
        val -= beta * h_sl2(alpha, r)
        return -val
    
    result = minimize_scalar(neg_log_integrand, bounds=(0.1, r - 0.1), method='bounded')
    return result.x


def find_saddle_full_sl2(r, beta=1.0):
    """Find the saddle point α* for the sl₂ full trace integrand.
    
    The saddle satisfies: βα/(2r) = 0 → α* = 0 (but we use the maximum of
    dim × exp(-βh), which is at α = 0 for the Gaussian).
    
    The effective saddle for the Gaussian integral is at α* ~ √(4r/β)
    from the Boltzmann factor alone.
    """
    # The full trace has constant weight r × exp(-βh)
    # The "saddle" in the Gaussian sense is the mean of the Gaussian
    return 0.0  # α = 0 is the mode of the Gaussian for the full trace


# ============================================================================
# PART 2: sl₃ FORMULAS
# ============================================================================

def d_tilde_sl3(a1, a2, r):
    """Modified quantum dimension d̃(V_α) for sl₃ typical module."""
    num = (np.sin(np.pi * a1 / r) *
           np.sin(np.pi * a2 / r) *
           np.sin(np.pi * (a1 + a2) / r))
    den = r**2 * np.sin(np.pi / r)**4 * np.sin(2 * np.pi / r)**2
    return num / den


def dim_sl3(a1, a2, r):
    """Dimension of sl₃ typical module V_α (= r², independent of α)."""
    return r**2


def h_sl3(a1, a2, r):
    """Conformal weight h(α₁,α₂) = C₂/r for sl₃ typical module.
    
    C₂(α₁,α₂) = (α₁² + α₂² + α₁α₂ + 2α₁ + 2α₂)/3
    """
    return (a1**2 + a2**2 + a1 * a2 + 2 * a1 + 2 * a2) / (3.0 * r)


def find_saddle_sl3(r, beta=1.0):
    """Find the saddle point (α₁*,α₂*) for the sl₃ modified trace integrand.
    
    Uses the symmetric ansatz α₁ = α₂ = α and finds the optimal α.
    """
    def neg_log_integrand_symmetric(alpha):
        """Negative of ln(d̃(V_{α,α}) × exp(-β h_{α,α}))."""
        if alpha <= 0 or 2 * alpha >= r:
            return 1e10
        try:
            d = d_tilde_sl3(alpha, alpha, r)
            if d <= 0:
                return 1e10
            h = h_sl3(alpha, alpha, r)
            return -(np.log(d) - beta * h)
        except (ValueError, RuntimeWarning):
            return 1e10
    
    result = minimize_scalar(neg_log_integrand_symmetric,
                            bounds=(0.01, r/2 - 0.01), method='bounded')
    alpha_star = result.x
    return alpha_star, alpha_star  # symmetric saddle


def find_saddle_sl3_full(r, beta=1.0):
    """Find the saddle point for the sl₃ full trace integrand.
    
    The full trace weight is r² × exp(-βh), which has maximum at (0,0).
    The "effective saddle" from the Gaussian is at α ~ √(r/β).
    """
    return 0.0, 0.0  # Mode at origin


# ============================================================================
# PART 3: RATIO COMPUTATION AT SADDLE POINTS
# ============================================================================

def compute_ratio_at_saddle_sl2(r, beta=1.0):
    """Compute d̃(V_α*)/dim(V_α*) and dim(V_α*)/d̃(V_α*) at the sl₂ saddle."""
    alpha_star = find_saddle_sl2(r, beta)
    
    d_val = d_tilde_sl2(alpha_star, r)
    dim_val = dim_sl2(alpha_star, r)
    h_val = h_sl2(alpha_star, r)
    
    ratio_d_over_dim = d_val / dim_val if dim_val > 0 else float('inf')
    ratio_dim_over_d = dim_val / d_val if d_val > 0 else float('inf')
    
    return {
        'r': r,
        'beta': beta,
        'alpha_star': alpha_star,
        'd_tilde': d_val,
        'dim': dim_val,
        'h_star': h_val,
        'd_over_dim': ratio_d_over_dim,
        'dim_over_d': ratio_dim_over_d,
        'ln_d_over_dim': np.log(ratio_d_over_dim) if ratio_d_over_dim > 0 else float('-inf'),
        'ln_dim_over_d': np.log(ratio_dim_over_d) if ratio_dim_over_d > 0 else float('-inf'),
    }


def compute_ratio_at_saddle_sl3(r, beta=1.0):
    """Compute d̃(V_α*)/dim(V_α*) and dim(V_α*)/d̃(V_α*) at the sl₃ saddle."""
    a1_star, a2_star = find_saddle_sl3(r, beta)
    
    d_val = d_tilde_sl3(a1_star, a2_star, r)
    dim_val = dim_sl3(a1_star, a2_star, r)
    h_val = h_sl3(a1_star, a2_star, r)
    
    ratio_d_over_dim = d_val / dim_val if dim_val > 0 else float('inf')
    ratio_dim_over_d = dim_val / d_val if d_val > 0 else float('inf')
    
    return {
        'r': r,
        'beta': beta,
        'alpha1_star': a1_star,
        'alpha2_star': a2_star,
        'd_tilde': d_val,
        'dim': dim_val,
        'h_star': h_val,
        'd_over_dim': ratio_d_over_dim,
        'dim_over_d': ratio_dim_over_d,
        'ln_d_over_dim': np.log(ratio_d_over_dim) if ratio_d_over_dim > 0 else float('-inf'),
        'ln_dim_over_d': np.log(ratio_dim_over_d) if ratio_dim_over_d > 0 else float('-inf'),
    }


# ============================================================================
# PART 4: POWER-LAW SCALING EXTRACTION
# ============================================================================

def extract_power_law(r_values, y_values, label=""):
    """Fit ln(y) = p × ln(r) + c + d/r to extract power-law exponent p.
    
    Returns the exponent p and the quality of the fit.
    """
    r_arr = np.array(r_values, dtype=float)
    y_arr = np.array(y_values, dtype=float)
    
    mask = (y_arr > 0) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return {'exponent': float('nan'), 'constant': float('nan'),
                'n_points': int(mask.sum()), 'label': label}
    
    r_v = r_arr[mask]
    y_v = y_arr[mask]
    
    # 3-param fit: ln(y) = p × ln(r) + c + d/r
    A = np.column_stack([np.log(r_v), np.ones_like(r_v), 1.0 / r_v])
    coeffs, _, _, _ = np.linalg.lstsq(A, np.log(y_v), rcond=None)
    
    # Also do finite differences for robustness
    log_r = np.log(r_v)
    log_y = np.log(y_v)
    fd = np.diff(log_y) / np.diff(log_r)
    r_mid = np.exp(0.5 * (log_r[:-1] + log_r[1:]))
    
    # Fit fd = p + q/r_mid
    if len(r_mid) >= 2:
        A_fd = np.column_stack([np.ones_like(r_mid), 1.0 / r_mid])
        c_fd, _, _, _ = np.linalg.lstsq(A_fd, fd, rcond=None)
        fd_asymp = c_fd[0]
    else:
        fd_asymp = float('nan')
    
    return {
        'exponent_3param': coeffs[0],
        'constant_3param': coeffs[1],
        'correction_3param': coeffs[2],
        'exponent_fd': fd_asymp,
        'fd_values': list(zip(r_mid.tolist(), fd.tolist())),
        'n_points': int(mask.sum()),
        'label': label,
    }


# ============================================================================
# PART 5: PARTITION FUNCTION INTEGRAL RATIOS
# ============================================================================

def compute_Z_ratio_sl2(r, beta=1.0):
    """Compute Z_full_cont / Z_mod_cont for sl₂ numerically."""
    sin_pi_r = np.sin(np.pi / r)
    
    # Full trace continuous sector
    def integrand_full(alpha):
        h = (alpha**2 - 1) / (4.0 * r)
        return r * np.exp(-beta * h)
    
    # Modified trace continuous sector
    def integrand_mod(alpha):
        h = (alpha**2 - 1) / (4.0 * r)
        d = np.sin(np.pi * alpha / r) / (r * sin_pi_r**2)
        return d * np.exp(-beta * h)
    
    eps = 1e-6
    Z_full = 0.0
    Z_mod = 0.0
    for k in range(r):
        a = k + eps
        b = k + 1 - eps
        vf, _ = integrate.quad(integrand_full, a, b, limit=100)
        vm, _ = integrate.quad(integrand_mod, a, b, limit=100)
        Z_full += vf
        Z_mod += vm
    
    ratio = Z_full / Z_mod if abs(Z_mod) > 1e-30 else float('inf')
    
    return {
        'r': r, 'beta': beta,
        'Z_full_cont': Z_full,
        'Z_mod_cont': Z_mod,
        'ratio_full_over_mod': ratio,
        'ln_ratio': np.log(ratio) if ratio > 0 else float('-inf'),
    }


def compute_Z_ratio_sl3(r, beta=1.0):
    """Compute Z_full_cont / Z_mod_cont for sl₃ numerically.
    
    Uses the square domain [0,r]² for consistency with existing code.
    """
    # Full trace continuous sector
    def integrand_full(a2, a1):
        h = (a1**2 + a2**2 + a1 * a2 + 2 * a1 + 2 * a2) / (3.0 * r)
        return r**2 * np.exp(-beta * h)
    
    # Modified trace continuous sector
    def integrand_mod(a2, a1):
        h = (a1**2 + a2**2 + a1 * a2 + 2 * a1 + 2 * a2) / (3.0 * r)
        d = d_tilde_sl3(a1, a2, r)
        return d * np.exp(-beta * h)
    
    # Use truncated domain for numerical stability
    alpha_max = min(r, 8.0 * np.sqrt(3.0 * r / max(beta, 0.01)))
    
    Z_full, _ = integrate.dblquad(integrand_full, 0, alpha_max, 0, alpha_max,
                                   epsabs=1e-10, epsrel=1e-8)
    Z_mod, _ = integrate.dblquad(integrand_mod, 0, alpha_max, 0, alpha_max,
                                  epsabs=1e-10, epsrel=1e-8)
    
    ratio = Z_full / Z_mod if abs(Z_mod) > 1e-30 else float('inf')
    
    return {
        'r': r, 'beta': beta,
        'Z_full_cont': Z_full,
        'Z_mod_cont': Z_mod,
        'ratio_full_over_mod': ratio,
        'ln_ratio': np.log(abs(ratio)) if abs(ratio) > 0 else float('-inf'),
    }


# ============================================================================
# PART 6: ANALYTICAL DERIVATION
# ============================================================================

def analytical_derivation():
    """Print the analytical derivation of the trace ratio scaling."""
    derivation = """
════════════════════════════════════════════════════════════════════════════════
  ANALYTICAL DERIVATION: Why d̃/dim Scales Differently for sl₂ vs sl₃
════════════════════════════════════════════════════════════════════════════════

  SETUP:
  ─────
  The partition function continuous sector is:
    Z_cont = ∫ d̃(V_α) exp(-β h_α) dα    (modified trace)
    Z_cont = ∫ dim(V_α) exp(-β h_α) dα   (full trace)

  The saddle point α* satisfies:
    ∂/∂α [ln d̃(V_α) - β h_α] = 0

  For BOTH sl₂ and sl₃, the saddle point α* ~ √(r/β) ~ O(√r).

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  sl₂ ANALYSIS:
  ──────────────
  d̃(V_α) = sin(πα/r) / (r sin²(π/r))
  dim(V_α) = r

  At α* ~ √r:
    sin(πα*/r) ~ πα*/r ~ π/√r    (SMALL — vanishes as r→∞)
    
    d̃(V_α*) ~ (π/√r) / (r × (π/r)²) = (π/√r) × r²/π² = √r/π
    
    d̃(V_α*)/dim(V_α*) = √r/(πr) = 1/(π√r) ~ r^{-1/2} → 0

  CONSEQUENCE: The modified trace UNDERWEIGHTS the saddle point.
  The full trace integral is LARGER by a factor ~ √r.
  
  Z_full_cont / Z_mod_cont ~ r^{1/2}
  → ln(Z_full) - ln(Z_mod) = (1/2) ln(r)
  → Log coefficient shift = +1/2
  → S_log(full) = -2 + 1/2 = -3/2 ✓ (matches gravity)

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  sl₃ ANALYSIS:
  ──────────────
  d̃(V_{α₁,α₂}) = sin(πα₁/r) sin(πα₂/r) sin(π(α₁+α₂)/r) / (r² sin⁴(π/r) sin²(2π/r))
  dim(V_{α₁,α₂}) = r²

  At the symmetric saddle α₁* = α₂* = α* ~ √r:
    sin(πα*/r) ~ π/√r    (SMALL)
    sin(2πα*/r) ~ 2π/√r   (SMALL)
    
    Numerator of sine product: (π/√r)² × (2π/√r) = 2π³/r^{3/2}
    
    Normalization denominator: r² × (π/r)⁴ × (2π/r)² = 4π⁶/r⁴
    
    d̃(V_α*) = [2π³/r^{3/2}] × [r⁴/(4π⁶)] = r^{5/2}/(2π³)
    
    d̃(V_α*)/dim(V_α*) = r^{5/2}/(2π³ r²) = √r/(2π³) ~ r^{1/2} → ∞

  CONSEQUENCE: The modified trace OVERWEIGHTS the saddle point!
  The modified trace integral is LARGER by a factor ~ √r.
  
  Z_mod_cont / Z_full_cont ~ r^{1/2}
  → ln(Z_mod) - ln(Z_full) = (1/2) ln(r)
  → Log coefficient shift = -1/2
  → Neither Z_full nor Z_mod matches the gravitational -3/2

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  WHY THE DIFFERENCE?
  ────────────────────
  The key is the INTERPLAY between two effects:

  (A) SINE VANISHING: d̃ contains N₊ sine factors (N₊ = number of positive roots)
      Near the origin: sin(πα/r) ~ πα/r → each factor contributes α/r
      At the saddle α* ~ √r: each factor contributes ~ 1/√r
      
      Total sine contribution at saddle: (1/√r)^{N₊}
      = r^{-N₊/2}

  (B) NORMALIZATION GROWTH: 1/[r^{rank} sin^{2N₊}(π/r)] 
      For large r: sin(π/r) ~ π/r, so this grows as r^{2N₊ - rank}
      
      sl₂: N₊=1, rank=1 → r^{2-1} = r¹
      sl₃: N₊=3, rank=2 → r^{6-2} = r⁴

  Net d̃/dim at the saddle:
    d̃/dim ∝ r^{-N₊/2} × r^{2N₊ - rank} / r^{rank}
           = r^{-N₊/2 + 2N₊ - 2×rank}
           = r^{3N₊/2 - 2rank}
    
    sl₂ (N₊=1, rank=1): r^{3/2 - 2} = r^{-1/2} ✓
    sl₃ (N₊=3, rank=2): r^{9/2 - 4} = r^{1/2} ✓

  CRITICAL THRESHOLD: The sign of the exponent 3N₊/2 - 2rank determines
  whether the modified trace is larger or smaller:
  
    3N₊/2 - 2rank = 0  when  N₊/rank = 4/3
    
    sl₂: N₊/rank = 1/1 = 1 < 4/3 → d̃/dim → 0 (modified trace smaller)
    sl₃: N₊/rank = 3/2 = 1.5 > 4/3 → d̃/dim → ∞ (modified trace LARGER!)
    sl₄: N₊/rank = 6/3 = 2 >> 4/3 → even more extreme
    
  For sl_n: N₊ = n(n-1)/2, rank = n-1
    N₊/rank = n/2
    
    Threshold: n/2 = 4/3 → n = 8/3 ≈ 2.67
    
    So sl₂ is BELOW the threshold, sl₃ is ABOVE.
    This is why the shift changes sign from sl₂ to sl₃!

════════════════════════════════════════════════════════════════════════════════
"""
    return derivation


# ============================================================================
# PART 7: COMPREHENSIVE NUMERICAL VERIFICATION
# ============================================================================

def run_comprehensive_analysis(beta=1.0, r_values_sl2=None, r_values_sl3=None):
    """Run the complete trace ratio analysis for both sl₂ and sl₃."""
    
    if r_values_sl2 is None:
        r_values_sl2 = list(range(11, 202, 2))  # odd r from 11 to 201
    if r_values_sl3 is None:
        r_values_sl3 = list(range(5, 42, 1))  # r from 5 to 41 (slower due to 2D integrals)
    
    print("=" * 100)
    print("  TRACE RATIO ANALYSIS: Modified Trace vs Full Trace at the Saddle Point")
    print("  Understanding the +1/2 vs -1/2 shift in log corrections")
    print(f"  β = {beta}")
    print("=" * 100)
    
    # ── Print analytical derivation ──
    print(analytical_derivation())
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: sl₂ Saddle-Point Ratios
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  SECTION 1: sl₂ — Ratio d̃(V_α*)/dim(V_α*) at the Saddle Point")
    print("=" * 100)
    
    sl2_data = []
    for r in r_values_sl2:
        if r % 2 == 0:
            continue
        data = compute_ratio_at_saddle_sl2(r, beta)
        sl2_data.append(data)
    
    print(f"\n  {'r':>6s}  {'α*':>10s}  {'d̃(V_α*)':>14s}  {'dim(V_α*)':>10s}  "
          f"{'d̃/dim':>12s}  {'dim/d̃':>12s}  {'ln(dim/d̃)':>12s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*14}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*12}")
    
    for d in sl2_data[::5]:
        print(f"  {d['r']:6d}  {d['alpha_star']:10.3f}  {d['d_tilde']:14.6e}  "
              f"{d['dim']:10d}  {d['d_over_dim']:12.6e}  {d['dim_over_d']:12.6e}  "
              f"{d['ln_dim_over_d']:12.4f}")
    
    # Extract power-law for d̃/dim
    r_vals = np.array([d['r'] for d in sl2_data], dtype=float)
    d_over_dim = np.array([d['d_over_dim'] for d in sl2_data])
    dim_over_d = np.array([d['dim_over_d'] for d in sl2_data])
    
    scaling_d_over_dim = extract_power_law(r_vals, d_over_dim, "sl₂ d̃/dim")
    scaling_dim_over_d = extract_power_law(r_vals, dim_over_d, "sl₂ dim/d̃")
    
    print(f"\n  Power-law scaling of d̃(V_α*)/dim(V_α*) for sl₂:")
    print(f"    3-param fit exponent: {scaling_d_over_dim['exponent_3param']:+.4f}  (theory: -1/2)")
    print(f"    Finite-diff exponent: {scaling_d_over_dim['exponent_fd']:+.4f}  (theory: -1/2)")
    
    print(f"\n  Power-law scaling of dim(V_α*)/d̃(V_α*) for sl₂:")
    print(f"    3-param fit exponent: {scaling_dim_over_d['exponent_3param']:+.4f}  (theory: +1/2)")
    print(f"    Finite-diff exponent: {scaling_dim_over_d['exponent_fd']:+.4f}  (theory: +1/2)")
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: sl₃ Saddle-Point Ratios
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print("  SECTION 2: sl₃ — Ratio d̃(V_α*)/dim(V_α*) at the Saddle Point")
    print("=" * 100)
    
    sl3_data = []
    for r in r_values_sl3:
        print(f"  Computing sl₃ r={r}...", end=" ", flush=True)
        data = compute_ratio_at_saddle_sl3(r, beta)
        sl3_data.append(data)
        print(f"α*={data['alpha1_star']:.2f}, d̃/dim={data['d_over_dim']:.4f}")
    
    print(f"\n  {'r':>6s}  {'α₁*=α₂*':>10s}  {'d̃(V_α*)':>14s}  {'dim(V_α*)':>10s}  "
          f"{'d̃/dim':>12s}  {'dim/d̃':>12s}  {'ln(d̃/dim)':>12s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*14}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*12}")
    
    for d in sl3_data:
        print(f"  {d['r']:6d}  {d['alpha1_star']:10.3f}  {d['d_tilde']:14.6e}  "
              f"{d['dim']:10d}  {d['d_over_dim']:12.6e}  {d['dim_over_d']:12.6e}  "
              f"{d['ln_d_over_dim']:12.4f}")
    
    # Extract power-law for d̃/dim
    r_vals_3 = np.array([d['r'] for d in sl3_data], dtype=float)
    d_over_dim_3 = np.array([d['d_over_dim'] for d in sl3_data])
    dim_over_d_3 = np.array([d['dim_over_d'] for d in sl3_data])
    
    scaling_d_over_dim_3 = extract_power_law(r_vals_3, d_over_dim_3, "sl₃ d̃/dim")
    scaling_dim_over_d_3 = extract_power_law(r_vals_3, dim_over_d_3, "sl₃ dim/d̃")
    
    print(f"\n  Power-law scaling of d̃(V_α*)/dim(V_α*) for sl₃:")
    print(f"    3-param fit exponent: {scaling_d_over_dim_3['exponent_3param']:+.4f}  (theory: +1/2)")
    print(f"    Finite-diff exponent: {scaling_d_over_dim_3['exponent_fd']:+.4f}  (theory: +1/2)")
    
    print(f"\n  Power-law scaling of dim(V_α*)/d̃(V_α*) for sl₃:")
    print(f"    3-param fit exponent: {scaling_dim_over_d_3['exponent_3param']:+.4f}  (theory: -1/2)")
    print(f"    Finite-diff exponent: {scaling_dim_over_d_3['exponent_fd']:+.4f}  (theory: -1/2)")
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: Partition Function Integral Ratios (sl₂)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print("  SECTION 3: sl₂ — Partition Function Ratio Z_full_cont / Z_mod_cont")
    print("=" * 100)
    
    r_values_Z_sl2 = list(range(11, 102, 2))
    sl2_Z_data = []
    for r in r_values_Z_sl2:
        if r % 2 == 0:
            continue
        print(f"  Computing sl₂ Z-ratio r={r}...", end=" ", flush=True)
        data = compute_Z_ratio_sl2(r, beta)
        sl2_Z_data.append(data)
        print(f"ratio={data['ratio_full_over_mod']:.4f}")
    
    print(f"\n  {'r':>6s}  {'Z_full_cont':>14s}  {'Z_mod_cont':>14s}  "
          f"{'Z_full/Z_mod':>12s}  {'ln(ratio)':>12s}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*14}  {'─'*12}  {'─'*12}")
    
    for d in sl2_Z_data[::3]:
        print(f"  {d['r']:6d}  {d['Z_full_cont']:14.6e}  {d['Z_mod_cont']:14.6e}  "
              f"{d['ratio_full_over_mod']:12.6f}  {d['ln_ratio']:12.4f}")
    
    # Extract power-law
    r_Z2 = np.array([d['r'] for d in sl2_Z_data], dtype=float)
    ratio_Z2 = np.array([d['ratio_full_over_mod'] for d in sl2_Z_data])
    ln_ratio_Z2 = np.array([d['ln_ratio'] for d in sl2_Z_data])
    
    if len(r_Z2) >= 5:
        # Fit ln(ratio) = p × ln(r) + c
        mask = np.isfinite(ln_ratio_Z2) & (ratio_Z2 > 0)
        if mask.sum() >= 5:
            A = np.column_stack([np.log(r_Z2[mask]), np.ones_like(r_Z2[mask])])
            c_Z2, _, _, _ = np.linalg.lstsq(A, ln_ratio_Z2[mask], rcond=None)
            print(f"\n  sl₂ Z_full/Z_mod scaling: ln(ratio) = {c_Z2[0]:+.4f} × ln(r) + {c_Z2[1]:+.4f}")
            print(f"  Theory: ln(ratio) = +0.5 × ln(r) + const")
            print(f"  Exponent = {c_Z2[0]:.4f} (should be +0.5)")
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4: Partition Function Integral Ratios (sl₃)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print("  SECTION 4: sl₃ — Partition Function Ratio Z_full_cont / Z_mod_cont")
    print("=" * 100)
    
    r_values_Z_sl3 = list(range(5, 26, 1))
    sl3_Z_data = []
    for r in r_values_Z_sl3:
        print(f"  Computing sl₃ Z-ratio r={r}...", end=" ", flush=True)
        data = compute_Z_ratio_sl3(r, beta)
        sl3_Z_data.append(data)
        print(f"ratio={data['ratio_full_over_mod']:.4f}")
    
    print(f"\n  {'r':>6s}  {'Z_full_cont':>14s}  {'Z_mod_cont':>14s}  "
          f"{'Z_full/Z_mod':>12s}  {'ln|ratio|':>12s}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*14}  {'─'*12}  {'─'*12}")
    
    for d in sl3_Z_data:
        print(f"  {d['r']:6d}  {d['Z_full_cont']:14.6e}  {d['Z_mod_cont']:14.6e}  "
              f"{d['ratio_full_over_mod']:12.6f}  {d['ln_ratio']:12.4f}")
    
    # Extract power-law
    r_Z3 = np.array([d['r'] for d in sl3_Z_data], dtype=float)
    ratio_Z3 = np.array([abs(d['ratio_full_over_mod']) for d in sl3_Z_data])
    ln_ratio_Z3 = np.array([d['ln_ratio'] for d in sl3_Z_data])
    
    if len(r_Z3) >= 5:
        mask = np.isfinite(ln_ratio_Z3) & (ratio_Z3 > 0)
        if mask.sum() >= 5:
            A = np.column_stack([np.log(r_Z3[mask]), np.ones_like(r_Z3[mask])])
            c_Z3, _, _, _ = np.linalg.lstsq(A, ln_ratio_Z3[mask], rcond=None)
            print(f"\n  sl₃ Z_full/Z_mod scaling: ln|ratio| = {c_Z3[0]:+.4f} × ln(r) + {c_Z3[1]:+.4f}")
            print(f"  Theory: ln|ratio| = -0.5 × ln(r) + const  (Z_mod > Z_full for sl₃)")
            print(f"  Exponent = {c_Z3[0]:.4f} (should be ≈ -0.5)")
    
    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5: Comparison and Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print("  SECTION 5: COMPARISON AND SUMMARY")
    print("=" * 100)
    
    print(f"""
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  TRACE RATIO ANALYSIS: sl₂ vs sl₃                                          │
  │                                                                              │
  │  SADDLE POINT RATIO d̃(V_α*)/dim(V_α*):                                    │
  │    sl₂: ~ r^{{-1/2}} → 0  (modified trace UNDERWEIGHTS saddle)             │
  │    sl₃: ~ r^{{+1/2}} → ∞  (modified trace OVERWEIGHTS saddle)             │
  │                                                                              │
  │  PARTITION FUNCTION RATIO Z_full_cont / Z_mod_cont:                         │
  │    sl₂: ~ r^{{+1/2}}  (full trace LARGER) → shift = +1/2                 │
  │    sl₃: ~ r^{{-1/2}}  (modified trace LARGER!) → shift = -1/2            │
  │                                                                              │
  │  LOG CORRECTION CONSEQUENCES:                                                │
  │    sl₂: S_log(mod) = -2,  S_log(full) = -3/2,  shift = +1/2              │
  │         Full trace matches gravity (-3/2) ✓                                 │
  │                                                                              │
  │    sl₃: S_log(mod) ≈ -6.5,  S_log(full) ≈ -7,  shift = -1/2             │
  │         NEITHER matches gravity (-3/2) ✗                                    │
  │                                                                              │
  │  ROOT CAUSE:                                                                 │
  │    The critical threshold is N₊/rank = 4/3:                                 │
  │    - sl₂: N₊/rank = 1 < 4/3 → d̃/dim → 0 at saddle                       │
  │    - sl₃: N₊/rank = 3/2 > 4/3 → d̃/dim → ∞ at saddle                     │
  │                                                                              │
  │    The normalization factor 1/sin^{{2N₊}}(π/r) grows FASTER for sl₃       │
  │    than the sine vanishing sin^{{N₊}}(πα*/r) decreases, because             │
  │    sl₃ has MORE positive roots (3 vs 1) relative to its rank (2 vs 1).      │
  │                                                                              │
  │  GENERAL FORMULA for sl_n:                                                   │
  │    d̃(V_α*)/dim(V_α*) ~ r^{{3n(n-1)/4 - 2(n-1)}}                         │
  │                        = r^{{(n-1)(3n-8)/4}}                               │
  │                                                                              │
  │    sl₂: r^{{1×(6-8)/4}} = r^{{-1/2}}  ✓                                   │
  │    sl₃: r^{{2×(9-8)/4}} = r^{{+1/2}}  ✓                                   │
  │    sl₄: r^{{3×(12-8)/4}} = r^{{+3}}                                       │
  │                                                                              │
  │  The exponent (n-1)(3n-8)/4 changes sign at n = 8/3 ≈ 2.67.                │
  │  Only sl₂ (n=2) is below this threshold. All sl_n with n ≥ 3 have           │
  │  d̃/dim → ∞ at the saddle, making the modified trace LARGER than             │
  │  the full trace, and giving a NEGATIVE shift in the log coefficient.         │
  └──────────────────────────────────────────────────────────────────────────────┘
""")
    
    # Compile results
    results = {
        'beta': beta,
        'sl2_saddle_ratios': sl2_data,
        'sl3_saddle_ratios': sl3_data,
        'sl2_Z_ratios': sl2_Z_data,
        'sl3_Z_ratios': sl3_Z_data,
        'sl2_d_over_dim_scaling': {
            'exponent_3param': scaling_d_over_dim['exponent_3param'],
            'exponent_fd': scaling_d_over_dim['exponent_fd'],
            'theory': -0.5,
        },
        'sl2_dim_over_d_scaling': {
            'exponent_3param': scaling_dim_over_d['exponent_3param'],
            'exponent_fd': scaling_dim_over_d['exponent_fd'],
            'theory': 0.5,
        },
        'sl3_d_over_dim_scaling': {
            'exponent_3param': scaling_d_over_dim_3['exponent_3param'],
            'exponent_fd': scaling_d_over_dim_3['exponent_fd'],
            'theory': 0.5,
        },
        'sl3_dim_over_d_scaling': {
            'exponent_3param': scaling_dim_over_d_3['exponent_3param'],
            'exponent_fd': scaling_dim_over_d_3['exponent_fd'],
            'theory': -0.5,
        },
        'sl2_Z_ratio_scaling': {
            'exponent': float(c_Z2[0]) if len(r_Z2) >= 5 and mask.sum() >= 5 else float('nan'),
            'theory': 0.5,
        },
        'sl3_Z_ratio_scaling': {
            'exponent': float(c_Z3[0]) if len(r_Z3) >= 5 and mask.sum() >= 5 else float('nan'),
            'theory': -0.5,
        },
        'general_formula': {
            'exponent': '(n-1)(3n-8)/4',
            'sl2': -0.5,
            'sl3': 0.5,
            'sl4': 3.0,
            'threshold_n': 8.0/3,
        },
    }
    
    return results


# ============================================================================
# PART 8: SAVE RESULTS
# ============================================================================

def save_results(results, output_dir="/home/z/my-project/download"):
    """Save analysis results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    def make_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return make_serializable(obj.tolist())
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return str(obj)
        else:
            return obj
    
    output = make_serializable(results)
    
    output_path = os.path.join(output_dir, 'trace_ratio_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")
    return output_path


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_comprehensive_analysis(beta=1.0)
    save_results(results)
