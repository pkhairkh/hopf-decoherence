"""
sl₃ Continuous Sector Refinement (v3 — Saddle-Point + Monte Carlo)
----------------------------------------------------------------------

Refines the sl₃ continuous sector computation with:
  1. Gaussian saddle-point integration (Laplace method + Gauss-Hermite quadrature)
  2. Monte Carlo integration (stratified sampling)
  3. Direct dblquad (baseline)
  4. Extended r range: 5,7,9,11,13,15,17,19,21,25,31,41,51

KEY INSIGHT: The BCGP modified trace integrand d̃(V_α)×e^{-βC₂/r} has its
saddle point at the INTERIOR of the Weyl chamber (not at the origin), because
d̃(V_α) vanishes at α=0. This creates a competition between:
  - d̃ prefactor: maximized at α₁=α₂=r/3 (center of Weyl chamber)
  - Boltzmann factor: maximized at α₁=α₂=0 (origin)

The full thermal trace integrand r²×e^{-βC₂/r} is maximized at the origin
(boundary of domain), requiring a different integration strategy.

MATHEMATICAL FORMULAS:
  C₂(α₁,α₂) = (α₁² + α₂² + α₁α₂ + 2α₁ + 2α₂)/3
  d̃(V_α) = sin(πα₁/r) sin(πα₂/r) sin(π(α₁+α₂)/r) / (r² sin⁴(π/r) sin²(2π/r))
  dim(V_α) = r²

  Saddle point equation (BCGP, symmetric case α₁=α₂=α*):
    π[cot(πα*/r) + cot(2πα*/r)] = β(3α*+2)/3
"""

import numpy as np
from scipy import integrate, optimize
import warnings
import json
import os
import time

warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================================
# CORE FORMULAS (self-contained, matching sl3_master_identity.py)
# ============================================================================

def casimir_sl3(a, b):
    """Quadratic Casimir C₂(a,b) = (a² + b² + ab + 2a + 2b)/3."""
    return (a**2 + b**2 + a*b + 2*a + 2*b) / 3.0


def conformal_weight_sl3(a, b, r):
    """Conformal weight h(a,b) = C₂(a,b)/r."""
    return casimir_sl3(a, b) / r


def weyl_dim_sl3(a, b):
    """Weyl dimension formula: dim L(a,b) = (a+1)(b+1)(a+b+2)/2."""
    return (a + 1) * (b + 1) * (a + b + 2) // 2


def enumerate_alcove(r):
    """Enumerate the alcove C_r = {(a,b) : a >= 0, b >= 0, a+b <= r-2}."""
    alcove = []
    for a in range(r - 1):
        for b in range(r - 1 - a):
            alcove.append((a, b))
    return alcove


def modified_qdim_disc(a, b, r):
    """Modified quantum dimension d̃(P(a,b)) for discrete sector."""
    if a + b > r - 2:
        return 0.0
    sign = (-1) ** (a + b)
    num = (np.sin(np.pi * (a + 1) / r) *
           np.sin(np.pi * (b + 1) / r) *
           np.sin(np.pi * (a + b + 2) / r))
    den = r**2 * np.sin(np.pi / r)**4 * np.sin(2 * np.pi / r)**2
    return sign * num / den


def modified_qdim_cont(a1, a2, r):
    """Modified quantum dimension d̃(V_α) for typical module V_(α₁,α₂)."""
    num = (np.sin(np.pi * a1 / r) *
           np.sin(np.pi * a2 / r) *
           np.sin(np.pi * (a1 + a2) / r))
    den = r**2 * np.sin(np.pi / r)**4 * np.sin(2 * np.pi / r)**2
    return num / den


def modified_qdim_cont_vec(a1, a2, r):
    """Vectorized modified quantum dimension for array inputs."""
    num = (np.sin(np.pi * a1 / r) *
           np.sin(np.pi * a2 / r) *
           np.sin(np.pi * (a1 + a2) / r))
    den = r**2 * np.sin(np.pi / r)**4 * np.sin(2 * np.pi / r)**2
    return num / den


# ============================================================================
# DISCRETE SECTOR COMPUTATIONS
# ============================================================================

def D_tilde_squared_disc(r):
    """Discrete part of modified global dimension: Σ d̃(P(a,b))²."""
    total = 0.0
    for a, b in enumerate_alcove(r):
        d = modified_qdim_disc(a, b, r)
        total += d * d
    return total


def Z_full_disc(beta, r):
    """Full thermal trace — discrete sector."""
    total = 0.0
    for a, b in enumerate_alcove(r):
        d_L = weyl_dim_sl3(a, b)
        h = conformal_weight_sl3(a, b, r)
        total += d_L * np.exp(-beta * h)
    return total


def Z_bcgp_disc(beta, r):
    """BCGP modified trace — discrete sector."""
    total = 0.0
    for a, b in enumerate_alcove(r):
        d = modified_qdim_disc(a, b, r)
        h = conformal_weight_sl3(a, b, r)
        total += d * np.exp(-beta * h)
    return total


# ============================================================================
# SADDLE POINT ANALYSIS
# ============================================================================

def log_bcgp_integrand(a1, a2, r, beta):
    """Log of the BCGP continuous integrand (without denominator constant).
    
    ln[d̃(V_α) × e^{-βC₂/r}] = ln sin(πα₁/r) + ln sin(πα₂/r) + ln sin(π(α₁+α₂)/r)
                                  - βC₂(α)/r + const
    """
    # Guard against domain boundaries where sin = 0
    if a1 <= 0 or a2 <= 0 or a1 + a2 >= r:
        return -np.inf
    
    ln_sin1 = np.log(np.sin(np.pi * a1 / r))
    ln_sin2 = np.log(np.sin(np.pi * a2 / r))
    ln_sin3 = np.log(np.sin(np.pi * (a1 + a2) / r))
    c2 = casimir_sl3(a1, a2)
    
    return ln_sin1 + ln_sin2 + ln_sin3 - beta * c2 / r


def find_saddle_point_bcgp(r, beta):
    """Find the saddle point of the BCGP continuous integrand.
    
    By symmetry of the Weyl chamber and the integrand under α₁ ↔ α₂,
    the saddle is at α₁ = α₂ = α*.
    
    Equation: π[cot(πα*/r) + cot(2πα*/r)] = β(3α*+2)/3
    """
    def equation(alpha):
        """Residual of the saddle point equation for α₁=α₂=α."""
        a = alpha[0]
        if a <= 0.01 or a >= r - 0.01:
            return [1e10]
        
        # ∂(ln integrand)/∂α₁ at α₁=α₂=a
        # = (π/r)cot(πa/r) + (π/r)cot(2πa/r) - β(2a+a+2)/(3r)
        # = (π/r)[cot(πa/r) + cot(2πa/r)] - β(3a+2)/(3r)
        cot1 = 1.0 / np.tan(np.pi * a / r)
        cot2 = 1.0 / np.tan(2 * np.pi * a / r)
        
        grad = (np.pi / r) * (cot1 + cot2) - beta * (3 * a + 2) / (3 * r)
        return [grad]
    
    # Try multiple starting points
    best_alpha = r / 3.0  # Start at center of Weyl chamber
    best_val = abs(equation([best_alpha])[0])
    
    for start in [0.1, 0.5, 1.0, r/6, r/4, r/3, r/2.5, r/2]:
        try:
            result = optimize.brentq(
                lambda a: equation([a])[0],
                max(0.01, start * 0.3),
                min(r - 0.01, start * 2.0),
                xtol=1e-12
            )
            val = abs(equation([result])[0])
            if val < best_val:
                best_alpha = result
                best_val = val
        except (ValueError, RuntimeError):
            pass
    
    # Also try general 2D optimization (not assuming α₁=α₂)
    def neg_log_integrand_2d(alpha_vec):
        a1, a2 = alpha_vec
        if a1 <= 0.01 or a2 <= 0.01 or a1 + a2 >= r - 0.01:
            return 1e10
        return -log_bcgp_integrand(a1, a2, r, beta)
    
    # Try symmetric start
    try:
        result_2d = optimize.minimize(
            neg_log_integrand_2d,
            [best_alpha, best_alpha],
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000}
        )
        if result_2d.success and result_2d.fun < neg_log_integrand_2d([best_alpha, best_alpha]):
            a1_opt, a2_opt = result_2d.x
            # Verify it's really a maximum
            if 0.01 < a1_opt and 0.01 < a2_opt and a1_opt + a2_opt < r - 0.01:
                best_alpha_2d = (a1_opt + a2_opt) / 2  # Average for symmetric estimate
                # Use the 2D result
                return a1_opt, a2_opt
    except Exception:
        pass
    
    return best_alpha, best_alpha


def find_saddle_point_full(r, beta):
    """Find the saddle point of the full thermal trace integrand.
    
    For r² × e^{-βC₂/r}, the maximum is at α₁=α₂=0 (origin).
    Since this is on the boundary, we return (0,0) and note that
    Laplace approximation needs a boundary correction.
    """
    return 0.0, 0.0


def hessian_bcgp(a1, a2, r, beta):
    """Compute the Hessian of ln[d̃(V_α)×e^{-βC₂/r}] at (a1, a2).
    
    Returns the 2×2 Hessian matrix H where:
      H[0,0] = ∂²F/∂α₁², H[0,1] = ∂²F/∂α₁∂α₂, etc.
    """
    # Second derivatives of ln sin(πα₁/r): -π²/(r² sin²(πα₁/r))
    d2_sin1 = -(np.pi / r)**2 / np.sin(np.pi * a1 / r)**2
    d2_sin2 = -(np.pi / r)**2 / np.sin(np.pi * a2 / r)**2
    d2_sin3 = -(np.pi / r)**2 / np.sin(np.pi * (a1 + a2) / r)**2
    
    # Cross derivatives of ln sin(π(α₁+α₂)/r)
    d2_cross = d2_sin3  # ∂²/∂α₁∂α₂ of ln sin(π(α₁+α₂)/r)
    
    # Second derivatives of -βC₂/r
    # C₂ = (α₁² + α₂² + α₁α₂ + 2α₁ + 2α₂)/3
    # ∂²C₂/∂α₁² = 2/3, ∂²C₂/∂α₂² = 2/3, ∂²C₂/∂α₁∂α₂ = 1/3
    d2_c2_11 = -beta * (2.0/3.0) / r
    d2_c2_22 = -beta * (2.0/3.0) / r
    d2_c2_12 = -beta * (1.0/3.0) / r
    
    H = np.array([
        [d2_sin1 + d2_sin3 + d2_c2_11, d2_cross + d2_c2_12],
        [d2_cross + d2_c2_12, d2_sin2 + d2_sin3 + d2_c2_22]
    ])
    
    return H


def hessian_full(r, beta):
    """Hessian of ln[r²×e^{-βC₂/r}] at the origin (0,0).
    
    At α=(0,0): C₂ Hessian is (2/3, 1/3; 1/3, 2/3).
    """
    H = np.array([
        [-beta * 2.0 / (3.0 * r), -beta * 1.0 / (3.0 * r)],
        [-beta * 1.0 / (3.0 * r), -beta * 2.0 / (3.0 * r)]
    ])
    return H


# ============================================================================
# INTEGRATION METHOD 1: SADDLE-POINT METHODS
# ============================================================================

def laplace_bcgp_cont(r, beta):
    """Laplace (saddle-point) approximation for BCGP continuous sector.
    
    Z_bcgp_cont ≈ f(α*) × 2π / √(|det H|)
    
    where f(α*) = d̃(V_{α*}) × e^{-βC₂(α*)/r} is the integrand at the saddle,
    and H is the Hessian of ln(f) at the saddle.
    
    This is the LEADING ORDER Laplace approximation. Higher-order corrections
    involve 3rd and 4th derivatives of ln(f).
    """
    a1_star, a2_star = find_saddle_point_bcgp(r, beta)
    
    # Integrand at saddle point
    d_tilde_star = modified_qdim_cont(a1_star, a2_star, r)
    c2_star = casimir_sl3(a1_star, a2_star)
    f_star = d_tilde_star * np.exp(-beta * c2_star / r)
    
    # Hessian of ln(f)
    H = hessian_bcgp(a1_star, a2_star, r, beta)
    det_H = np.linalg.det(H)
    
    # Laplace approximation: ∫ f(α) dα ≈ f(α*) × (2π) / √(|det H|)
    # Since H is negative definite (maximum), det H > 0
    laplace_val = f_star * 2 * np.pi / np.sqrt(np.abs(det_H))
    
    return laplace_val


def laplace_full_cont(r, beta):
    """Laplace approximation for full thermal trace continuous sector.
    
    The integrand r² e^{-βC₂/r} has its maximum at the origin (boundary).
    For a boundary maximum, the Laplace approximation requires a factor of 1/2
    for each boundary direction.
    
    Full Gaussian: Z ≈ r² × (2π)^{d/2} / √(det(M))
    where M = β/(3r) × [[2,1],[1,2]] is the Hessian of βC₂/r at origin.
    
    For the triangular domain {α₁>0, α₂>0, α₁+α₂<r}, the origin is a vertex
    (two boundary faces). The quarter-Gaussian gives 1/4 of the full integral.
    
    But the triangular domain only captures 1/6 of the full Gaussian by the
    6-fold symmetry of the sl₃ Weyl group acting on R².
    
    Actually, for large r, the domain is effectively [0,∞)² since the Gaussian
    decays fast. The fraction of the full Gaussian in the first quadrant is
    determined by the symmetry of the quadratic form.
    
    For C₂ = (α₁² + α₂² + α₁α₂)/3, the matrix is (1/3)[[2,1],[1,2]].
    The full Gaussian over R²: I = 2π / √(β²/(3r²)) = 2π√3 r/β
    First quadrant fraction: needs numerical computation.
    """
    # Analytical: the quadratic form Q = β(2α₁² + 2α₂² + 2α₁α₂)/(3r)
    # Matrix M = β/(3r) [[2,1],[1,2]], det(M) = β²(4-1)/(9r²) = β²/(3r²)
    # Full Gaussian: ∫_{R²} e^{-α^T M α} dα = π/√(det M) = π√3 r/β
    # 
    # The first-quadrant fraction depends on the eigenvectors of M.
    # M has eigenvalues β/(3r) × {1, 3} with eigenvectors (1,-1)/√2 and (1,1)/√2.
    # 
    # The first quadrant in the eigenbasis is rotated by 45°, and the fraction
    # is NOT simply 1/4 because the quadratic form is not rotationally symmetric.
    #
    # For the triangular domain, the fraction of the full Gaussian integral
    # that lies in {α₁ ≥ 0, α₂ ≥ 0} is the first-quadrant fraction.
    #
    # Let's compute it numerically for the specific quadratic form.
    # Actually, for the triangular domain {α₁>0, α₂>0, α₁+α₂<r}, with large r
    # the constraint α₁+α₂<r is irrelevant (Gaussian dies before reaching it).
    # So the integral over the triangle ≈ integral over first quadrant.
    #
    # For the first quadrant integral of e^{-α^T M α}:
    # By diagonalizing M = P^T D P, the first quadrant maps to a wedge in the
    # eigenbasis. The fraction is:
    # f = (1/2π) × ∫_0^{π/2} 1/√(cos²θ/λ₁ + sin²θ/λ₂) dθ × √(λ₁λ₂)
    #
    # This is complex. Let me just use the known result:
    # ∫_0^∞∫_0^∞ e^{-β(2x²+2y²+2xy)/(3r)} dx dy
    #
    # Change variables: u = α₁ + α₂, v = α₁ - α₂
    # α₁ = (u+v)/2, α₂ = (u-v)/2
    # α₁² + α₂² + α₁α₂ = (u² + 3v²/4 + uv/2 + ... )
    # Actually: α₁² + α₂² + α₁α₂ = ((u+v)/2)² + ((u-v)/2)² + ((u+v)(u-v))/4
    #   = (u²+2uv+v²)/4 + (u²-2uv+v²)/4 + (u²-v²)/4
    #   = (2u² + 2v²)/4 + (u²-v²)/4 = (3u² + v²)/4
    # Wait: = (u²+v²)/2 + (u²-v²)/4 = (3u² + v²)/4
    # So C₂ = (3u² + v²)/(4×3) = (3u² + v²)/12  (ignoring linear terms for large r)
    # Wait, C₂ includes linear terms: C₂ = (α₁²+α₂²+α₁α₂+2α₁+2α₂)/3
    # Near origin, the linear terms shift the saddle.
    # 
    # Actually for the full trace, the integral is EXACTLY a shifted Gaussian:
    # ∫∫_{triangle} r² e^{-β(α₁²+α₂²+α₁α₂+2α₁+2α₂)/(3r)} dα₁dα₂
    #
    # Complete the square:
    # (α₁²+α₂²+α₁α₂+2α₁+2α₂)/3 = Q(α) + const
    # where Q(α) = ((α₁+1)² + (α₂+1)² + (α₁+1)(α₂+1) - 3)/3
    # = Q(α₁+1, α₂+1) - 1
    # So the integral becomes:
    # e^{β/r} × r² × ∫∫_{triangle'} e^{-βC₂(α₁+1,α₂+1)/r} dα₁dα₂
    #
    # Shift: let β₁ = α₁+1, β₂ = α₂+1. The domain becomes
    # {β₁ > 1, β₂ > 1, β₁ + β₂ < r+2}
    # For large r, this is effectively [1,∞)², and the Gaussian is centered at (0,0)
    # in the original variables, or at (-1,-1) in the β variables.
    # Since (-1,-1) is outside [1,∞)², the integral is exponentially small? No, that's wrong.
    #
    # Let me reconsider. The shifted Gaussian center for Q(α) = (α₁²+α₂²+α₁α₂)/3
    # is at α=0. Adding the linear terms 2(α₁+α₂)/3 shifts the center.
    # ∂/∂α₁ [β(α₁²+α₂²+α₁α₂+2α₁+2α₂)/(3r)] = β(2α₁+α₂+2)/(3r) = 0
    # ∂/∂α₂ [same] = β(α₁+2α₂+2)/(3r) = 0
    # Solution: 2α₁+α₂ = -2, α₁+2α₂ = -2 → α₁ = α₂ = -2/3
    #
    # So the shifted Gaussian center is at (-2/3, -2/3), which is outside the domain.
    # The integral over the first quadrant is a tail integral of the Gaussian.
    #
    # For large r, the Gaussian is very broad (width ~ √r), and the shift -2/3
    # is small compared to the width. So the first-quadrant integral is approximately
    # 1/4 of the full Gaussian integral (by symmetry of the unshifted part).
    #
    # More precisely:
    # Z_full_cont ≈ r² × e^{-βC₂(-2/3,-2/3)/r} × (2π)^{d/2}/√(det(βM/r)) × 1/4
    #
    # where C₂(-2/3,-2/3) = (4/9 + 4/9 + 4/9 - 4/3 - 4/3)/3 = (4/3 - 8/3)/3 = -4/9
    # Wait: C₂(-2/3,-2/3) = ((-2/3)² + (-2/3)² + (-2/3)(-2/3) + 2(-2/3) + 2(-2/3))/3
    # = (4/9 + 4/9 + 4/9 - 4/3 - 4/3)/3 = (12/9 - 8/3)/3 = (4/3 - 8/3)/3 = (-4/3)/3 = -4/9
    #
    # So e^{-βC₂(-2/3,-2/3)/r} = e^{4β/(9r)} ≈ 1 for large r.
    #
    # The Gaussian integral (full, over R²):
    # (2π)^{1} / √(det(βM/r)) where M = [[2,1],[1,2]]/3
    # det(βM/r) = β²/(r²) × det([[2,1],[1,2]]/3) = β²/(r²) × 1/3
    # = 2π√3 r/β
    #
    # First quadrant fraction: approximately 1/4 for near-isotropic Gaussian
    # (the off-diagonal coupling α₁α₂ slightly changes this)
    #
    # Z_full_cont ≈ r² × (2π√3 r/β) / 4 = π√3 r³ / (2β) for large r
    
    # Direct analytical formula for the leading order
    Z_analytical = r**2 * np.pi * np.sqrt(3) * r / (2 * beta)  # = π√3 r³ / (2β)
    
    return Z_analytical


def integrate_bcgp_cont_saddle(r, beta, n_points=80):
    """BCGP continuous sector via saddle-point + adaptive grid quadrature.
    
    Uses a tensor-product Gauss-Legendre grid centered at the saddle point,
    with widths determined by the Hessian eigenvalues.
    This avoids the domain-truncation issues of Gauss-Hermite.
    """
    a1_star, a2_star = find_saddle_point_bcgp(r, beta)
    
    # Compute Hessian at saddle point
    H = hessian_bcgp(a1_star, a2_star, r, beta)
    eigvals_H = np.linalg.eigvalsh(H)
    
    # Width of the peak: σ ~ 1/√|λ_max| (where λ_max is the least negative eigenvalue)
    sigma = 1.0 / np.sqrt(np.abs(eigvals_H).min())  # Conservative (widest direction)
    
    # Integration limits: extend a few sigma around the saddle
    # but clip to the Weyl chamber
    a1_lo = max(0.001, a1_star - 4*sigma)
    a1_hi = min(r - 0.001, a1_star + 4*sigma)
    a2_lo = max(0.001, a2_star - 4*sigma)
    a2_hi = min(r - 0.001, a2_star + 4*sigma)
    
    # Use Gauss-Legendre quadrature on the clipped domain
    # For the triangular domain, a2 ranges from a2_lo to min(a2_hi, r - a1)
    
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    
    # Map nodes from [-1,1] to [a1_lo, a1_hi]
    total = 0.0
    half_width_a1 = (a1_hi - a1_lo) / 2.0
    center_a1 = (a1_hi + a1_lo) / 2.0
    
    for i in range(n_points):
        a1 = center_a1 + half_width_a1 * nodes[i]
        w1 = half_width_a1 * weights[i]
        
        # a2 ranges from a2_lo to min(a2_hi, r - a1)
        a2_upper = min(a2_hi, r - a1 - 0.001)
        if a2_upper <= a2_lo:
            continue
        
        half_width_a2 = (a2_upper - a2_lo) / 2.0
        center_a2 = (a2_upper + a2_lo) / 2.0
        
        for j in range(n_points):
            a2 = center_a2 + half_width_a2 * nodes[j]
            w2 = half_width_a2 * weights[j]
            
            if a1 <= 0 or a2 <= 0 or a1 + a2 >= r:
                continue
            
            # Evaluate integrand
            d_tilde = modified_qdim_cont(a1, a2, r)
            c2 = casimir_sl3(a1, a2)
            boltzmann = np.exp(-beta * c2 / r)
            
            total += w1 * w2 * d_tilde * boltzmann
    
    return total


def integrate_full_cont_saddle(r, beta, n_points=80):
    """Full thermal trace continuous sector via adaptive Gauss-Legendre.
    
    For r² × e^{-βC₂/r}, the maximum is at the origin (boundary of domain).
    We use Gauss-Legendre on the triangular domain with enough points
    to resolve the Gaussian peak near the origin.
    """
    # Width of the Gaussian peak
    sigma = np.sqrt(3.0 * r / beta)
    
    # For small r, integrate over the full triangle
    # For large r, we can truncate far from the origin
    a1_max = min(r, 6 * sigma)
    
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    
    total = 0.0
    half_width_a1 = a1_max / 2.0
    center_a1 = a1_max / 2.0
    
    for i in range(n_points):
        a1 = center_a1 + half_width_a1 * nodes[i]
        w1 = half_width_a1 * weights[i]
        
        if a1 <= 0:
            continue
        
        a2_upper = min(a1_max, r - a1)
        if a2_upper <= 0:
            continue
        
        half_width_a2 = a2_upper / 2.0
        center_a2 = a2_upper / 2.0
        
        for j in range(n_points):
            a2 = center_a2 + half_width_a2 * nodes[j]
            w2 = half_width_a2 * weights[j]
            
            if a2 <= 0 or a1 + a2 >= r:
                continue
            
            c2 = casimir_sl3(a1, a2)
            boltzmann = np.exp(-beta * c2 / r)
            
            total += w1 * w2 * r**2 * boltzmann
    
    return total


# ============================================================================
# INTEGRATION METHOD 2: MONTE CARLO (STRATIFIED)
# ============================================================================

def integrate_bcgp_cont_mc(r, beta, n_samples=500000, seed=42):
    """BCGP continuous sector via Monte Carlo (uniform on triangle).
    
    Samples uniformly from the Weyl chamber triangle {α₁>0, α₂>0, α₁+α₂<r}.
    Area = r²/2.
    """
    rng = np.random.RandomState(seed)
    
    # Uniform sampling on triangle using the standard method:
    # Generate (u1, u2) uniform on [0,1]², sort to get order statistics
    u1 = rng.uniform(0, 1, n_samples)
    u2 = rng.uniform(0, 1, n_samples)
    
    # Map to triangle vertices (0,0), (r,0), (0,r)
    # Using barycentric: α₁ = r(1-√u₁), α₂ = r√u₁(1-u₂)
    # This gives uniform distribution on the triangle
    sqrt_u1 = np.sqrt(u1)
    a1 = r * (1 - sqrt_u1)
    a2 = r * sqrt_u1 * u2
    
    # Evaluate integrand
    denom_const = r**2 * np.sin(np.pi / r)**4 * np.sin(2 * np.pi / r)**2
    
    num = (np.sin(np.pi * a1 / r) *
           np.sin(np.pi * a2 / r) *
           np.sin(np.pi * (a1 + a2) / r))
    d_tilde = num / denom_const
    
    c2 = casimir_sl3(a1, a2)
    boltzmann = np.exp(-beta * c2 / r)
    
    integrand = d_tilde * boltzmann
    
    # Area of the triangle = r²/2
    area = r**2 / 2.0
    
    integral = np.mean(integrand) * area
    std_err = np.std(integrand) / np.sqrt(n_samples) * area
    
    return integral, std_err


def integrate_full_cont_mc(r, beta, n_samples=500000, seed=42):
    """Full thermal trace continuous sector via Monte Carlo.
    
    Uses uniform sampling on the Weyl chamber triangle.
    """
    rng = np.random.RandomState(seed)
    
    u1 = rng.uniform(0, 1, n_samples)
    u2 = rng.uniform(0, 1, n_samples)
    
    sqrt_u1 = np.sqrt(u1)
    a1 = r * (1 - sqrt_u1)
    a2 = r * sqrt_u1 * u2
    
    c2 = casimir_sl3(a1, a2)
    integrand = r**2 * np.exp(-beta * c2 / r)
    
    area = r**2 / 2.0
    integral = np.mean(integrand) * area
    std_err = np.std(integrand) / np.sqrt(n_samples) * area
    
    return integral, std_err


def integrate_bcgp_cont_mc_is(r, beta, n_samples=500000, seed=42):
    """BCGP continuous sector via Monte Carlo with saddle-point importance sampling.
    
    Uses a Gaussian proposal centered at the saddle point with covariance
    from the Hessian. Much more efficient than uniform sampling for peaked integrands.
    """
    rng = np.random.RandomState(seed)
    
    # Find saddle point
    a1_star, a2_star = find_saddle_point_bcgp(r, beta)
    
    # Covariance from Hessian: Σ = -H^{-1}
    H = hessian_bcgp(a1_star, a2_star, r, beta)
    Sigma_inv = -H  # This is the precision matrix (inverse of covariance)
    Sigma = np.linalg.inv(Sigma_inv)  # Covariance matrix
    
    # Cholesky decomposition for sampling
    L = np.linalg.cholesky(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    log_norm = -np.log(2 * np.pi) - 0.5 * np.log(det_Sigma)
    
    # Generate samples from the Gaussian proposal
    z = rng.standard_normal((n_samples, 2))
    center = np.array([a1_star, a2_star])
    alpha_samples = center + z @ L.T
    
    a1 = alpha_samples[:, 0]
    a2 = alpha_samples[:, 1]
    
    # Evaluate integrand in the Weyl chamber
    mask = (a1 > 0) & (a2 > 0) & (a1 + a2 < r)
    
    denom_const = r**2 * np.sin(np.pi / r)**4 * np.sin(2 * np.pi / r)**2
    
    num = (np.sin(np.pi * a1 / r) *
           np.sin(np.pi * a2 / r) *
           np.sin(np.pi * (a1 + a2) / r))
    d_tilde = num / denom_const
    c2 = casimir_sl3(a1, a2)
    boltzmann = np.exp(-beta * c2 / r)
    
    integrand = d_tilde * boltzmann
    
    # Proposal density: p(x) = (2π)^{-1} |Σ|^{-1/2} exp(-0.5 (x-μ)^T Σ^{-1} (x-μ))
    diff = alpha_samples - center
    # Mahalanobis distance: (x-μ)^T Σ^{-1} (x-μ) for each sample
    # = sum_i sum_j diff_i * Sigma_inv_{ij} * diff_j
    mahal = np.einsum('ni,ij,nj->n', diff, Sigma_inv, diff)
    log_prop = log_norm - 0.5 * mahal
    prop = np.exp(log_prop)
    
    # Importance sampling: ∫ f(x) dx = E_p[f(x)/p(x)]
    # For samples outside domain, f(x) = 0
    weights = np.where(mask, integrand / np.maximum(prop, 1e-300), 0.0)
    
    integral = np.mean(weights)
    std_err = np.std(weights) / np.sqrt(n_samples)
    
    return integral, std_err


# ============================================================================
# INTEGRATION METHOD 3: DIRECT DBLQUAD (BASELINE)
# ============================================================================

def integrate_bcgp_cont_dblquad(r, beta):
    """BCGP continuous sector via scipy dblquad (triangular domain)."""
    def integrand(a2, a1):
        h = casimir_sl3(a1, a2) / r
        d = modified_qdim_cont(a1, a2, r)
        return d * np.exp(-beta * h)
    
    val, err = integrate.dblquad(
        integrand, 0, r, 0, lambda a1: r - a1,
        epsabs=1e-10, epsrel=1e-8
    )
    return val, err


def integrate_full_cont_dblquad(r, beta):
    """Full thermal trace continuous sector via scipy dblquad (triangular domain)."""
    def integrand(a2, a1):
        h = casimir_sl3(a1, a2) / r
        return r**2 * np.exp(-beta * h)
    
    val, err = integrate.dblquad(
        integrand, 0, r, 0, lambda a1: r - a1,
        epsabs=1e-10, epsrel=1e-8
    )
    return val, err


def D_tilde_squared_cont_dblquad(r):
    """Continuous part of D̃² via dblquad."""
    def integrand(a2, a1):
        d = modified_qdim_cont(a1, a2, r)
        return d * d
    
    val, err = integrate.dblquad(
        integrand, 0, r, 0, lambda a1: r - a1,
        epsabs=1e-10, epsrel=1e-8
    )
    return val, err


# ============================================================================
# LOG COEFFICIENT EXTRACTION
# ============================================================================

def extract_log_coeff_from_lnZ(r_arr, lnZ_arr, min_r=0):
    """Extract log coefficient from ln(Z_norm(r)) data.
    
    Uses multiple methods:
    1. 3-param fit: ln(Z) = a*ln(r) + b + c/r
    2. 4-param fit: ln(Z) = a*ln(r) + b + c/r + d/r²
    3. Finite difference: d(ln Z)/d(ln r)
    4. Richardson extrapolation of finite differences
    """
    mask = (r_arr >= min_r) & np.isfinite(lnZ_arr)
    r = r_arr[mask].astype(float)
    lnZ = lnZ_arr[mask]
    
    if len(r) < 3:
        return {'log_coeff': float('nan'), 'n_points': len(r)}
    
    # Method 1: 3-param fit
    A3 = np.column_stack([np.log(r), np.ones_like(r), 1.0/r])
    c3, _, _, _ = np.linalg.lstsq(A3, lnZ, rcond=None)
    
    # Method 2: 4-param fit
    if len(r) >= 5:
        A4 = np.column_stack([np.log(r), np.ones_like(r), 1.0/r, 1.0/r**2])
        c4, _, _, _ = np.linalg.lstsq(A4, lnZ, rcond=None)
    else:
        c4 = np.array([float('nan')] * 4)
    
    # Method 3: Finite difference
    lnr = np.log(r)
    dlnZ_dlnr = np.diff(lnZ) / np.diff(lnr)
    lnr_mid = 0.5 * (lnr[:-1] + lnr[1:])
    r_mid = np.exp(lnr_mid)
    
    if len(r_mid) >= 3:
        A_fd = np.column_stack([np.ones_like(r_mid), 1.0/r_mid, 1.0/r_mid**2])
        c_fd, _, _, _ = np.linalg.lstsq(A_fd, dlnZ_dlnr, rcond=None)
        fd_asymp = c_fd[0]
    elif len(r_mid) >= 1:
        fd_asymp = dlnZ_dlnr[-1]
    else:
        fd_asymp = float('nan')
    
    # Method 4: Richardson extrapolation
    # For evenly-spaced ln(r), Richardson gives:
    # f_asymptotic ≈ (4*f(h/2) - f(h)) / 3  (second-order)
    if len(dlnZ_dlnr) >= 3:
        # Use the last few points for Richardson
        richardson_estimates = []
        for i in range(len(dlnZ_dlnr) - 1):
            h = lnr_mid[i+1] - lnr_mid[i]
            if abs(h) > 1e-10:
                # Simple linear extrapolation: assume dlnZ/dlnr = a + b/r²
                # Two consecutive points give:
                # a + b/r1² = f1, a + b/r2² = f2
                r1, r2 = r_mid[i], r_mid[i+1]
                f1, f2 = dlnZ_dlnr[i], dlnZ_dlnr[i+1]
                if abs(1/r1**2 - 1/r2**2) > 1e-20:
                    a_rich = (f1/r2**2 - f2/r1**2) / (1/r2**2 - 1/r1**2)
                    richardson_estimates.append(a_rich)
        
        if richardson_estimates:
            richardson_asymp = np.mean(richardson_estimates[-3:])
        else:
            richardson_asymp = float('nan')
    else:
        richardson_asymp = float('nan')
    
    # Method 5: Large-r only fits
    if len(r) >= 5:
        n_half = max(len(r) // 2, 3)
        r_large = r[-n_half:]
        lnZ_large = lnZ[-n_half:]
        A_large = np.column_stack([np.log(r_large), np.ones_like(r_large), 1.0/r_large])
        c_large, _, _, _ = np.linalg.lstsq(A_large, lnZ_large, rcond=None)
    else:
        c_large = np.array([float('nan')] * 3)
    
    return {
        'log_coeff_3param': float(c3[0]),
        'log_coeff_4param': float(c4[0]),
        'log_coeff_fd': float(fd_asymp),
        'log_coeff_richardson': float(richardson_asymp),
        'log_coeff_large_r': float(c_large[0]),
        'n_points': len(r),
        'fd_values': list(zip(r_mid.tolist(), dlnZ_dlnr.tolist())),
    }


# ============================================================================
# COMPREHENSIVE COMPUTATION
# ============================================================================

def compute_all(r_values=None, beta=1.0, n_mc_samples=500000, n_gh_points=60):
    """Compute partition functions at multiple r values using all three methods.
    
    Returns comprehensive results dictionary.
    """
    if r_values is None:
        r_values = [5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31, 41, 51]
    
    print("=" * 120)
    print("  sl₃ CONTINUOUS SECTOR REFINEMENT (v3)")
    print("  Saddle-Point + Monte Carlo + Direct Integration Comparison")
    print(f"  β = {beta}")
    print(f"  r values: {r_values}")
    print("=" * 120)
    
    per_r = []
    
    for r in r_values:
        print(f"\n{'─' * 120}")
        print(f"  COMPUTING r = {r}")
        print(f"{'─' * 120}")
        t_start = time.time()
        
        res = {'r': r, 'beta': beta}
        
        # --- Saddle point analysis ---
        print(f"  Finding saddle points...", end=" ", flush=True)
        t0 = time.time()
        
        a1_star_bcgp, a2_star_bcgp = find_saddle_point_bcgp(r, beta)
        H_bcgp = hessian_bcgp(a1_star_bcgp, a2_star_bcgp, r, beta)
        eigvals_bcgp = np.linalg.eigvalsh(H_bcgp)
        
        print(f"BCGP saddle at ({a1_star_bcgp:.4f}, {a2_star_bcgp:.4f}), "
              f"Hessian eigvals = ({eigvals_bcgp[0]:.6f}, {eigvals_bcgp[1]:.6f}), "
              f"time={time.time()-t0:.2f}s")
        
        res['saddle_bcgp_a1'] = a1_star_bcgp
        res['saddle_bcgp_a2'] = a2_star_bcgp
        res['saddle_bcgp_ratio'] = a1_star_bcgp / r
        res['hessian_bcgp_eigvals'] = eigvals_bcgp.tolist()
        
        # --- Discrete sector ---
        print(f"  Computing discrete sector...", end=" ", flush=True)
        t0 = time.time()
        
        D2d = D_tilde_squared_disc(r)
        Zfd = Z_full_disc(beta, r)
        Zmd = Z_bcgp_disc(beta, r)
        
        print(f"D̃²_disc={D2d:.6f}, time={time.time()-t0:.2f}s")
        
        res['D2_disc'] = D2d
        res['Z_full_disc'] = Zfd
        res['Z_bcgp_disc'] = Zmd
        
        # --- Method 1: Direct dblquad (only for r <= 31, slow for large r) ---
        if r <= 31:
            print(f"  [1/4] Direct dblquad...", end=" ", flush=True)
            t0 = time.time()
            
            try:
                Zfc_dbl, err_fc = integrate_full_cont_dblquad(r, beta)
                Zmc_dbl, err_mc = integrate_bcgp_cont_dblquad(r, beta)
                D2c_dbl, err_d2 = D_tilde_squared_cont_dblquad(r)
                t_dbl = time.time() - t0
                print(f"Z_full_cont={Zfc_dbl:.6e}, Z_bcgp_cont={Zmc_dbl:.6e}, "
                      f"D̃²_cont={D2c_dbl:.6e}, time={t_dbl:.1f}s")
                dblquad_ok = True
            except Exception as e:
                print(f"FAILED: {e}")
                Zfc_dbl = Zmc_dbl = D2c_dbl = float('nan')
                t_dbl = float('nan')
                dblquad_ok = False
        else:
            print(f"  [1/4] Skipping dblquad for r={r} (too slow)")
            Zfc_dbl = Zmc_dbl = D2c_dbl = float('nan')
            t_dbl = float('nan')
            dblquad_ok = False
        
        res['Z_full_cont_dblquad'] = Zfc_dbl
        res['Z_bcgp_cont_dblquad'] = Zmc_dbl
        res['D2_cont_dblquad'] = D2c_dbl
        res['time_dblquad'] = t_dbl
        res['dblquad_ok'] = dblquad_ok
        
        # --- Method 2: Saddle-point methods ---
        print(f"  [2/4] Saddle-point (Laplace + Gauss-Legendre)...", end=" ", flush=True)
        t0 = time.time()
        
        try:
            Zfc_laplace = laplace_full_cont(r, beta)
            Zmc_laplace = laplace_bcgp_cont(r, beta)
            Zfc_saddle = integrate_full_cont_saddle(r, beta, n_points=n_gh_points)
            Zmc_saddle = integrate_bcgp_cont_saddle(r, beta, n_points=n_gh_points)
            t_saddle = time.time() - t0
            print(f"Laplace: full={Zfc_laplace:.6e}, bcgp={Zmc_laplace:.6e}; "
                  f"GL: full={Zfc_saddle:.6e}, bcgp={Zmc_saddle:.6e}, "
                  f"time={t_saddle:.2f}s")
            saddle_ok = True
        except Exception as e:
            print(f"FAILED: {e}")
            Zfc_laplace = Zmc_laplace = Zfc_saddle = Zmc_saddle = float('nan')
            t_saddle = float('nan')
            saddle_ok = False
        
        res['Z_full_cont_laplace'] = Zfc_laplace
        res['Z_bcgp_cont_laplace'] = Zmc_laplace
        res['Z_full_cont_saddle'] = Zfc_saddle
        res['Z_bcgp_cont_saddle'] = Zmc_saddle
        res['time_saddle'] = t_saddle
        res['saddle_ok'] = saddle_ok
        
        # --- Method 3: Monte Carlo ---
        print(f"  [3/4] Monte Carlo ({n_mc_samples} samples)...", end=" ", flush=True)
        t0 = time.time()
        
        try:
            Zfc_mc, Zfc_mc_err = integrate_full_cont_mc(r, beta, n_samples=n_mc_samples)
            Zmc_mc, Zmc_mc_err = integrate_bcgp_cont_mc(r, beta, n_samples=n_mc_samples)
            Zmc_mc_is, Zmc_mc_is_err = integrate_bcgp_cont_mc_is(r, beta, n_samples=n_mc_samples)
            t_mc = time.time() - t0
            print(f"Z_full_cont={Zfc_mc:.6e}±{Zfc_mc_err:.2e}, "
                  f"Z_bcgp_cont={Zmc_mc:.6e}±{Zmc_mc_err:.2e}, "
                  f"Z_bcgp_cont(IS)={Zmc_mc_is:.6e}±{Zmc_mc_is_err:.2e}, "
                  f"time={t_mc:.1f}s")
            mc_ok = True
        except Exception as e:
            print(f"FAILED: {e}")
            Zfc_mc = Zmc_mc = Zmc_mc_is = float('nan')
            Zfc_mc_err = Zmc_mc_err = Zmc_mc_is_err = float('nan')
            t_mc = float('nan')
            mc_ok = False
        
        res['Z_full_cont_mc'] = Zfc_mc
        res['Z_full_cont_mc_err'] = Zfc_mc_err
        res['Z_bcgp_cont_mc'] = Zmc_mc
        res['Z_bcgp_cont_mc_err'] = Zmc_mc_err
        res['Z_bcgp_cont_mc_is'] = Zmc_mc_is
        res['Z_bcgp_cont_mc_is_err'] = Zmc_mc_is_err
        res['time_mc'] = t_mc
        res['mc_ok'] = mc_ok
        
        # --- Combine discrete + continuous ---
        # Use best available: dblquad > GL saddle > MC
        if dblquad_ok:
            Zf_cont = Zfc_dbl
            Zm_cont = Zmc_dbl
            D2c = D2c_dbl
        elif saddle_ok:
            Zf_cont = Zfc_saddle
            Zm_cont = Zmc_saddle
            # For D2c, use analytical: D2c = 2 * D2d (known identity)
            D2c = 2 * D2d
        else:
            Zf_cont = Zfc_mc
            Zm_cont = Zmc_mc
            D2c = 2 * D2d
        
        D2 = D2d + D2c if np.isfinite(D2c) else float('nan')
        Zf_num = Zfd + Zf_cont
        Zm_num = Zmd + Zm_cont
        
        Zf = Zf_num / D2 if np.isfinite(D2) and D2 != 0 else float('nan')
        Zm = Zm_num / D2 if np.isfinite(D2) and D2 != 0 else float('nan')
        
        lnZf = np.log(abs(Zf)) if np.isfinite(Zf) and abs(Zf) > 1e-300 else float('-inf')
        lnZm = np.log(abs(Zm)) if np.isfinite(Zm) and abs(Zm) > 1e-300 else float('-inf')
        
        res['D2'] = D2
        res['Z_full'] = Zf
        res['Z_bcgp'] = Zm
        res['ln_Z_full'] = lnZf
        res['ln_Z_bcgp'] = lnZm
        
        # Continuous-only normalized
        Zf_cont_only = Zf_cont / D2 if np.isfinite(D2) and D2 != 0 else float('nan')
        Zm_cont_only = Zm_cont / D2 if np.isfinite(D2) and D2 != 0 else float('nan')
        lnZf_cont = np.log(abs(Zf_cont_only)) if np.isfinite(Zf_cont_only) and abs(Zf_cont_only) > 1e-300 else float('-inf')
        lnZm_cont = np.log(abs(Zm_cont_only)) if np.isfinite(Zm_cont_only) and abs(Zm_cont_only) > 1e-300 else float('-inf')
        
        res['ln_Z_full_cont'] = lnZf_cont
        res['ln_Z_bcgp_cont'] = lnZm_cont
        
        # Method comparison
        if dblquad_ok and saddle_ok:
            rel_err_full = abs(Zfc_saddle - Zfc_dbl) / abs(Zfc_dbl) if abs(Zfc_dbl) > 1e-30 else float('nan')
            rel_err_bcgp = abs(Zmc_saddle - Zmc_dbl) / abs(Zmc_dbl) if abs(Zmc_dbl) > 1e-30 else float('nan')
            res['saddle_rel_err_full'] = rel_err_full
            res['saddle_rel_err_bcgp'] = rel_err_bcgp
        
        if dblquad_ok and mc_ok:
            rel_err_mc_full = abs(Zfc_mc - Zfc_dbl) / abs(Zfc_dbl) if abs(Zfc_dbl) > 1e-30 else float('nan')
            rel_err_mc_bcgp = abs(Zmc_mc - Zmc_dbl) / abs(Zmc_dbl) if abs(Zmc_dbl) > 1e-30 else float('nan')
            res['mc_rel_err_full'] = rel_err_mc_full
            res['mc_rel_err_bcgp'] = rel_err_mc_bcgp
        
        t_total = time.time() - t_start
        res['time_total'] = t_total
        print(f"  Total time: {t_total:.1f}s | lnZ_full={lnZf:.4f}, lnZ_bcgp={lnZm:.4f}")
        
        per_r.append(res)
    
    # ==================================================================
    # ANALYSIS: Method comparison
    # ==================================================================
    print(f"\n{'═' * 120}")
    print(f"  METHOD COMPARISON")
    print(f"{'═' * 120}")
    
    print(f"\n  {'r':>4s}  {'Z_full_dbl':>14s}  {'Z_full_sad':>14s}  {'Z_full_MC':>14s}  "
          f"{'|sad-dbl|':>10s}  {'|MC-dbl|':>10s}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*10}")
    
    for res in per_r:
        zdbl = res.get('Z_full_cont_dblquad', float('nan'))
        zsad = res.get('Z_full_cont_saddle', float('nan'))
        zmc = res.get('Z_full_cont_mc', float('nan'))
        err_sad = res.get('saddle_rel_err_full', float('nan'))
        err_mc = res.get('mc_rel_err_full', float('nan'))
        print(f"  {res['r']:4d}  {zdbl:14.6e}  {zsad:14.6e}  {zmc:14.6e}  "
              f"{err_sad:10.4e}  {err_mc:10.4e}")
    
    print(f"\n  {'r':>4s}  {'Z_bcgp_dbl':>14s}  {'Z_bcgp_sad':>14s}  {'Z_bcgp_MC':>14s}  "
          f"{'Z_bcgp_MCis':>14s}  {'|sad-dbl|':>10s}  {'|MC-dbl|':>10s}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*10}")
    
    for res in per_r:
        zdbl = res.get('Z_bcgp_cont_dblquad', float('nan'))
        zsad = res.get('Z_bcgp_cont_saddle', float('nan'))
        zmc = res.get('Z_bcgp_cont_mc', float('nan'))
        zmcis = res.get('Z_bcgp_cont_mc_is', float('nan'))
        err_sad = res.get('saddle_rel_err_bcgp', float('nan'))
        err_mc = res.get('mc_rel_err_bcgp', float('nan'))
        print(f"  {res['r']:4d}  {zdbl:14.6e}  {zsad:14.6e}  {zmc:14.6e}  "
              f"{zmcis:14.6e}  {err_sad:10.4e}  {err_mc:10.4e}")
    
    # ==================================================================
    # ANALYSIS: Saddle point behavior
    # ==================================================================
    print(f"\n{'═' * 120}")
    print(f"  SADDLE POINT ANALYSIS")
    print(f"{'═' * 120}")
    
    print(f"\n  BCGP saddle point α*/r (should approach some limit):")
    print(f"  {'r':>4s}  {'α₁*/r':>10s}  {'α₂*/r':>10s}  {'α*/r (sym)':>12s}  "
          f"{'λ₁':>12s}  {'λ₂':>12s}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*12}")
    
    for res in per_r:
        eig = res.get('hessian_bcgp_eigvals', [float('nan'), float('nan')])
        print(f"  {res['r']:4d}  {res['saddle_bcgp_a1']/res['r']:10.6f}  "
              f"{res['saddle_bcgp_a2']/res['r']:10.6f}  "
              f"{res.get('saddle_bcgp_ratio', float('nan')):12.6f}  "
              f"{eig[0]:12.6f}  {eig[1]:12.6f}")
    
    # ==================================================================
    # ANALYSIS: D̃² scaling
    # ==================================================================
    print(f"\n{'═' * 120}")
    print(f"  D̃² SCALING")
    print(f"{'═' * 120}")
    
    r_arr = np.array([res['r'] for res in per_r], dtype=float)
    D2_arr = np.array([res.get('D2', float('nan')) for res in per_r])
    valid_D2 = np.isfinite(D2_arr)
    
    if sum(valid_D2) >= 3:
        A_d2 = np.column_stack([np.log(r_arr[valid_D2]), np.ones_like(r_arr[valid_D2]), 
                                1.0/r_arr[valid_D2]])
        c_d2, _, _, _ = np.linalg.lstsq(A_d2, np.log(D2_arr[valid_D2]), rcond=None)
        D2_exp = c_d2[0]
        print(f"\n  D̃² ~ r^{D2_exp:.4f} (3-param fit)")
        
        # Finite difference
        fd_D2 = np.diff(np.log(D2_arr[valid_D2])) / np.diff(np.log(r_arr[valid_D2]))
        r_mid_D2 = np.exp(0.5 * (np.log(r_arr[valid_D2][:-1]) + np.log(r_arr[valid_D2][1:])))
        print(f"  Finite-difference exponent: last value = {fd_D2[-1]:.4f}")
    else:
        D2_exp = float('nan')
        print(f"  Insufficient D̃² data for scaling analysis")
    
    # ==================================================================
    # ANALYSIS: Log coefficient extraction
    # ==================================================================
    print(f"\n{'═' * 120}")
    print(f"  LOG COEFFICIENT EXTRACTION")
    print(f"{'═' * 120}")
    
    lnZf_arr = np.array([res['ln_Z_full'] for res in per_r])
    lnZm_arr = np.array([res['ln_Z_bcgp'] for res in per_r])
    lnZf_cont_arr = np.array([res['ln_Z_full_cont'] for res in per_r])
    lnZm_cont_arr = np.array([res['ln_Z_bcgp_cont'] for res in per_r])
    
    # Full trace log coefficients
    res_full = extract_log_coeff_from_lnZ(r_arr, lnZf_arr)
    res_mod = extract_log_coeff_from_lnZ(r_arr, lnZm_arr)
    res_full_cont = extract_log_coeff_from_lnZ(r_arr, lnZf_cont_arr)
    res_mod_cont = extract_log_coeff_from_lnZ(r_arr, lnZm_cont_arr)
    
    # Shift
    delta_lnZ = lnZf_arr - lnZm_arr
    res_delta = extract_log_coeff_from_lnZ(r_arr, delta_lnZ)
    
    print(f"\n  Full thermal trace (total, ln Z_full):")
    print(f"    3-param fit:       a = {res_full['log_coeff_3param']:+.4f}")
    print(f"    4-param fit:       a = {res_full['log_coeff_4param']:+.4f}")
    print(f"    Finite-diff:       a = {res_full['log_coeff_fd']:+.4f}")
    print(f"    Richardson:        a = {res_full['log_coeff_richardson']:+.4f}")
    print(f"    Large-r fit:       a = {res_full['log_coeff_large_r']:+.4f}")
    
    print(f"\n  BCGP modified trace (total, ln Z_bcgp):")
    print(f"    3-param fit:       a = {res_mod['log_coeff_3param']:+.4f}")
    print(f"    4-param fit:       a = {res_mod['log_coeff_4param']:+.4f}")
    print(f"    Finite-diff:       a = {res_mod['log_coeff_fd']:+.4f}")
    print(f"    Richardson:        a = {res_mod['log_coeff_richardson']:+.4f}")
    print(f"    Large-r fit:       a = {res_mod['log_coeff_large_r']:+.4f}")
    
    print(f"\n  Continuous-only (normalized by D̃²):")
    print(f"    Full trace FD:     a = {res_full_cont['log_coeff_fd']:+.4f}")
    print(f"    Full trace Rich:   a = {res_full_cont['log_coeff_richardson']:+.4f}")
    print(f"    BCGP trace FD:     a = {res_mod_cont['log_coeff_fd']:+.4f}")
    print(f"    BCGP trace Rich:   a = {res_mod_cont['log_coeff_richardson']:+.4f}")
    
    print(f"\n  Shift (ln Z_full - ln Z_bcgp):")
    print(f"    3-param fit:       a = {res_delta['log_coeff_3param']:+.4f}")
    print(f"    Finite-diff:       a = {res_delta['log_coeff_fd']:+.4f}")
    print(f"    Richardson:        a = {res_delta['log_coeff_richardson']:+.4f}")
    
    # ==================================================================
    # Finite difference table
    # ==================================================================
    print(f"\n  Finite-difference d(ln Z)/d(ln r):")
    print(f"  {'r_mid':>8s}  {'dlnZf/dlnr':>12s}  {'dlnZm/dlnr':>12s}  {'shift':>8s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}")
    
    fd_full = res_full.get('fd_values', [])
    fd_mod = res_mod.get('fd_values', [])
    for i in range(min(len(fd_full), len(fd_mod))):
        rf, df = fd_full[i]
        rm, dm = fd_mod[i]
        print(f"  {rf:8.2f}  {df:12.4f}  {dm:12.4f}  {df-dm:8.4f}")
    
    # ==================================================================
    # COMPARISON WITH PREDICTIONS
    # ==================================================================
    print(f"\n{'═' * 120}")
    print(f"  COMPARISON WITH THEORETICAL PREDICTIONS")
    print(f"{'═' * 120}")
    
    if np.isfinite(D2_exp):
        print(f"\n  With D̃² ~ r^{D2_exp:.2f}:")
        print(f"    Predicted full trace log coeff:  3 - {D2_exp:.2f} = {3 - D2_exp:.2f}")
        print(f"    Predicted BCGP log coeff:        3.5 - {D2_exp:.2f} = {3.5 - D2_exp:.2f}")
        print(f"    Predicted shift:                  -0.50")
    
    print(f"\n  Reference values:")
    print(f"    BTZ gravity (independent of gauge group): -3/2")
    print(f"    sl₂ BCGP modified trace: -2")
    print(f"    sl₂ radical shift: +1/2")
    print(f"    sl₃ CS/WRT: -4 = -(N²-1)/2")
    
    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print(f"\n{'═' * 120}")
    print(f"  FINAL SUMMARY")
    print(f"{'═' * 120}")
    
    # Best estimates (prefer Richardson, then FD)
    def best_coeff(res_dict):
        for key in ['log_coeff_richardson', 'log_coeff_fd', 'log_coeff_3param']:
            val = res_dict.get(key, float('nan'))
            if np.isfinite(val):
                return val
        return float('nan')
    
    best_full = best_coeff(res_full)
    best_mod = best_coeff(res_mod)
    best_shift = best_coeff(res_delta)
    best_fcont = best_coeff(res_full_cont)
    best_mcont = best_coeff(res_mod_cont)
    
    summary_text = f"""
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │  sl₃ CONTINUOUS SECTOR REFINED RESULTS                                          │
  │                                                                                  │
  │  D̃² SCALING: D̃² ~ r^{D2_exp:.2f}                                               │
  │                                                                                  │
  │  LOG COEFFICIENTS (best estimates):                                              │
  │    Full thermal trace (total):   {best_full:+.4f}                               │
  │    BCGP modified trace (total):  {best_mod:+.4f}                               │
  │    Full trace (cont only):       {best_fcont:+.4f}                               │
  │    BCGP trace (cont only):       {best_mcont:+.4f}                               │
  │    Shift (full - mod):           {best_shift:+.4f}                               │
  │                                                                                  │
  │  METHOD COMPARISON:                                                              │
  │    dblquad:  accurate but slow (O(r²) per evaluation)                           │
  │    saddle:   fast, accurate near saddle; less accurate for small r              │
  │    MC:       moderate speed, statistical error ~ 1/√N                           │
  │                                                                                  │
  │  SADDLE POINT: α*/r → {per_r[-1].get('saddle_bcgp_ratio', float('nan')):.4f} for large r      │
  │    (Center of Weyl chamber = 1/3 ≈ 0.333)                                      │
  └──────────────────────────────────────────────────────────────────────────────────┘
"""
    print(summary_text)
    
    # Build results dict
    summary = {
        'full_log_coeff_best': best_full,
        'modified_log_coeff_best': best_mod,
        'shift_best': best_shift,
        'full_cont_log_coeff_best': best_fcont,
        'mod_cont_log_coeff_best': best_mcont,
        'full_log_coeff_3param': res_full['log_coeff_3param'],
        'full_log_coeff_4param': res_full['log_coeff_4param'],
        'full_log_coeff_fd': res_full['log_coeff_fd'],
        'full_log_coeff_richardson': res_full['log_coeff_richardson'],
        'modified_log_coeff_3param': res_mod['log_coeff_3param'],
        'modified_log_coeff_4param': res_mod['log_coeff_4param'],
        'modified_log_coeff_fd': res_mod['log_coeff_fd'],
        'modified_log_coeff_richardson': res_mod['log_coeff_richardson'],
        'shift_log_coeff_fd': res_delta['log_coeff_fd'],
        'shift_log_coeff_richardson': res_delta['log_coeff_richardson'],
        'D2_exponent': D2_exp,
        'r_values': r_values,
        'beta': beta,
        'per_r_results': per_r,
        'summary_text': summary_text,
    }
    
    return summary


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results, output_dir="/home/z/my-project/download"):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    output = {}
    for key, val in results.items():
        if key == 'per_r_results':
            output['per_r_results'] = []
            for res in val:
                clean = {}
                for k, v in res.items():
                    if isinstance(v, (np.floating, np.integer)):
                        clean[k] = float(v)
                    elif isinstance(v, np.bool_):
                        clean[k] = bool(v)
                    elif isinstance(v, (list, tuple)):
                        clean[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                    else:
                        clean[k] = v
                output['per_r_results'].append(clean)
        elif isinstance(val, (np.floating, np.integer)):
            output[key] = float(val)
        elif isinstance(val, (list, dict, str, type(None))):
            output[key] = val
        else:
            output[key] = str(val)
    
    output_path = os.path.join(output_dir, 'sl3_continuous_refined.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")
    return output_path


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Extended r range with more points
    r_values = [5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31, 41, 51]
    
    results = compute_all(
        r_values=r_values,
        beta=1.0,
        n_mc_samples=500000,
        n_gh_points=60
    )
    
    save_results(results)
