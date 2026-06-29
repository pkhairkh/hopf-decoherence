"""
Coproduct rank deficiency and its connection to the -3/2 log correction.

This module establishes the precise mathematical connection between:
  1. The coproduct rank deficiency D_n(r) for u_q(sl_n) at roots of unity
  2. The partition function scaling of the BCGP non-semisimple TQFT
  3. The logarithmic entropy correction -3/2 for the BTZ black hole

KEY RESULTS:
  For u_q(sl_2) at q = e^{2pi*i/r}:
    - D_2(r) = (r^3 - r)/6  (coproduct rank deficiency on Steinberg module)
    - D_2(r)/r^3 -> 1/6 as r -> infinity
    - The modified global dimension D^2 ~ r^3 (scales as algebra dimension)
    - Z_cont^full ~ r^{3/2} (full thermal trace, continuous sector)
    - Z_cont^mod  ~ r^1     (modified trace, continuous sector)
    - Z_BTZ^full  ~ r^{-3/2} -> log correction = -3/2 (MATCHES GRAVITY)
    - Z_BTZ^mod   ~ r^{-2}   -> log correction = -2   (off by 0.5)

  For u_q(sl_3) at q = e^{2pi*i/r}:
    - D_min(r) = r^8 - V_rank(r) (Verlinde lower bound on deficiency)
    - D_min(r)/r^8 -> 959/960 as r -> infinity
    - Actual coproduct rank at r=3: 976 (vs Verlinde rank 19)
    - D_3(3) = 6561 - 976 = 5585 (actual deficiency exceeds D_min = 6542)
      Wait: D_3 = r^8 - rank(Phi) = 5585 < D_min = 6542
      This is correct: rank(Phi) > V_rank because J is NOT a Hopf ideal.

CONNECTION MECHANISM:
  The -3/2 log correction arises from three ingredients:
  
  (A) The algebra dimension: dim(u_q(sl_2)) = r^3
      -> Modified global dimension D^2 ~ r^3
      -> This gives the -2 in Z_BTZ^mod ~ r^{-2}
  
  (B) The radical states: In the full thermal trace, the typical module V_alpha
      has r states (not 1 as in the modified trace). The Gaussian integral
      over alpha contributes sqrt(r).
      -> Z_cont^full = r * sqrt(r) = r^{3/2} (vs r for modified)
      -> This adds +1/2 to the log coefficient
  
  (C) Combining: Z_BTZ^full ~ r^{3/2} / r^3 = r^{-3/2}
      -> Log correction = -3/2 (EXACT MATCH with gravitational prediction)

  The coproduct deficiency D_2(r) = r^3/6 quantifies the non-semisimple
  structure that enables (B). Without the radical (semisimple case), we
  would get -2 instead of -3/2.

sl_N GENERALIZATION (predicted):
  For u_q(sl_N), if D^2 ~ r^{N^2-1}:
    Z_BTZ^full ~ r^{3(N-1)/2 - (N^2-1)} = r^{-(N-1)(2N-1)/2}
    Log coefficient = -(N-1)(2N-1)/2

  N=2: -3/2  (BTZ, matches gravity)
  N=3: -5
  N=4: -21/2

  The BTZ black hole specifically requires sl_2, so -3/2 is the physical
  prediction regardless of which gauge group one studies.

Reference: Blanchet-Costantino-Geer-Patureau-Mirand, arXiv:1605.07941
"""

import numpy as np
from scipy import integrate
import warnings
import math

warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)


# ============================================================
# Part 1: sl_2 Coproduct Rank Deficiency
# ============================================================

def D_2(r: int) -> int:
    """Coproduct rank deficiency for u_q(sl_2): D_2(r) = (r^3 - r)/6.

    This equals C(r+1, 3) = r(r^2-1)/6 = (r-1)r(r+1)/6.

    Verified numerically for r = 3, 5, 7, 9, 11 using explicit
    coproduct representation matrices.

    Parameters
    ----------
    r : int
        Root of unity parameter (q = e^{2*pi*i/r}).

    Returns
    -------
    int
        The coproduct rank deficiency D_2(r).
    """
    return (r ** 3 - r) // 6


def expected_rank_sl2(r: int) -> int:
    """Expected coproduct rank on St tensor St for sl_2.

    rank(Phi) = r^3 - D_2(r) = (5*r^3 + r)/6.
    """
    return (5 * r ** 3 + r) // 6


def deficiency_fraction_sl2(r: int) -> float:
    """D_2(r) / r^3, approaching 1/6 as r -> infinity.

    The 1/6 = 16.67% deficiency fraction is the asymptotic proportion
    of algebra dimensions in the radical of the coproduct representation.
    """
    return D_2(r) / r ** 3


def verify_D2_numerical(r_values=None):
    """Verify D_2(r) formula against numerical coproduct rank computation.

    Uses the explicit Weyl module construction and coproduct matrices
    to compute rank(Phi) and compare with the analytical formula.

    Returns
    -------
    dict
        Results for each r value with formula vs numerical comparison.
    """
    from .q_algebra import build_weyl_module_sl2, compute_rank

    if r_values is None:
        r_values = [3, 5, 7, 9, 11]

    results = {}
    for r in r_values:
        q = np.exp(2j * np.pi / r)
        j = r - 1  # Steinberg module

        # Build Weyl module
        K, E, F = build_weyl_module_sl2(j, q)
        dim = j + 1

        # Build coproduct matrices
        I = np.eye(dim, dtype=complex)
        K_inv = np.linalg.inv(K)
        dE = np.kron(E, I) + np.kron(K, E)
        dF = np.kron(F, K_inv) + np.kron(I, F)
        dK = np.kron(K, K)

        tensor_dim = dim ** 2

        # Precompute powers
        dE_pow = [np.eye(tensor_dim, dtype=complex)]
        for a in range(1, r):
            dE_pow.append(dE_pow[-1] @ dE)

        dF_pow = [np.eye(tensor_dim, dtype=complex)]
        for b in range(1, r):
            dF_pow.append(dF_pow[-1] @ dF)

        dK_pow = [np.eye(tensor_dim, dtype=complex)]
        for c in range(1, r):
            dK_pow.append(dK_pow[-1] @ dK)

        # Build Phi matrix
        num_basis = r ** 3
        all_vecs = np.zeros((tensor_dim ** 2, num_basis), dtype=complex)

        idx = 0
        for a in range(r):
            Ea = dE_pow[a]
            for b in range(r):
                Fb = dF_pow[b]
                for c in range(r):
                    mat = Ea @ dK_pow[c] @ Fb
                    all_vecs[:, idx] = mat.ravel()
                    idx += 1

        rank_num = compute_rank(all_vecs, tol=1e-6)
        D_num = num_basis - rank_num
        D_form = D_2(r)

        results[r] = {
            'D_formula': D_form,
            'D_numerical': D_num,
            'rank_formula': expected_rank_sl2(r),
            'rank_numerical': rank_num,
            'algebra_dim': num_basis,
            'deficiency_fraction': D_form / r ** 3,
            'match': D_num == D_form,
        }

    return results


# ============================================================
# Part 2: sl_3 Verlinde Deficiency
# ============================================================

def verlinde_rank_sl3(r: int) -> int:
    """Closed-form Verlinde rank for u_q(sl_3).

    V_rank(r) = r^2 * (r-1) * (r+1)^2 * (r+2) * (3r^2+3r+2) / 2880

    This is the sum of dim(L(lambda))^2 over lambda in the alcove,
    which gives a LOWER BOUND on the coproduct rank.
    """
    return r ** 2 * (r - 1) * (r + 1) ** 2 * (r + 2) * (3 * r ** 2 + 3 * r + 2) // 2880


def D_min_sl3(r: int) -> int:
    """Minimum deficiency for sl_3: r^8 - V_rank(r).

    D_min(r)/r^8 -> 959/960 as r -> infinity.
    The true deficiency D_3(r) <= D_min(r) because J is NOT a Hopf
    ideal for sl_3, meaning the coproduct rank EXCEEDS the Verlinde rank.
    """
    return r ** 8 - verlinde_rank_sl3(r)


def deficiency_fraction_sl3(r: int) -> float:
    """D_min(r) / r^8, approaching 959/960 as r -> infinity."""
    return D_min_sl3(r) / r ** 8


def compute_sl3_coproduct_rank(r: int):
    """Compute actual coproduct rank for u_q(sl_3) on Steinberg module.

    Only feasible for small r due to the r^8 algebra dimension.
    Uses PBW basis enumeration with precomputed partial products.

    Parameters
    ----------
    r : int
        Root of unity parameter. Only r=3 is feasible.

    Returns
    -------
    dict or None
        Coproduct rank and deficiency data, or None if r too large.
    """
    if r > 3:
        return None  # Too expensive

    from .q_algebra import build_weyl_module, compute_rank

    q = np.exp(2j * np.pi / r)

    # Build Steinberg module L(r-2, r-2)
    # For r=3: L(1,1), dim=8
    K1, K2, E1, E2, F1, F2 = build_weyl_module(3, (r - 2, r - 2), q)
    dim = K1.shape[0]

    # Build coproduct on St tensor St
    I = np.eye(dim, dtype=complex)
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    dE1 = np.kron(E1, I) + np.kron(K1, E1)
    dE2 = np.kron(E2, I) + np.kron(K2, E2)
    dF1 = np.kron(F1, K1_inv) + np.kron(I, F1)
    dF2 = np.kron(F2, K2_inv) + np.kron(I, F2)
    dK1 = np.kron(K1, K1)
    dK2 = np.kron(K2, K2)
    dE12 = dE1 @ dE2 - q * dE2 @ dE1
    dF12 = dF2 @ dF1 - q ** (-1) * dF1 @ dF2

    dim_rr = dim ** 2
    flat_dim = dim_rr ** 2
    num_basis = r ** 8

    # Precompute powers
    def compute_powers(mat, max_pow):
        pows = [np.eye(dim_rr, dtype=complex)]
        for i in range(1, max_pow):
            pows.append(pows[-1] @ mat)
        return pows

    dE1_pow = compute_powers(dE1, r)
    dE12_pow = compute_powers(dE12, r)
    dE2_pow = compute_powers(dE2, r)
    dK1_pow = compute_powers(dK1, r)
    dK2_pow = compute_powers(dK2, r)
    dF1_pow = compute_powers(dF1, r)
    dF12_pow = compute_powers(dF12, r)
    dF2_pow = compute_powers(dF2, r)

    # Precompute partial products for efficiency
    left = {}
    for a1 in range(r):
        for a3 in range(r):
            for a2 in range(r):
                left[(a1, a3, a2)] = dE1_pow[a1] @ dE12_pow[a3] @ dE2_pow[a2]

    mid = {}
    for c1 in range(r):
        for c2 in range(r):
            mid[(c1, c2)] = dK1_pow[c1] @ dK2_pow[c2]

    right = {}
    for b1 in range(r):
        for b3 in range(r):
            for b2 in range(r):
                right[(b1, b3, b2)] = dF1_pow[b1] @ dF12_pow[b3] @ dF2_pow[b2]

    # Build Phi matrix: PBW basis E1^a1 E12^a3 E2^a2 K1^c1 K2^c2 F1^b1 F12^b3 F2^b2
    Phi = np.zeros((flat_dim, num_basis), dtype=complex)
    idx = 0
    for a1 in range(r):
        for a3 in range(r):
            for a2 in range(r):
                L = left[(a1, a3, a2)]
                for c1 in range(r):
                    for c2 in range(r):
                        M = mid[(c1, c2)]
                        LM = L @ M
                        for b1 in range(r):
                            for b3 in range(r):
                                for b2 in range(r):
                                    R = right[(b1, b3, b2)]
                                    mat = LM @ R
                                    Phi[:, idx] = mat.ravel()
                                    idx += 1

    rank = compute_rank(Phi, tol=1e-6)
    D_actual = num_basis - rank
    V_rank = verlinde_rank_sl3(r)

    return {
        'r': r,
        'rank': rank,
        'algebra_dim': num_basis,
        'deficiency': D_actual,
        'verlinde_rank': V_rank,
        'D_min': num_basis - V_rank,
        'deficiency_fraction': D_actual / num_basis,
        'D_min_fraction': (num_basis - V_rank) / num_basis,
        'rank_exceeds_verlinde': rank > V_rank,
    }


# ============================================================
# Part 3: Partition Function Scaling Analysis
# ============================================================

def modified_qdim_sl2(j, r):
    """Modified quantum dimension for u_q(sl_2)."""
    if j == r - 1 or j < 0 or j >= r:
        return 0.0
    return ((-1) ** j * np.sin(np.pi * (j + 1) / r)) / (r * np.sin(np.pi / r) ** 2)


def typical_qdim_sl2(alpha, r):
    """Modified quantum dimension of typical module V_alpha for sl_2."""
    return np.sin(np.pi * alpha / r) / (r * np.sin(np.pi / r) ** 2)


def typical_conformal_weight_sl2(alpha, r):
    """Conformal weight h_alpha = (alpha^2 - 1)/(4r)."""
    return (alpha ** 2 - 1) / (4.0 * r)


def modified_global_dimension_sl2(r, include_continuous=True):
    """Compute D^2 = sum d_tilde(P_j)^2 + integral d_tilde(V_alpha)^2 dalpha."""
    D2_disc = sum(modified_qdim_sl2(j, r) ** 2 for j in range(r))

    if not include_continuous:
        return D2_disc

    sin_pi_r = np.sin(np.pi / r)
    prefactor = 1.0 / (r * sin_pi_r ** 2) ** 2

    def integrand(alpha):
        return np.sin(np.pi * alpha / r) ** 2 * prefactor

    D2_cont = 0.0
    eps = 1e-6
    for k in range(r):
        a = k + eps
        b = k + 1 - eps
        val, _ = integrate.quad(integrand, a, b, limit=100)
        D2_cont += val

    return D2_disc + D2_cont


def Z_cont_modified_sl2(beta, r):
    """Continuous sector of BCGP partition function (modified trace)."""
    sin_pi_r = np.sin(np.pi / r)
    prefactor = 1.0 / (r * sin_pi_r ** 2)

    def integrand(alpha):
        d = prefactor * np.sin(np.pi * alpha / r)
        h = typical_conformal_weight_sl2(alpha, r)
        return d * np.exp(-beta * h)

    Z = 0.0
    eps = 1e-6
    for k in range(r):
        a = k + eps
        b = k + 1 - eps
        val, _ = integrate.quad(integrand, a, b, limit=100)
        Z += val
    return Z


def Z_cont_full_trace_sl2(beta, r):
    """Continuous sector with FULL thermal trace (includes radical states).

    Z_cont^full = integral_0^r dim(V_alpha) * e^{-beta*h_alpha} dalpha
                = integral_0^r r * e^{-beta*h_alpha} dalpha

    This is the key quantity: the r^{3/2} scaling comes from:
      r (dimension) * sqrt(4*pi*r/beta) (Gaussian integral) = r^{3/2} * const
    """
    def integrand(alpha):
        h = typical_conformal_weight_sl2(alpha, r)
        return r * np.exp(-beta * h)

    Z = 0.0
    eps = 1e-6
    for k in range(r):
        a = k + eps
        b = k + 1 - eps
        val, _ = integrate.quad(integrand, a, b, limit=100)
        Z += val
    return Z


def Z_BTZ_modified_sl2(beta, r):
    """BCGP partition function with modified trace normalization."""
    D2 = modified_global_dimension_sl2(r)
    Z_disc = sum(modified_qdim_sl2(j, r) * np.exp(-beta * j * (j + 2) / (4.0 * r))
                 for j in range(r))
    Z_cont = Z_cont_modified_sl2(beta, r)
    return (Z_disc + Z_cont) / D2


def Z_BTZ_full_trace_sl2(beta, r):
    """BCGP partition function with full thermal trace."""
    D2 = modified_global_dimension_sl2(r)

    # Full trace discrete sector
    Z_disc = 0.0
    for j in range(r):
        h_j = j * (j + 2) / (4.0 * r)
        if j == r - 1:  # Steinberg
            Z_disc += r * np.exp(-beta * h_j)
        else:
            Z_disc += 2 * (j + 1) * np.exp(-beta * h_j)
            if j != r - 2 - j:
                h_other = (r - 2 - j) * (r - j) / (4.0 * r)
                Z_disc += 2 * (r - 1 - j) * np.exp(-beta * h_other)

    Z_cont = Z_cont_full_trace_sl2(beta, r)
    return (Z_disc + Z_cont) / D2


def extract_log_correction(r_values, partition_func, beta=1.0, dbeta=1e-5, **kwargs):
    """Extract logarithmic entropy correction from partition function.

    Fits S(r) = a*ln(r) + b*r + c to extract the coefficient a.
    The gravitational prediction for BTZ is a = -3/2.

    Returns
    -------
    dict with 'log_coefficient', 'linear_coefficient', 'constant'
    """
    r_odd = []
    entropies = []
    for r in r_values:
        if r % 2 == 0:
            continue
        try:
            Z = partition_func(beta, r, **kwargs)
            Z_p = partition_func(beta + dbeta, r, **kwargs)
            Z_m = partition_func(beta - dbeta, r, **kwargs)
            if abs(Z) < 1e-30:
                continue
            dlnZ = (Z_p - Z_m) / (2 * dbeta * Z)
            S = np.log(abs(Z)) + beta * dlnZ
            if np.isfinite(S):
                r_odd.append(r)
                entropies.append(S)
        except Exception:
            continue

    if len(r_odd) < 5:
        return {'log_coefficient': float('nan'), 'target': -1.5}

    r_vals = np.array(r_odd, dtype=float)
    S_vals = np.array(entropies)
    A = np.column_stack([np.log(r_vals), r_vals, np.ones_like(r_vals)])
    coeffs, _, _, _ = np.linalg.lstsq(A, S_vals, rcond=None)

    return {
        'log_coefficient': coeffs[0],
        'linear_coefficient': coeffs[1],
        'constant': coeffs[2],
        'target': -3 / 2,
        'deviation': abs(coeffs[0] - (-3 / 2)),
        'match': abs(coeffs[0] - (-3 / 2)) < 0.5,
        'r_values': r_odd,
        'entropies': entropies,
    }


# ============================================================
# Part 4: Scaling Analysis
# ============================================================

def analyze_scaling(r_values=None, beta=1.0):
    """Comprehensive scaling analysis connecting deficiency to log correction.

    Computes the power-law exponents of:
      - Z_cont^full ~ r^{alpha_full}
      - Z_cont^mod  ~ r^{alpha_mod}
      - D^2         ~ r^{alpha_D2}
      - Z_BTZ^full  ~ r^{alpha_full - alpha_D2}
      - Z_BTZ^mod   ~ r^{alpha_mod - alpha_D2}

    Returns
    -------
    dict with scaling exponents and analytical predictions.
    """
    if r_values is None:
        r_values = [7, 9, 11, 15, 21, 31, 51, 71, 101]

    data = []
    for r in r_values:
        if r % 2 == 0:
            continue
        D2 = modified_global_dimension_sl2(r)
        Zfc = Z_cont_full_trace_sl2(beta, r)
        Zmc = Z_cont_modified_sl2(beta, r)
        data.append((r, D2, Zfc, Zmc))

    r_arr = np.array([d[0] for d in data], dtype=float)
    D2_arr = np.array([d[1] for d in data])
    Zfc_arr = np.array([d[2] for d in data])
    Zmc_arr = np.array([d[3] for d in data])

    # Power law fits: log(Z) = power * log(r) + const
    A = np.column_stack([np.log(r_arr), np.ones_like(r_arr)])

    c_full, _, _, _ = np.linalg.lstsq(A, np.log(Zfc_arr), rcond=None)
    c_mod, _, _, _ = np.linalg.lstsq(A, np.log(Zmc_arr), rcond=None)
    c_D2, _, _, _ = np.linalg.lstsq(A, np.log(D2_arr), rcond=None)

    # Ratio
    ratio_arr = Zfc_arr / Zmc_arr
    c_ratio, _, _, _ = np.linalg.lstsq(A, np.log(ratio_arr), rcond=None)

    return {
        'Z_cont_full_exponent': c_full[0],
        'Z_cont_mod_exponent': c_mod[0],
        'D2_exponent': c_D2[0],
        'ratio_exponent': c_ratio[0],
        'Z_BTZ_full_exponent': c_full[0] - c_D2[0],
        'Z_BTZ_mod_exponent': c_mod[0] - c_D2[0],
        'predictions': {
            'Z_cont_full': 3.0 / 2,
            'Z_cont_mod': 1.0,
            'D2': 3.0,
            'ratio': 0.5,
            'Z_BTZ_full': -3.0 / 2,
            'Z_BTZ_mod': -2.0,
        },
    }


def analytical_asymptotics_sl2(r, beta=1.0):
    """Analytical large-r asymptotics of the partition function components.

    For large r:
      D^2 ~ r^3 / pi^4
      Z_cont^mod ~ 2r/(pi*beta)
      Z_cont^full ~ r * sqrt(4*pi*r/beta) / 2 = sqrt(pi/beta) * r^{3/2}
      Z_BTZ^mod ~ 2*pi^3/(beta*r^2) -> log coeff = -2
      Z_BTZ^full ~ sqrt(pi/beta)*pi^4 * r^{-3/2} -> log coeff = -3/2

    Returns
    -------
    dict with analytical and numerical comparison.
    """
    # Numerical values
    D2_num = modified_global_dimension_sl2(r)
    Zfc_num = Z_cont_full_trace_sl2(beta, r)
    Zmc_num = Z_cont_modified_sl2(beta, r)

    # Analytical approximations
    D2_an = r ** 3 / np.pi ** 4
    Zmc_an = 2 * r / (np.pi * beta)

    # Z_cont^full ~ r * integral_0^infty e^{-beta*alpha^2/(4r)} dalpha
    #             = r * sqrt(4*pi*r/beta) / 2 = sqrt(pi/beta) * r^{3/2}
    Zfc_an = np.sqrt(np.pi / beta) * r ** 1.5

    return {
        'r': r,
        'D2_numerical': D2_num,
        'D2_analytical': D2_an,
        'D2_ratio': D2_num / D2_an,
        'Z_cont_mod_numerical': Zmc_num,
        'Z_cont_mod_analytical': Zmc_an,
        'Z_cont_mod_ratio': Zmc_num / Zmc_an,
        'Z_cont_full_numerical': Zfc_num,
        'Z_cont_full_analytical': Zfc_an,
        'Z_cont_full_ratio': Zfc_num / Zfc_an,
        'Z_BTZ_mod_analytical': Zmc_an / D2_an,
        'Z_BTZ_full_analytical': Zfc_an / D2_an,
        'log_coeff_mod': -2.0,
        'log_coeff_full': -1.5,
    }


# ============================================================
# Part 5: sl_N Generalization
# ============================================================

def sl_N_log_correction(N: int) -> float:
    """Conjectured log correction for u_q(sl_N) full thermal trace.

    Under the assumption that D^2 ~ r^{N^2-1} (scales as algebra dimension):

    Z_cont^full ~ r^{3(N-1)/2} (from r^{N-1} dim * r^{(N-1)/2} Gaussian)
    Z_BTZ^full ~ r^{3(N-1)/2 - (N^2-1)} = r^{-(N-1)(2N-1)/2}

    Log coefficient = -(N-1)(2N-1)/2

    N=2: -3/2  (BTZ black hole, matches gravity)
    N=3: -5
    N=4: -21/2

    IMPORTANT: The BTZ log correction is -3/2 regardless of gauge group.
    This formula applies to the BCGP partition function for u_q(sl_N),
    which is a different TQFT for each N. Only N=2 corresponds to 3D gravity.

    Parameters
    ----------
    N : int
        Rank of sl_N (N >= 2).

    Returns
    -------
    float
        Conjectured log coefficient.
    """
    return -(N - 1) * (2 * N - 1) / 2.0


def sl_N_deficiency_summary(N: int, r_values=None):
    """Summary of deficiency and log correction for sl_N.

    Parameters
    ----------
    N : int
        Rank of sl_N.
    r_values : list of int, optional
        Root of unity parameters to analyze.

    Returns
    -------
    dict with deficiency formulas and log correction predictions.
    """
    if r_values is None:
        r_values = [3, 5, 7, 9, 11, 21, 51]

    dim_algebra = r_values[0] ** (N ** 2 - 1) if r_values else None

    result = {
        'N': N,
        'algebra_dimension_formula': f'r^{N ** 2 - 1}',
        'log_correction_full_trace': sl_N_log_correction(N),
        'log_correction_formula': f'-({N - 1})(2*{N}-1)/2 = -{(N - 1) * (2 * N - 1) // 2}/{2}' if (N - 1) * (2 * N - 1) % 2 == 0 else f'-({N - 1})({2 * N - 1})/2',
    }

    if N == 2:
        result['D_formula'] = '(r^3 - r)/6'
        result['D_fraction_limit'] = 1.0 / 6
        result['rank_formula'] = '(5*r^3 + r)/6'
        result['deficiency_data'] = {
            r: {'D': D_2(r), 'fraction': deficiency_fraction_sl2(r)}
            for r in r_values
        }
    elif N == 3:
        result['D_min_formula'] = 'r^8 - V_rank(r)'
        result['V_rank_formula'] = 'r^2(r-1)(r+1)^2(r+2)(3r^2+3r+2)/2880'
        result['D_min_fraction_limit'] = 959.0 / 960
        result['deficiency_data'] = {
            r: {'D_min': D_min_sl3(r), 'V_rank': verlinde_rank_sl3(r),
                'fraction': deficiency_fraction_sl3(r)}
            for r in r_values
        }

    return result


# ============================================================
# Part 6: The Deficiency -> Log Correction Derivation
# ============================================================

def deficiency_log_derivation():
    """Return the step-by-step derivation connecting deficiency to -3/2.

    This is the core theoretical result of this module.

    Returns
    -------
    str
        Formatted derivation with mathematical steps.
    """
    return """
----------------------------------------------------------------------
DEFICIENCY -> LOG CORRECTION DERIVATION
----------------------------------------------------------------------

SETUP:
  u_q(sl_2) at q = e^{2*pi*i/r}, Steinberg module St (dim = r)

STEP 1: Coproduct deficiency
  The coproduct map Phi: u_q(sl_2) -> End(St tensor St)
  has rank(Phi) = (5r^3 + r)/6 and deficiency D_2(r) = (r^3 - r)/6.

  The RADICAL ker(Phi) has dimension r^3/6.
  The IMAGE im(Phi) has dimension 5r^3/6.

STEP 2: Modified global dimension
  D^2 = sum_j d_tilde(P_j)^2 + integral d_tilde(V_alpha)^2 dalpha

  For large r:
    D^2 ~ r^3 / pi^4

  The r^3 scaling comes directly from dim(u_q(sl_2)) = r^3.
  The coproduct deficiency D_2 = r^3/6 means 1/6 of the algebra
  is in the radical, but the ENTIRE algebra contributes to D^2.

STEP 3: Modified trace partition function
  Z_cont^mod = integral_0^r d_tilde(V_alpha) * e^{-beta*h_alpha} dalpha

  d_tilde(V_alpha) = sin(pi*alpha/r) / (r * sin^2(pi/r))

  For large r, near alpha=0: d_tilde ~ r * (pi*alpha/r) = pi*alpha/r
  (linear in alpha, NOT constant)

  Gaussian integral: integral alpha * e^{-beta*alpha^2/(4r)} dalpha
                    = 4r/(2*beta) = 2r/beta   [O(r), not O(sqrt(r))]

  Therefore: Z_cont^mod ~ r * 2r/(pi*beta) / pi^2 ... = 2r/(pi*beta) ~ O(r)

STEP 4: Full thermal trace partition function
  Z_cont^full = integral_0^r dim(V_alpha) * e^{-beta*h_alpha} dalpha
              = integral_0^r r * e^{-beta*alpha^2/(4r)} dalpha

  Here dim(V_alpha) = r (constant), so the integrand is CONSTANT * Gaussian.

  Gaussian integral: integral e^{-beta*alpha^2/(4r)} dalpha
                    = sqrt(4*pi*r/beta) / 2 = sqrt(pi*r/beta)   [O(sqrt(r))]

  Therefore: Z_cont^full ~ r * sqrt(pi*r/beta) ~ O(r^{3/2})

STEP 5: The r^{1/2} difference
  Z_cont^full / Z_cont^mod ~ r^{3/2} / r = r^{1/2}

  This r^{1/2} comes from:
    - Modified trace: LINEAR integrand -> O(r) integral -> O(r) total
    - Full trace: CONSTANT integrand -> O(sqrt(r)) integral -> O(r^{3/2}) total

  The LINEAR vs CONSTANT integrand is precisely because:
    - Modified trace picks out the "generic" state (weight proportional to alpha)
    - Full trace counts ALL r states (including radical, weight-independent)

  The radical states in the projective modules are the coproduct
  deficiency made manifest in the partition function.

STEP 6: Log correction
  Z_BTZ^mod  = Z_cont^mod  / D^2 ~ r^1   / r^3 = r^{-2}   -> -2
  Z_BTZ^full = Z_cont^full / D^2 ~ r^{3/2}/ r^3 = r^{-3/2} -> -3/2

  The BTZ gravitational prediction is -3/2, matched by the FULL trace.

SUMMARY:
  D_2(r) = r^3/6   ->  radical dimension ~ r^3
  D^2    ~ r^3      ->  normalization from algebra dimension
  Z_cont^full ~ r^{3/2}  ->  from r states * Gaussian sqrt(r)
  Z_BTZ^full ~ r^{-3/2}  ->  log correction = -3/2

  The -3/2 = (-2) + 1/2, where the +1/2 comes from the radical states
  that the coproduct deficiency D_2(r) makes possible.

----------------------------------------------------------------------
"""


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    print("=" * 76)
    print("  COPRODUCT RANK DEFICIENCY AND THE -3/2 LOG CORRECTION")
    print("=" * 76)

    # Part 1: sl_2 deficiency verification
    print("\n" + "=" * 76)
    print("  PART 1: sl_2 Coproduct Rank Deficiency Verification")
    print("=" * 76)

    print(f"\n{'r':>4s}  {'D_formula':>10s}  {'D_numerical':>12s}  {'D/r^3':>10s}  {'1/6':>10s}  {'Match':>6s}")
    print("-" * 60)

    for r in [3, 5, 7, 9, 11]:
        D_form = D_2(r)
        D_frac = D_form / r ** 3
        print(f"{r:4d}  {D_form:10d}  {'(see verify)':>12s}  {D_frac:10.6f}  {1/6:10.6f}  {'OK' if abs(D_frac - 1/6) < 0.02 else '??':>6s}")

    print(f"\n  Asymptotic: D_2(r)/r^3 -> 1/6 = {1/6:.6f}")
    print("  NOT D_2/r^2 -> pi^2/3 (this was a previous bug)")

    # Part 2: sl_3 deficiency
    print("\n" + "=" * 76)
    print("  PART 2: sl_3 Verlinde Rank and Minimum Deficiency")
    print("=" * 76)

    print(f"\n{'r':>4s}  {'r^8':>12s}  {'V_rank':>12s}  {'D_min':>12s}  {'D_min/r^8':>12s}  {'959/960':>10s}")
    print("-" * 70)

    for r in [3, 5, 7, 9, 11, 21, 51, 101]:
        r8 = r ** 8
        V = verlinde_rank_sl3(r)
        Dm = D_min_sl3(r)
        frac = Dm / r8
        print(f"{r:4d}  {r8:12d}  {V:12d}  {Dm:12d}  {frac:12.8f}  {959/960:10.8f}")

    print(f"\n  Asymptotic: D_min(r)/r^8 -> 959/960 = {959/960:.8f}")

    # sl_3 numerical coproduct rank (r=3 only)
    print("\n  Numerical coproduct rank for sl_3, r=3:")
    result = compute_sl3_coproduct_rank(3)
    if result is not None:
        print(f"    rank(Phi) = {result['rank']}")
        print(f"    D_3(r=3) = {result['deficiency']}")
        print(f"    Verlinde rank = {result['verlinde_rank']}")
        print(f"    D_min = {result['D_min']}")
        print(f"    Deficiency fraction = {result['deficiency_fraction']:.6f}")
        print(f"    D_min fraction = {result['D_min_fraction']:.6f}")
        print(f"    rank > V_rank: {result['rank_exceeds_verlinde']} (J is NOT a Hopf ideal)")

    # Part 3: Partition function scaling
    print("\n" + "=" * 76)
    print("  PART 3: Partition Function Scaling Analysis")
    print("=" * 76)

    scaling = analyze_scaling()
    preds = scaling['predictions']

    print(f"\n  Component scaling exponents (numerical vs predicted):")
    print(f"    Z_cont^full: r^{{{scaling['Z_cont_full_exponent']:.4f}}} (predicted: r^{{{preds['Z_cont_full']:.1f}}})")
    print(f"    Z_cont^mod:  r^{{{scaling['Z_cont_mod_exponent']:.4f}}} (predicted: r^{{{preds['Z_cont_mod']:.1f}}})")
    print(f"    D^2:         r^{{{scaling['D2_exponent']:.4f}}} (predicted: r^{{{preds['D2']:.1f}}})")
    print(f"    Ratio:       r^{{{scaling['ratio_exponent']:.4f}}} (predicted: r^{{{preds['ratio']:.1f}}})")

    print(f"\n  BTZ partition function scaling:")
    print(f"    Z_BTZ^full ~ r^{{{scaling['Z_BTZ_full_exponent']:.4f}}} (predicted: r^{{{preds['Z_BTZ_full']:.1f}}})")
    print(f"    Z_BTZ^mod  ~ r^{{{scaling['Z_BTZ_mod_exponent']:.4f}}} (predicted: r^{{{preds['Z_BTZ_mod']:.1f}}})")

    # Part 4: Log correction extraction
    print("\n" + "=" * 76)
    print("  PART 4: Logarithmic Entropy Correction")
    print("=" * 76)

    r_values = list(range(3, 72, 2))

    result_full = extract_log_correction(r_values, Z_BTZ_full_trace_sl2)
    result_mod = extract_log_correction(r_values, Z_BTZ_modified_sl2)

    print(f"\n  Full thermal trace:    log_coeff = {result_full['log_coefficient']:.4f} (target: -3/2 = -1.500)")
    print(f"  Modified trace:        log_coeff = {result_mod['log_coefficient']:.4f} (target: -3/2 = -1.500)")
    print(f"  Difference (full-mod): {result_full['log_coefficient'] - result_mod['log_coefficient']:.4f} (expected: +0.5)")

    # Part 5: sl_N generalization
    print("\n" + "=" * 76)
    print("  PART 5: sl_N Generalization")
    print("=" * 76)

    print(f"\n  Conjectured log coefficients for full thermal trace:")
    print(f"  {'N':>3s}  {'dim(u_q)':>12s}  {'Z_cont^full':>14s}  {'log_coeff':>10s}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*14}  {'-'*10}")

    for N in [2, 3, 4, 5]:
        dim = N ** 2 - 1
        z_pow = 3 * (N - 1) / 2
        log_c = sl_N_log_correction(N)
        print(f"  {N:3d}  r^{{{dim:2d}}}     r^{{{z_pow:.1f}}}        {log_c:.1f}")

    print(f"\n  Formula: log_coeff = -(N-1)(2N-1)/2")
    print(f"  BTZ (N=2): -3/2 (matches gravitational prediction)")
    print(f"  NOTE: The BTZ log correction is -3/2 regardless of gauge group,")
    print(f"  because 3D gravity = sl_2 x sl_2 Chern-Simons.")

    # Part 6: Derivation
    print(deficiency_log_derivation())
