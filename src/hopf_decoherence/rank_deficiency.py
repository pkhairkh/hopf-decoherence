"""
Coproduct rank deficiency formulas for u_q(sl_n).

Core results:
  D_2(ell) = (ell^3 - ell) / 6 = C(ell+1, 3)   [sl_2]
  D(ell, k1, k2) = C(k1+k2-ell+3, 3) if k1+k2 >= ell-1, else 0
  rank(Phi) = (5*ell^3 + ell) / 6  [sl_2 Steinberg]

Verlinde rank for sl_3:
  V_rank(ell) = ell^2(ell-1)(ell+1)^2(ell+2)(3ell^2+3ell+2) / 2880
  D_min(ell) = ell^8 - V_rank(ell)
  D_min/ell^8 -> 959/960 as ell -> infinity
"""

import math


# ============================================================
# sl_2 Rank Deficiency
# ============================================================

def D_ell(ell: int) -> int:
    """Coproduct rank deficiency for u_q(sl_2): D(ell) = (ell^3 - ell) / 6.

    Equivalent forms:
      D(ell) = C(ell+1, 3)         (binomial coefficient)
      D(ell) = ell(ell^2 - 1) / 6  (factored)

    Verified numerically for ell = 3, 5, 7.
    """
    return (ell ** 3 - ell) // 6


def expected_rank(ell: int) -> int:
    """Expected rank(Phi) on St tensor St for sl_2: (5*ell^3 + ell) / 6."""
    return (5 * ell ** 3 + ell) // 6


def D_k1k2(ell: int, k1: int, k2: int) -> int:
    """Coproduct rank deficiency for specific representation pair.

    D(ell, k1, k2) = C(k1+k2-ell+3, 3) if k1+k2 >= ell-1, else 0.

    This equals the sum of (j - ell + 1)^2 over non-simple CG summands
    in the decomposition of L(k1) tensor L(k2).
    """
    if k1 + k2 < ell - 1:
        return 0
    n = k1 + k2 - ell + 3
    if n < 3:
        return 0
    return n * (n - 1) * (n - 2) // 6  # C(n, 3)


def deficiency_fraction_sl2(ell: int) -> float:
    """D(ell) / ell^3, the fraction of algebra dimensions lost.

    Approaches 1/6 as ell -> infinity.
    """
    return D_ell(ell) / ell ** 3


def entropy_deficit_sl2(ell: int) -> float:
    """Entropy deficit Delta_H = -ln(rank(Phi) / ell^3).

    Approaches ln(6/5) ~ 0.1823 nats as ell -> infinity.
    This is the O(1) correction from the coproduct deficiency.
    """
    return -math.log(expected_rank(ell) / ell ** 3)


# ============================================================
# sl_3 Verlinde Rank and Deficiency
# ============================================================

def verlinde_rank_sl3(ell: int) -> int:
    """Closed-form Verlinde rank for u_q(sl_3).

    V_rank(ell) = ell^2(ell-1)(ell+1)^2(ell+2)(3ell^2+3ell+2) / 2880

    This is the sum of dim(L(lambda))^2 over lambda in the alcove.
    It is a lower bound on the actual coproduct rank (which exceeds it
    because J is NOT a Hopf ideal for sl_3).
    """
    return ell ** 2 * (ell - 1) * (ell + 1) ** 2 * (ell + 2) * (3 * ell ** 2 + 3 * ell + 2) // 2880


def verlinde_rank_sl3_computed(ell: int) -> int:
    """Compute Verlinde rank by direct summation over the alcove.

    Verifies the closed-form formula.
    """
    total = 0
    for a in range(ell - 1):
        for b in range(ell - 1 - a):
            d = (a + 1) * (b + 1) * (a + b + 2) // 2
            total += d * d
    return total


def D_min_sl3(ell: int) -> int:
    """Minimum deficiency for sl_3: ell^8 - V_rank(ell).

    This is a LOWER BOUND on the actual coproduct deficiency.
    The true deficiency D_3(ell) >= D_min(ell) because the coproduct
    rank exceeds the Verlinde rank (J is not a Hopf ideal).

    D_min(ell) / ell^8 -> 959/960 ~ 99.896% as ell -> infinity.
    """
    return ell ** 8 - verlinde_rank_sl3(ell)


def deficiency_fraction_sl3(ell: int) -> float:
    """D_min(ell) / ell^8, the minimum deficiency fraction for sl_3."""
    return D_min_sl3(ell) / ell ** 8


# ============================================================
# Cross-Representation Deficiency Cascade
# ============================================================

def deficiency_cascade(ell: int, k: int) -> int:
    """Deficiency cascade: D(ell, k) = D_2(2k - ell + 2).

    For k >= (ell-1)/2, the diagonal deficiency D(ell, k, k) equals
    D_2(ell) evaluated at the effective dimension 2k - ell + 2.
    This reveals a self-similar structure in the deficiency.
    """
    V_def = 2 * k - ell + 2
    if V_def < 3:
        return 0
    return (V_def ** 3 - V_def) // 6


# ============================================================
# Excess Sum Verification
# ============================================================

def excess_sum(ell: int, k1: int, k2: int) -> int:
    """Sum of (j - ell + 1)^2 over non-simple CG summands.

    In the CG decomposition L(k1) tensor L(k2) = direct_sum L(j),
    the non-simple summands have j >= ell. The excess sum is:
      sum (j - ell + 1)^2 over j >= ell with appropriate parity.

    This equals D(ell, k1, k2).
    """
    j_min = abs(k1 - k2)
    j_max = k1 + k2
    # CG summands: j = j_min, j_min+2, ..., j_max
    total = 0
    for j in range(j_min, j_max + 1, 2):
        if j >= ell:
            total += (j - ell + 1) ** 2
    return total
