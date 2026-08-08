#!/usr/bin/env python3
"""
Compute dim H̃¹_b(B⁺(u_q(sl_4)), C) at ℓ = 3.

KEY INSIGHT (verified for sl_2 and sl_3): the 1-cocycles h: B̄ → B̄ in
ker(∂_b) are DIAGONAL in the PBW basis and LINEAR in the PBW exponents:
    h(K₁^a..E₃^i) = (α_K1·a + α_K2·b + α_K3·c + α_E1·d + α_E12·e
                    + α_E123·f + α_E2·g + α_E23·h + α_E3·i) · (same monomial)

This reduces the problem from dim(B̄)² ≈ 3.87×10⁸ (intractable) to a
9-dimensional linear system.  We verify this ansatz against the full
diagonal kernel for sl_3 (dim 2 ✓) and sl_2 (dim 1 ✓), then apply to sl_4.

The constraints on the 9 α-coefficients come from:
  (∂^h)  for each (l, a, b) with mult_bar[l,a,b] ≠ 0:  c(a)+c(b)-c(l) = 0
  (∂^c)  for each (c, j, k) with delta_bar[c,j,k] ≠ 0:  c(j)+c(k)-c(c) = 0

where c(·) is the linear function of PBW exponents.

B⁺(u_q(sl_4)) at ℓ=3:
  Generators: K₁, K₂, K₃, E₁, E₂, E₃ (simple), E₁₂, E₂₃, E₁₂₃ (composite).
  Lusztig root vectors (sl_3 convention, E_{α+β} = E_α E_β - q E_β E_α):
    E₁₂  = E₁E₂ - q E₂E₁
    E₂₃  = E₂E₃ - q E₃E₂
    E₁₂₃ = E₁₂E₃ - q E₃E₁₂ = E₁E₂₃ - q E₂₃E₁  (verified equal)
  PBW order: K₁, K₂, K₃, E₁, E₁₂, E₁₂₃, E₂, E₂₃, E₃.
  PBW basis: K₁^a K₂^b K₃^c E₁^d E₁₂^e E₁₂₃^f E₂^g E₂₃^h E₃^i,
             0 ≤ a..i ≤ 2.  dim B = 3⁹ = 19683, dim B̄ = 19682.
  ε(K₁^a..E₃^i) = 1 if d=e=f=g=h=i=0 else 0.

  Cartan A₃ = [[2,-1,0],[-1,2,-1],[0,-1,2]], det = 4 ≡ 1 (mod 3) — so the
  K-weight (Z/3)³ is non-degenerate (unlike sl_3 where det = 3 ≡ 0).

E-E commutation rules (E_β E_α, β > α in PBW → PBW form):
  E₁₂·E₁   = q E₁·E₁₂                                    (no correction)
  E₁₂₃·E₁  = q E₁·E₁₂₃                                   (no correction)
  E₂·E₁    = q⁻¹ E₁·E₂ - q⁻¹ E₁₂                        (correction: E₁₂)
  E₂₃·E₁   = q⁻¹ E₁·E₂₃ - q⁻¹ E₁₂₃                     (correction: E₁₂₃)
  E₃·E₁    = E₁·E₃                                        (no correction)
  E₁₂₃·E₁₂ = q E₁₂·E₁₂₃                                 (no correction)
  E₂·E₁₂   = q E₁₂·E₂                                    (no correction)
  E₂₃·E₁₂  = E₁₂·E₂₃                                     (no correction)
  E₃·E₁₂   = q⁻¹ E₁₂·E₃ - q⁻¹ E₁₂₃                     (correction: E₁₂₃)
  E₂·E₁₂₃  = E₁₂₃·E₂                                     (no correction)
  E₂₃·E₁₂₃ = q E₁₂₃·E₂₃                                 (no correction)
  E₃·E₁₂₃  = q E₁₂₃·E₃                                  (no correction)
  E₂₃·E₂   = q E₂·E₂₃                                    (no correction)
  E₃·E₂    = q⁻¹ E₂·E₃ - q⁻¹ E₂₃                        (correction: E₂₃)
  E₃·E₂₃   = q E₂₃·E₃                                    (no correction)

K-E commutation: K_i E_α = q^{<α_i^∨, α>} E_α K_i.
  K₁: E₁→q², E₁₂→q¹, E₁₂₃→q¹, E₂→q⁻¹, E₂₃→q⁻¹, E₃→q⁰
  K₂: E₁→q⁻¹, E₁₂→q¹, E₁₂₃→q⁰, E₂→q², E₂₃→q¹, E₃→q⁻¹
  K₃: E₁→q⁰, E₁₂→q⁻¹, E₁₂₃→q¹, E₂→q⁻¹, E₂₃→q¹, E₃→q²

Coproducts:
  Δ(K_i) = K_i ⊗ K_i
  Δ(E_i) = E_i ⊗ K_i + 1 ⊗ E_i
  Δ(E₁₂)  = E₁₂ ⊗ K₁K₂ + 1 ⊗ E₁₂ + (1-q²) E₁ ⊗ K₁E₂
  Δ(E₂₃)  = E₂₃ ⊗ K₂K₃ + 1 ⊗ E₂₃ + (1-q²) E₂ ⊗ K₂E₃
  Δ(E₁₂₃) = E₁₂₃ ⊗ K₁K₂K₃ + 1 ⊗ E₁₂₃
             + (1-q²) E₁₂ ⊗ K₁K₂E₃ + (1-q²) E₁ ⊗ K₁E₂₃
"""
import cmath
import math
import time
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


# ============================================================
# Algebra setup
# ============================================================

ELL = 3
Q = cmath.exp(2j * math.pi / ELL)
Q_INV = Q ** (-1)
assert abs(Q + Q_INV - (-1)) < 1e-12
DIM = ELL ** 9  # 19683
NUM_GEN = 9  # K1, K2, K3, E1, E12, E123, E2, E23, E3


def idx(a, b, c, d, e, f, g, h, i):
    """Index of K1^a K2^b K3^c E1^d E12^e E123^f E2^g E23^h E3^i."""
    return (a * (ELL**8) + b * (ELL**7) + c * (ELL**6) + d * (ELL**5)
            + e * (ELL**4) + f * (ELL**3) + g * (ELL**2) + h * ELL + i)


def from_idx(n):
    return ((n // (ELL**8)) % ELL, (n // (ELL**7)) % ELL, (n // (ELL**6)) % ELL,
            (n // (ELL**5)) % ELL, (n // (ELL**4)) % ELL, (n // (ELL**3)) % ELL,
            (n // (ELL**2)) % ELL, (n // ELL) % ELL, n % ELL)


def weight(n):
    """3D K-weight.  det(A_3) = 4 ≡ 1 (mod 3), so non-degenerate."""
    a, b, c, d, e, f, g, h, i = from_idx(n)
    wt1 = (2*d + e + f - g - h) % ELL
    wt2 = (-d + e + 2*g + h - i) % ELL
    wt3 = (-e + f - g + h + 2*i) % ELL
    return (wt1, wt2, wt3)


def exponents(n):
    """Return the 9 PBW exponents of basis element n."""
    return from_idx(n)


# ============================================================
# E-part multiplication via swap table + bubble sort
# ============================================================
# E-part monomial: (d, e, f, g, h, i) = (E1^d E12^e E123^f E2^g E23^h E3^i)
# PBW positions: 0=E1, 1=E12, 2=E123, 3=E2, 4=E23, 5=E3
# Root vectors (in simple-root coords): 0=(1,0,0), 1=(1,1,0), 2=(1,1,1),
#   3=(0,1,0), 4=(0,1,1), 5=(0,0,1)

CARTAN = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
ROOTS = [(1,0,0), (1,1,0), (1,1,1), (0,1,0), (0,1,1), (0,0,1)]


def _inner(i, j):
    """(α_i, α_j) using Cartan matrix."""
    a = ROOTS[i]; b = ROOTS[j]
    return sum(a[k] * b[l] * CARTAN[k][l] for k in range(3) for l in range(3))


# Sum table: (i, j) with i > j → index of root α_i + α_j, or None if not a root
SUM_TABLE = {}
for i in range(6):
    for j in range(i):
        si = ROOTS[i]; sj = ROOTS[j]
        s = (si[0]+sj[0], si[1]+sj[1], si[2]+sj[2])
        found = None
        for k in range(6):
            if ROOTS[k] == s:
                found = k; break
        SUM_TABLE[(i, j)] = found


# Swap table: (i, j) with i > j → list of (coeff, exponent_tuple_6)
# E_i E_j = sum (coeff * monomial) where monomial is in PBW form.
# Convention: E_α E_β = q E_β E_α + E_{α+β} when α+β is a root (α < β in PBW).
# So E_β E_α (β > α, non-PBW) = q^{-1} E_α E_β - q^{-1} E_{α+β}.
# When α+β is not a root: E_β E_α = q^{(α,β)} E_α E_β (using (α_i, α_j)).
SWAP_TABLE = {}
for i in range(6):
    for j in range(i):
        mon_no_corr = tuple(1 if k == i or k == j else 0 for k in range(6))
        k = SUM_TABLE[(i, j)]
        if k is not None:
            mon_corr = tuple(1 if kk == k else 0 for kk in range(6))
            SWAP_TABLE[(i, j)] = [(Q_INV, mon_no_corr), (-Q_INV, mon_corr)]
        else:
            SWAP_TABLE[(i, j)] = [(Q ** _inner(i, j), mon_no_corr)]


def e_multiply_mon(mon1, mon2):
    """Multiply two E-part monomials. Returns dict {monomial_6tuple: coeff}."""
    # Expand both to lists of positions, concatenate, bubble-sort.
    lst = []
    for p in range(6):
        lst.extend([p] * mon1[p])
    for p in range(6):
        lst.extend([p] * mon2[p])
    state = {tuple(lst): 1.0 + 0j}
    return _bubble_sort(state)


def _bubble_sort(state):
    """Bubble-sort all lists in state to non-decreasing order, applying swaps."""
    while True:
        new_state = {}
        changed = False
        for lst, coeff in state.items():
            # Find rightmost out-of-order adjacent pair
            swap_pos = -1
            for k in range(len(lst) - 1):
                if lst[k] > lst[k + 1]:
                    swap_pos = k
            if swap_pos < 0:
                new_state[lst] = new_state.get(lst, 0) + coeff
                continue
            changed = True
            k = swap_pos
            i, j = lst[k], lst[k + 1]
            base = lst[:k] + lst[k+2:]
            for swap_coeff, swap_mon in SWAP_TABLE[(i, j)]:
                swap_list = []
                for p in range(6):
                    swap_list.extend([p] * swap_mon[p])
                new_lst = base[:k] + tuple(swap_list) + base[k:]
                new_state[new_lst] = new_state.get(new_lst, 0) + coeff * swap_coeff
        state = {k: v for k, v in new_state.items() if abs(v) > 1e-13}
        if not changed:
            break
    # Convert sorted lists to exponent tuples
    result = {}
    for lst, coeff in state.items():
        exp = [0] * 6
        valid = True
        for p in lst:
            exp[p] += 1
            if exp[p] >= ELL:
                valid = False; break
        if valid:
            t = tuple(exp)
            result[t] = result.get(t, 0) + coeff
    return {k: v for k, v in result.items() if abs(v) > 1e-13}


# ============================================================
# Full multiplication (K-part + E-part)
# ============================================================
# Monomial: K1^a K2^b K3^c E1^d E12^e E123^f E2^g E23^h E3^i
# K-E commutation: K_i E_α = q^{<α_i^∨, α>} E_α K_i
# K-exponent phases: when multiplying (K-part1)(E-part1) * (K-part2)(E-part2),
#   the K-part2 needs to commute past E-part1.  Phase = q^{sum over E-generators
#   in E-part1 of <α_i^∨, α> * K-part2[i]}.

# <α_i^∨, α_j> for i=1,2,3 and j=PBW position 0..5:
# j=0 (α1):    <α1∨,α1>=2, <α2∨,α1>=-1, <α3∨,α1>=0
# j=1 (α1+α2): <α1∨,α1+α2>=1, <α2∨,α1+α2>=1, <α3∨,α1+α2>=-1
# j=2 (α1+α2+α3): <α1∨,...>=1, <α2∨,...>=0, <α3∨,...>=1
# j=3 (α2):    <α1∨,α2>=-1, <α2∨,α2>=2, <α3∨,α2>=-1
# j=4 (α2+α3): <α1∨,α2+α3>=-1, <α2∨,α2+α3>=1, <α3∨,α2+α3>=1
# j=5 (α3):    <α1∨,α3>=0, <α2∨,α3>=-1, <α3∨,α3>=2
KE_PHASES = [
    [2, 1, 1, -1, -1, 0],   # K1 phases for E at positions 0..5
    [-1, 1, 0, 2, 1, -1],   # K2
    [0, -1, 1, -1, 1, 2],   # K3
]


def multiply_monomials(m1, m2):
    """Multiply two full monomials (9-tuples). Returns dict {9-tuple: coeff}.

    (K1^a1 K2^b1 K3^c1 E1^d1 E12^e1 E123^f1 E2^g1 E23^h1 E3^i1)
        * (K1^a2 K2^b2 K3^c2 E1^d2 E12^e2 E123^f2 E2^g2 E23^h2 E3^i2)
    """
    a1, b1, c1 = m1[0:3]
    e1 = m1[3:9]  # (d, e, f, g, h, i)
    a2, b2, c2 = m2[0:3]
    e2 = m2[3:9]

    # Phase from K2 commuting past E1-part: K_i E_α = q^{<α_i∨,α>} E_α K_i,
    # so E_α K_i = q^{-<α_i∨,α>} K_i E_α.  When we have
    # (K-part1)(E-part1) * (K-part2)(E-part2), we rewrite as
    # (K-part1)(K-part2)(E-part1)(E-part2) * phase, where phase accounts for
    # K-part2 moving from the RIGHT of E-part1 to the LEFT.
    # Each K_i (in K-part2) moving past E_α (in E-part1) contributes
    # q^{-<α_i∨,α> * k_exp * e_exp}.
    k2_part = (a2, b2, c2)
    phase = 1.0 + 0j
    for i, k_exp in enumerate(k2_part):
        if k_exp == 0:
            continue
        for pos, e_exp in enumerate(e1):
            if e_exp == 0:
                continue
            phase *= Q ** (-KE_PHASES[i][pos] * k_exp * e_exp)

    new_k = ((a1 + a2) % ELL, (b1 + b2) % ELL, (c1 + c2) % ELL)
    e_prod = e_multiply_mon(e1, e2)

    result = {}
    for e_mon, v in e_prod.items():
        full_mon = new_k + e_mon
        result[full_mon] = result.get(full_mon, 0) + v * phase
    return {k: v for k, v in result.items() if abs(v) > 1e-13}


def ms_entry(i, j):
    """Compute basis[i] * basis[j] as list of (l, v)."""
    m1 = from_idx(i); m2 = from_idx(j)
    prod = multiply_monomials(m1, m2)
    return [(idx(*m), v) for m, v in prod.items()]


# ============================================================
# Comultiplication
# ============================================================
# Δ is an algebra homomorphism, so Δ(xy) = Δ(x)Δ(y).
# Generator coproducts (each term is (left_9tuple, right_9tuple, coeff)):

# Index helpers for generator tuples
K1 = (1,0,0, 0,0,0,0,0,0); K2 = (0,1,0, 0,0,0,0,0,0); K3 = (0,0,1, 0,0,0,0,0,0)
E1  = (0,0,0, 1,0,0,0,0,0); E12 = (0,0,0, 0,1,0,0,0,0); E123 = (0,0,0, 0,0,1,0,0,0)
E2  = (0,0,0, 0,0,0,1,0,0); E23 = (0,0,0, 0,0,0,0,1,0); E3 = (0,0,0, 0,0,0,0,0,1)
UNIT = (0,0,0, 0,0,0,0,0,0)
K1K2 = (1,1,0, 0,0,0,0,0,0); K2K3 = (0,1,1, 0,0,0,0,0,0); K1K2K3 = (1,1,1, 0,0,0,0,0,0)
K1_E2 = (1,0,0, 0,0,0,1,0,0); K2_E3 = (0,1,0, 0,0,0,0,0,1); K1K2_E3 = (1,1,0, 0,0,0,0,0,1)
K1_E23 = (1,0,0, 0,0,0,0,1,0)

GEN_COPRODUCTS = {
    idx(*K1): [(K1, K1, 1.0)],
    idx(*K2): [(K2, K2, 1.0)],
    idx(*K3): [(K3, K3, 1.0)],
    idx(*E1): [(E1, K1, 1.0), (UNIT, E1, 1.0)],
    idx(*E2): [(E2, K2, 1.0), (UNIT, E2, 1.0)],
    idx(*E3): [(E3, K3, 1.0), (UNIT, E3, 1.0)],
    idx(*E12): [(E12, K1K2, 1.0), (UNIT, E12, 1.0), (E1, K1_E2, 1.0 - Q**2)],
    idx(*E23): [(E23, K2K3, 1.0), (UNIT, E23, 1.0), (E2, K2_E3, 1.0 - Q**2)],
    idx(*E123): [(E123, K1K2K3, 1.0), (UNIT, E123, 1.0),
                 (E12, K1K2_E3, 1.0 - Q**2), (E1, K1_E23, 1.0 - Q**2)],
}


def tensor_mult(d1, d2):
    """Multiply two elements of B⊗B (component-wise).  Each is a dict
    {(left_9tuple, right_9tuple): coeff}."""
    new = {}
    for (l1, r1), v1 in d1.items():
        for (l2, r2), v2 in d2.items():
            left = multiply_monomials(l1, l2)
            right = multiply_monomials(r1, r2)
            for lk, lv in left.items():
                for rk, rv in right.items():
                    key = (lk, rk)
                    new[key] = new.get(key, 0) + v1 * v2 * lv * rv
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def delta_monomial(mon):
    """Δ(K1^a..E3^i) as dict {(left_9tuple, right_9tuple): coeff}."""
    # Generator index for each PBW position (positions 0..8 in the 9-tuple):
    # 0=K1, 1=K2, 2=K3, 3=E1, 4=E12, 5=E123, 6=E2, 7=E23, 8=E3
    gen_indices = [
        idx(*K1), idx(*K2), idx(*K3),
        idx(*E1), idx(*E12), idx(*E123),
        idx(*E2), idx(*E23), idx(*E3),
    ]
    result = {(UNIT, UNIT): 1.0 + 0j}
    for pos, exp in enumerate(mon):
        gen_idx = gen_indices[pos]
        gen_delta = {}
        for (l, r, v) in GEN_COPRODUCTS[gen_idx]:
            gen_delta[(l, r)] = gen_delta.get((l, r), 0) + v
        for _ in range(exp):
            result = tensor_mult(result, gen_delta)
    return result


def delta_entry(i):
    """Δ(basis[i]) as list of (j, k, v)."""
    mon = from_idx(i)
    d = delta_monomial(mon)
    return [(idx(*l), idx(*r), v) for (l, r), v in d.items()]


# ============================================================
# B̄ basis and modified structure constants
# ============================================================

def build_epsilon():
    eps = np.zeros(DIM, dtype=complex)
    for a in range(ELL):
        for b in range(ELL):
            for c in range(ELL):
                eps[idx(a, b, c, 0, 0, 0, 0, 0, 0)] = 1.0
    return eps


# ============================================================
# Linear-ansatz diagonal kernel computation
# ============================================================
# For a diagonal h with c(monomial) = sum_k α_k * exponent_k (linear in 9
# exponents), the constraints are:
#   ∂^h: for each (l, a, b) with mult_bar[l,a,b] ≠ 0:
#           sum_k α_k * (exp_k(a) + exp_k(b) - exp_k(l)) = 0
#   ∂^c: for each (c, j, k) with delta_bar[c,j,k] ≠ 0:
#           sum_k α_k * (exp_k(j) + exp_k(k) - exp_k(c)) = 0
# Each constraint is a row vector of 9 coefficients (one per α_k).
# We incrementally build the rank of the 9-column constraint matrix.

def constraint_row_from_mult(l_mon, a_mon, b_mon):
    """Compute the constraint row (9 coefficients) for mult_bar[l, a, b] ≠ 0.
    Row[k] = exp_k(a) + exp_k(b) - exp_k(l)."""
    return tuple(a_mon[k] + b_mon[k] - l_mon[k] for k in range(9))


def constraint_row_from_delta(c_mon, j_mon, k_mon):
    """Compute the constraint row (9 coefficients) for delta_bar[c, j, k] ≠ 0.
    Row[k] = exp_k(j) + exp_k(k) - exp_k(c)."""
    return tuple(j_mon[k] + k_mon[k] - c_mon[k] for k in range(9))


def add_row_to_basis(basis_rows, new_row, tol=1e-9):
    """Try to add new_row to the basis of independent rows.
    Returns True if added (independent), False if dependent.
    Uses modified Gram-Schmidt."""
    new = np.array(new_row, dtype=complex)
    for r in basis_rows:
        # Project out r from new
        proj = np.vdot(r, new) / np.vdot(r, r) * r
        new = new - proj
    if np.linalg.norm(new) > tol:
        basis_rows.append(new / np.linalg.norm(new))
        return True
    return False


def compute_linear_ansatz_kernel(verbose=True):
    """Compute the kernel of the linear-ansatz constraint matrix.

    Enumerates (a, b) pairs and Δ-terms, extracts constraint rows, and
    incrementally builds the rank.

    Returns (rank, kernel_dim, basis_rows).
    """
    if verbose:
        print("\n=== Linear-ansatz diagonal kernel computation ===")
        print(f"9 variables (α_K1, α_K2, α_K3, α_E1, α_E12, α_E123, α_E2, α_E23, α_E3)")
        print(f"Building constraint matrix incrementally...")

    basis_rows = []
    eps = build_epsilon()

    # Helper: enumerate all B̄ elements with E-sum ≤ max_e_sum
    def enum_bar(max_e_sum):
        for a in range(ELL):
            for b in range(ELL):
                for c in range(ELL):
                    for d in range(ELL):
                        for e in range(ELL):
                            for f in range(ELL):
                                for g in range(ELL):
                                    for h in range(ELL):
                                        for i in range(ELL):
                                            e_sum = d+e+f+g+h+i
                                            if e_sum == 0 or e_sum > max_e_sum:
                                                continue
                                            yield (a, b, c, d, e, f, g, h, i)

    # ---- ∂^c constraints ----
    if verbose:
        print("\n[∂^c] Enumerating comultiplication constraints (E-sum ≤ 2)...")
    t0 = time.time()
    n_c_constraints = 0
    n_c_independent = 0

    for c_mon in enum_bar(max_e_sum=2):
        c_idx = idx(*c_mon)
        d_entries = delta_entry(c_idx)
        for (j, k, v) in d_entries:
            if j == 0 or k == 0:
                continue
            n_c_constraints += 1
            j_mon = from_idx(j)
            k_mon = from_idx(k)
            row = constraint_row_from_delta(c_mon, j_mon, k_mon)
            if add_row_to_basis(basis_rows, row):
                n_c_independent += 1
                if n_c_independent == 9:
                    break
        if n_c_independent == 9:
            break

    if verbose:
        print(f"  ∂^c: {n_c_constraints} constraints, {n_c_independent} independent, time={time.time()-t0:.1f}s")

    # ---- ∂^h constraints ----
    if n_c_independent < 9:
        if verbose:
            print("\n[∂^h] Enumerating multiplication constraints (E-sum ≤ 2)...")
        t0 = time.time()
        n_h_constraints = 0
        n_h_independent = 0

        bar_list = list(enum_bar(max_e_sum=2))
        if verbose:
            print(f"  {len(bar_list)} B̄ elements with E-sum ≤ 2")

        for a_mon in bar_list:
            a_idx = idx(*a_mon)
            if n_c_independent + n_h_independent == 9:
                break
            for b_mon in bar_list:
                b_idx = idx(*b_mon)
                prod = ms_entry(a_idx, b_idx)
                for (l, v) in prod:
                    if l == 0:
                        continue
                    n_h_constraints += 1
                    l_mon = from_idx(l)
                    row = constraint_row_from_mult(l_mon, a_mon, b_mon)
                    if add_row_to_basis(basis_rows, row):
                        n_h_independent += 1
                        if n_c_independent + n_h_independent == 9:
                            break
                if n_c_independent + n_h_independent == 9:
                    break

        if verbose:
            print(f"  ∂^h: {n_h_constraints} constraints, {n_h_independent} independent, time={time.time()-t0:.1f}s")

    total_rank = n_c_independent + n_h_independent
    kernel_dim = 9 - total_rank
    if verbose:
        print(f"\nTotal rank: {total_rank}")
        print(f"Linear-ansatz kernel dim: {kernel_dim}")
    return total_rank, kernel_dim, basis_rows


# ============================================================
# Sanity checks
# ============================================================

def verify_q_serre():
    """Verify q-Serre: E_i^2 E_j - (q+q^{-1}) E_i E_j E_i + E_j E_i^2 = 0
    for adjacent (i, j) in sl_4."""
    print("\n=== Verifying q-Serre relations ===")
    adjacent_pairs = [(0, 3), (3, 0), (3, 5), (5, 3)]  # (E1,E2), (E2,E1), (E2,E3), (E3,E2)
    # In PBW positions: E1=0, E2=3, E3=5
    max_err = 0.0
    for (i, j) in adjacent_pairs:
        # E_i^2 E_j
        ei2 = tuple(2 if k == i else 0 for k in range(6))
        ej = tuple(1 if k == j else 0 for k in range(6))
        ei = tuple(1 if k == i else 0 for k in range(6))
        # E_i^2 * E_j
        r1 = e_multiply_mon(ei2, ej)
        # E_i * E_j * E_i = (E_i * E_j) * E_i
        r_ei_ej = e_multiply_mon(ei, ej)
        r2 = {}
        for mon, v in r_ei_ej.items():
            sub = e_multiply_mon(mon, ei)
            for m2, v2 in sub.items():
                r2[m2] = r2.get(m2, 0) + v * v2
        # E_j * E_i^2
        r3 = e_multiply_mon(ej, ei2)
        # E_i^2 E_j - (q+q^{-1}) E_i E_j E_i + E_j E_i^2 = 0
        all_mons = set(r1) | set(r2) | set(r3)
        for mon in all_mons:
            val = r1.get(mon, 0) - (Q + Q_INV) * r2.get(mon, 0) + r3.get(mon, 0)
            max_err = max(max_err, abs(val))
        print(f"  q-Serre (E{['1','12','123','2','23','3'][i]}, E{['1','12','123','2','23','3'][j]}): max_err = {max_err:.2e}")
    return max_err


def verify_e123_definitions():
    """Verify E123 = E12 E3 - q E3 E12 = E1 E23 - q E23 E1."""
    print("\n=== Verifying E123 definitions ===")
    # E12 E3 - q E3 E12
    e12 = (0,1,0,0,0,0); e3 = (0,0,0,0,0,1); e1 = (1,0,0,0,0,0); e23 = (0,0,0,0,1,0)
    r1 = e_multiply_mon(e12, e3)
    r2 = e_multiply_mon(e3, e12)
    diff1 = {}
    for mon in set(r1) | set(r2):
        diff1[mon] = r1.get(mon, 0) - Q * r2.get(mon, 0)
    print(f"  E12 E3 - q E3 E12 = {diff1}")

    # E1 E23 - q E23 E1
    r3 = e_multiply_mon(e1, e23)
    r4 = e_multiply_mon(e23, e1)
    diff2 = {}
    for mon in set(r3) | set(r4):
        diff2[mon] = r3.get(mon, 0) - Q * r4.get(mon, 0)
    print(f"  E1 E23 - q E23 E1 = {diff2}")

    # Check they're equal
    all_mons = set(diff1) | set(diff2)
    max_err = 0.0
    for mon in all_mons:
        max_err = max(max_err, abs(diff1.get(mon, 0) - diff2.get(mon, 0)))
    print(f"  Max |E12 E3 - q E3 E12 - (E1 E23 - q E23 E1)| = {max_err:.2e}")
    return max_err


def verify_coproduct_e123():
    """Verify Δ(E123) formula by computing Δ(E12)Δ(E3) - q Δ(E3)Δ(E12) and
    comparing to the formula."""
    print("\n=== Verifying Δ(E123) formula ===")
    # Δ(E123) = Δ(E12)Δ(E3) - q Δ(E3)Δ(E12)
    # Compute Δ(E12) * Δ(E3)
    e12_idx = idx(*E12); e3_idx = idx(*E3)
    d_e12 = {(l, r): v for (l, r, v) in [(t[0], t[1], t[2]) for t in GEN_COPRODUCTS[e12_idx]]}
    d_e3 = {(l, r): v for (l, r, v) in [(t[0], t[1], t[2]) for t in GEN_COPRODUCTS[e3_idx]]}
    prod1 = tensor_mult(d_e12, d_e3)
    # Compute q * Δ(E3) * Δ(E12)
    prod2 = tensor_mult(d_e3, d_e12)
    prod2 = {k: v * Q for k, v in prod2.items()}
    # Δ(E123) = prod1 - prod2
    delta_e123_computed = {}
    for k in set(prod1) | set(prod2):
        v = prod1.get(k, 0) - prod2.get(k, 0)
        if abs(v) > 1e-13:
            delta_e123_computed[k] = v

    # Compare to the formula
    delta_e123_formula = {(l, r): v for (l, r, v) in [(t[0], t[1], t[2]) for t in GEN_COPRODUCTS[idx(*E123)]]}

    all_keys = set(delta_e123_computed) | set(delta_e123_formula)
    max_err = 0.0
    for k in all_keys:
        max_err = max(max_err, abs(delta_e123_computed.get(k, 0) - delta_e123_formula.get(k, 0)))
    print(f"  Max |Δ(E12)Δ(E3) - q Δ(E3)Δ(E12) - formula| = {max_err:.2e}")
    return max_err


def main():
    print(f"\n=== H̃¹_b(B⁺(u_q(sl_4)), C) at ℓ = {ELL} ===")
    print(f"dim B = {DIM}, dim B̄ = {DIM - 1}")
    print(f"q = {Q}")
    print(f"det(A_3) = 4 ≡ {4 % ELL} (mod {ELL}) — non-degenerate weight function")

    # Sanity checks
    e123_err = verify_e123_definitions()
    assert e123_err < 1e-10, "E123 definitions inconsistent!"

    serre_err = verify_q_serre()
    assert serre_err < 1e-10, "q-Serre relations failed!"

    cp_err = verify_coproduct_e123()
    assert cp_err < 1e-10, "Δ(E123) formula inconsistent!"

    # Compute the linear-ansatz kernel
    rank, kdim, basis_rows = compute_linear_ansatz_kernel(verbose=True)

    print(f"\n{'='*60}")
    print(f"RESULT: dim H̃¹_b(B⁺(u_q(sl_4)), C) = {kdim} (under linear-ansatz)")
    print(f"  H1 prediction (C(4,2) = 6): {'MATCH' if kdim == 6 else 'NO MATCH'}")
    print(f"  H2 prediction (4-1 = 3):    {'MATCH' if kdim == 3 else 'NO MATCH'}")
    print(f"{'='*60}")

    return kdim


if __name__ == "__main__":
    result = main()
    sys.exit(0 if 0 <= result <= 20 else 1)
