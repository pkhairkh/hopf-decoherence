#!/usr/bin/env python3
"""
Compute dim H̃¹_b(B⁺(u_q(sl_3)), C) at ℓ = 3.

The Mastnak–Witherspoon bialgebra cohomology H̃¹_b(B) at degree 1 (truncated
normalized complex) is the kernel of

    ∂_b : Hom(B̄, B̄) → Hom(B̄⊗B̄, B̄) ⊕ Hom(B̄, B̄⊗B̄)

    ∂_b h = (∂^h h, -∂^c h)

    (∂^h h)(a, b) = a·h(b) - h(a·b) + h(a)·b     (derivation)
    (∂^c h)(c)    = c₁⊗h(c₂) - Δ(h(c)) + h(c₁)⊗c₂   (coderivation)

with a, b, c ∈ B̄ = ker ε.  Since (Tot B⁰_+)¹ = 0, B̃¹_b = 0 and
H̃¹_b = Z̃¹_b = ker(∂_b).

B⁺(u_q(sl_3)) at ℓ=3:
  Generators K₁, K₂, E₁, E₂, E₁₂ (E₁₂ := E₁E₂ - q E₂E₁ is the Lusztig
  root vector for α₁+α₂).
  PBW basis {K₁^a K₂^b E₁^c E₁₂^e E₂^d : 0 ≤ a,b,c,e,d ≤ 2}, dim = 243.
  ε(K₁^a K₂^b E₁^c E₁₂^e E₂^d) = 1 if c=e=d=0 else 0.  dim B̄ = 242.
  2D weights (for block decomposition): wt = ((c+e) mod 3, (e+d) mod 3) ∈ (Z/3)².
  Coproduct:
    Δ(K_i) = K_i ⊗ K_i,         i = 1, 2,
    Δ(E_1) = E_1 ⊗ K_1 + 1 ⊗ E_1,
    Δ(E_2) = E_2 ⊗ K_2 + 1 ⊗ E_2,
    Δ(E₁₂) = E₁₂ ⊗ K₁K₂ + 1 ⊗ E₁₂ + (1 - q²) E₁ ⊗ K₁ E₂
            [= E₁₂ ⊗ K₁K₂ + 1 ⊗ E₁₂ + (q⁻¹ - q) K₁ E₂ ⊗ E₁  (equivalent form)]

The Δ(E₁₂) formula is derived from Δ(E₁₂) = Δ(E₁)Δ(E₂) - q Δ(E₂)Δ(E₁) and
the K-E commutation relations; see analysis note for the derivation.  We
sanity-check coassociativity and counitality numerically.

We compute dim ker(∂_b) by 2D weight-shift decomposition (9 blocks, one per
shift s ∈ (Z/3)²), then dim H̃¹_b = Σ_s (dim input_s - rank(∂_b_s)).
"""
import cmath
import math
import time
import sys
import numpy as np
from scipy import sparse
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import eigsh


# ============================================================
# Algebra setup (PBW basis, dim 243)
# ============================================================

ELL = 3
Q = cmath.exp(2j * math.pi / ELL)
Q_INV = Q ** (-1)
assert abs(Q + Q_INV - (-1)) < 1e-12
DIM = ELL ** 5  # 243


def idx(a, b, c, e, d):
    """Index of K1^a K2^b E1^c E12^e E2^d (PBW order: K1, K2, E1, E12, E2)."""
    return a * (ELL**4) + b * (ELL**3) + c * (ELL**2) + e * ELL + d


def from_idx(i):
    return (i // (ELL**4), (i // (ELL**3)) % ELL, (i // (ELL**2)) % ELL,
            (i // ELL) % ELL, i % ELL)


def weight(i):
    """2D K-weight of K1^a K2^b E1^c E12^e E2^d.

    The K-eigenvalue of E1^c E12^e E2^d is determined by the adjoint action of
    K1, K2: K_i E_alpha K_i^{-1} = q^{<alpha_i^vee, alpha>} E_alpha.
    For sl_3 in the basis (alpha_1, alpha_2):
        alpha_1 = (2, -1), alpha_2 = (-1, 2), alpha_1+alpha_2 = (1, 1)
    So K-weight of E1^c E12^e E2^d is:
        wt = c*(2,-1) + e*(1,1) + d*(-1,2) = (2c+e-d, -c+e+2d)  (mod 3)
    Note: det(Cartan) = 3 ≡ 0 mod 3, so this is NOT equivalent to (c+e, e+d)
    and the two weight functions give different decompositions.
    The TRUE K-weight (this function) is what is preserved by multiplication
    and comultiplication.
    """
    _, _, c, e, d = from_idx(i)
    return ((2*c + e - d) % ELL, (-c + e + 2*d) % ELL)


# ============================================================
# Multiplication: E-part (PBW order E1^c E12^e E2^d)
# (Lifted from scripts/verify_sl3_bplus_hh2.py.)
# ============================================================

def e_mult_by_E1_left(elem):
    new = {}
    for (c, e, d), v in elem.items():
        if c + 1 < ELL:
            key = (c + 1, e, d)
            new[key] = new.get(key, 0) + v
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def e_mult_by_E12_left(elem):
    """E12 * (E1^c E12^e E2^d) = q^c E1^c E12^{e+1} E2^d (using E12 E1 = q E1 E12)."""
    new = {}
    for (c, e, d), v in elem.items():
        if e + 1 < ELL:
            phase = Q ** c
            key = (c, e + 1, d)
            new[key] = new.get(key, 0) + v * phase
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def _e2_e1_power(c):
    """E2 * E1^c in PBW form {(c', e', d'): coeff}."""
    if c == 0:
        return {(0, 0, 1): 1.0}
    prev = _e2_e1_power(c - 1)
    result = {}
    for (cp, ep, dp), v in prev.items():
        if cp + 1 < ELL:
            key = (cp + 1, ep, dp)
            result[key] = result.get(key, 0) + v * Q_INV
    if c - 1 < ELL and 1 < ELL:
        key = (c - 1, 1, 0)
        phase = -Q ** (c - 2)
        result[key] = result.get(key, 0) + phase
    return {k: v for k, v in result.items() if abs(v) > 1e-13}


def e_mult_by_E2_left(elem):
    new = {}
    for (c, e, d), v in elem.items():
        if c > 0:
            terms = _e2_e1_power(c)
            for (cp, ep, dp), vp in terms.items():
                if ep + e < ELL and dp + d < ELL:
                    phase = Q ** (e * dp)
                    key = (cp, ep + e, dp + d)
                    new[key] = new.get(key, 0) + v * vp * phase
        else:
            if d + 1 < ELL:
                phase = Q ** e
                key = (0, e, d + 1)
                new[key] = new.get(key, 0) + v * phase
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def e_mult_by_E1_right(elem):
    new = {}
    for (c, e, d), v in elem.items():
        if d > 0:
            terms = _e2power_e1(d)
            for (cp, ep, dp), vp in terms.items():
                if c + cp < ELL and e + ep < ELL:
                    phase = Q ** (e * cp)
                    key = (c + cp, e + ep, dp)
                    new[key] = new.get(key, 0) + v * vp * phase
        else:
            if c + 1 < ELL:
                phase = Q ** e
                key = (c + 1, e, 0)
                new[key] = new.get(key, 0) + v * phase
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def _e2power_e1(d):
    """E2^d * E1 in PBW form {(c, e, d'): coeff}."""
    if d == 0:
        return {(1, 0, 0): 1.0}
    prev = _e2power_e1(d - 1)
    result = {}
    for (cp, ep, dp), vp in prev.items():
        if dp + 1 < ELL:
            key = (cp, ep, dp + 1)
            result[key] = result.get(key, 0) + vp * Q_INV
    if d - 1 < ELL:
        key = (0, 1, d - 1)
        phase = -Q ** (d - 2)
        result[key] = result.get(key, 0) + phase
    return {k: v for k, v in result.items() if abs(v) > 1e-13}


def e_mult_by_E12_right(elem):
    """(E1^c E12^e E2^d) * E12 = q^d E1^c E12^{e+1} E2^d."""
    new = {}
    for (c, e, d), v in elem.items():
        if e + 1 < ELL:
            phase = Q ** d
            key = (c, e + 1, d)
            new[key] = new.get(key, 0) + v * phase
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def e_mult_by_E2_right(elem):
    new = {}
    for (c, e, d), v in elem.items():
        if d + 1 < ELL:
            key = (c, e, d + 1)
            new[key] = new.get(key, 0) + v
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def e_multiply(left, right):
    """Multiply two E-parts (both in PBW form). left * right."""
    result = {}
    for (c2, e2, d2), v2 in right.items():
        acc = dict(left)
        for _ in range(c2):
            acc = e_mult_by_E1_right(acc)
        for _ in range(e2):
            acc = e_mult_by_E12_right(acc)
        for _ in range(d2):
            acc = e_mult_by_E2_right(acc)
        for mon, coeff in acc.items():
            result[mon] = result.get(mon, 0) + coeff * v2
    return {k: v for k, v in result.items() if abs(v) > 1e-13}


def multiply_monomials(a1, b1, c1, e1, d1, a2, b2, c2, e2, d2):
    """(K1^a1 K2^b1 E1^c1 E12^e1 E2^d1) * (K1^a2 K2^b2 E1^c2 E12^e2 E2^d2)."""
    phase_k = (Q ** (-2*a2*c1 - a2*e1 + a2*d1)) * (Q ** (b2*c1 - b2*e1 - 2*b2*d1))
    a_sum = (a1 + a2) % ELL
    b_sum = (b1 + b2) % ELL
    left_e = {(c1, e1, d1): 1.0}
    right_e = {(c2, e2, d2): 1.0}
    e_prod = e_multiply(left_e, right_e)
    result = {}
    for (c, e, d), v in e_prod.items():
        key = (a_sum, b_sum, c, e, d)
        result[key] = result.get(key, 0) + v * phase_k
    return {k: v for k, v in result.items() if abs(v) > 1e-13}


def build_mult(dim):
    """ms[i][j] = [(l, v), ...] for basis[i]*basis[j] = Σ v · basis[l]."""
    ms = [[[] for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        a1, b1, c1, e1, d1 = from_idx(i)
        for j in range(dim):
            a2, b2, c2, e2, d2 = from_idx(j)
            prod = multiply_monomials(a1, b1, c1, e1, d1, a2, b2, c2, e2, d2)
            for (a, b, c, e, d), v in prod.items():
                l = idx(a, b, c, e, d)
                ms[i][j].append((l, v))
    return ms


# ============================================================
# Comultiplication
# ============================================================

def tensor_mult(d1, d2):
    """Multiply two elements of B⊗B (component-wise)."""
    new = {}
    for (l1, r1), v1 in d1.items():
        for (l2, r2), v2 in d2.items():
            left = multiply_monomials(*l1, *l2)
            right = multiply_monomials(*r1, *r2)
            for lk, lv in left.items():
                for rk, rv in right.items():
                    key = (lk, rk)
                    new[key] = new.get(key, 0) + v1 * v2 * lv * rv
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def delta_monomial(a, b, c, e, d):
    """Δ(K1^a K2^b E1^c E12^e E2^d) as a dict {(l_tuple, r_tuple): coeff}.

    Uses the algebra-homomorphism property of Δ.
    Generator coproducts:
        Δ(K1) = K1 ⊗ K1
        Δ(K2) = K2 ⊗ K2
        Δ(E1) = E1 ⊗ K1 + 1 ⊗ E1
        Δ(E2) = E2 ⊗ K2 + 1 ⊗ E2
        Δ(E12) = E12 ⊗ K1 K2 + 1 ⊗ E12 + (1 - q²) E1 ⊗ K1 E2
                 [derived from Δ(E12) = Δ(E1)Δ(E2) - q Δ(E2)Δ(E1)]
    """
    result = {((0, 0, 0, 0, 0), (0, 0, 0, 0, 0)): 1.0}  # 1 ⊗ 1
    for _ in range(a):
        result = tensor_mult(result, {((1, 0, 0, 0, 0), (1, 0, 0, 0, 0)): 1.0})  # · Δ(K1)
    for _ in range(b):
        result = tensor_mult(result, {((0, 1, 0, 0, 0), (0, 1, 0, 0, 0)): 1.0})  # · Δ(K2)
    for _ in range(c):
        result = tensor_mult(result, {  # · Δ(E1) = E1 ⊗ K1 + 1 ⊗ E1
            ((0, 0, 1, 0, 0), (1, 0, 0, 0, 0)): 1.0,
            ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0)): 1.0,
        })
    for _ in range(e):
        result = tensor_mult(result, {  # · Δ(E12)
            ((0, 0, 0, 1, 0), (1, 1, 0, 0, 0)): 1.0,                  # E12 ⊗ K1 K2
            ((0, 0, 0, 0, 0), (0, 0, 0, 1, 0)): 1.0,                  # 1 ⊗ E12
            ((0, 0, 1, 0, 0), (1, 0, 0, 0, 1)): 1.0 - Q**2,           # (1-q²) E1 ⊗ K1 E2
        })
    for _ in range(d):
        result = tensor_mult(result, {  # · Δ(E2) = E2 ⊗ K2 + 1 ⊗ E2
            ((0, 0, 0, 0, 1), (0, 1, 0, 0, 0)): 1.0,
            ((0, 0, 0, 0, 0), (0, 0, 0, 0, 1)): 1.0,
        })
    return result


def build_delta(dim):
    """Delta[c] = [(j, k, v), ...] for Δ(basis[c]) = Σ v · basis[j]⊗basis[k]."""
    Delta = [None] * dim
    for c in range(dim):
        a, b, c_, e_, d_ = from_idx(c)
        Delta[c] = [(idx(*lt), idx(*rt), v)
                    for (lt, rt), v in delta_monomial(a, b, c_, e_, d_).items()]
    return Delta


def verify_coassoc(Delta, dim, n_samples=30, seed=0):
    """Verify (Δ⊗1)Δ = (1⊗Δ)Δ on a sample of basis elements."""
    rng = np.random.default_rng(seed)
    indices = list(rng.choice(dim, size=min(n_samples, dim), replace=False))
    max_err = 0.0
    for c in indices:
        left, right = {}, {}
        for (j, k, v) in Delta[c]:
            for (j2, k2, v2) in Delta[j]:
                key = (j2, k2, k)
                left[key] = left.get(key, 0) + v * v2
            for (j2, k2, v2) in Delta[k]:
                key = (j, j2, k2)
                right[key] = right.get(key, 0) + v * v2
        for key in set(left) | set(right):
            max_err = max(max_err, abs(left.get(key, 0) - right.get(key, 0)))
    return max_err, indices


def verify_counital(Delta, dim, epsilon):
    """Verify (ε⊗1)Δ = id and (1⊗ε)Δ = id."""
    max_err = 0.0
    for c in range(dim):
        for target in range(dim):
            s_l = sum(v * epsilon[j] for (j, k, v) in Delta[c] if k == target)
            s_r = sum(v * epsilon[k] for (j, k, v) in Delta[c] if j == target)
            expected = 1.0 if target == c else 0.0
            max_err = max(max_err, abs(s_l - expected), abs(s_r - expected))
    return max_err


def verify_delta_E12(Delta, dim):
    """Verify Δ(E12) = Δ(E1)Δ(E2) - q Δ(E2)Δ(E1) explicitly."""
    E1 = idx(0, 0, 1, 0, 0)
    E2 = idx(0, 0, 0, 0, 1)
    E12 = idx(0, 0, 0, 1, 0)
    # Compute Δ(E1)Δ(E2) - q Δ(E2)Δ(E1) as a dict {(j, k): v}.
    prod = {}
    for (j1, k1, v1) in Delta[E1]:
        for (j2, k2, v2) in Delta[E2]:
            # (j1 ⊗ k1) (j2 ⊗ k2) = (j1 j2) ⊗ (k1 k2).  Need to expand the products.
            # Use ms[j1][j2] for the left factor, ms[k1][k2] for the right.
            for (lj, lv) in ms_global[j1][j2]:
                for (rk, rv) in ms_global[k1][k2]:
                    key = (lj, rk)
                    prod[key] = prod.get(key, 0) + v1 * v2 * lv * rv
    for (j1, k1, v1) in Delta[E2]:
        for (j2, k2, v2) in Delta[E1]:
            for (lj, lv) in ms_global[j1][j2]:
                for (rk, rv) in ms_global[k1][k2]:
                    key = (lj, rk)
                    prod[key] = prod.get(key, 0) - Q * v1 * v2 * lv * rv
    # Compare to Delta[E12].
    expected = {(j, k): v for (j, k, v) in Delta[E12]}
    max_err = 0.0
    for key in set(prod) | set(expected):
        max_err = max(max_err, abs(prod.get(key, 0) - expected.get(key, 0)))
    return max_err


# Global reference for the verify_delta_E12 helper.
ms_global = None


# ============================================================
# B̄ basis and modified structure constants
# ============================================================
#
# B̄ = ker ε, dim 242.  Basis: b̄_i := basis[i] - ε[i]·basis[0]  for i = 1..242.
#
#   mult_bar[l, a, b] = mult[l, a, b] - ε[b]·δ_{l,a} - ε[a]·δ_{l,b}
#   delta_bar[c, j, k] = delta[c, j, k]  (just drop unit factors)

def build_epsilon(dim):
    eps = np.zeros(dim, dtype=complex)
    for a in range(ELL):
        for b in range(ELL):
            eps[idx(a, b, 0, 0, 0)] = 1.0
    return eps


def build_mult_bar(ms, epsilon, dim):
    mult_bar = {}
    for a in range(1, dim):
        for b in range(1, dim):
            terms = {l: v for (l, v) in ms[a][b]}
            for l in range(1, dim):
                v = terms.get(l, 0.0)
                if l == a:
                    v -= epsilon[b]
                if l == b:
                    v -= epsilon[a]
                if abs(v) > 1e-13:
                    mult_bar[(l, a, b)] = v
    return mult_bar


def build_delta_bar(Delta, dim):
    delta_bar = {}
    for c in range(1, dim):
        for (j, k, v) in Delta[c]:
            if j == 0 or k == 0:
                continue
            delta_bar[(c, j, k)] = v
    return delta_bar


# ============================================================
# ∂_b per shift (vectorized)
# ============================================================

def compute_shift_block(s, B_bar, weights, dim,
                        ms_by_b_arr, ms_by_a_arr, inv_mult_bar_arr,
                        inv_delta_left_by_k_arr, inv_delta_right_by_k_arr, delta_by_c_arr,
                        verbose=False):
    """Build ∂_b_s as a sparse matrix and compute its rank.

    Returns (n_cols, n_rows_h, n_rows_c, rank, nullity, time_info).

    Inputs (all pre-converted to numpy arrays):
      ms_by_b_arr[j]            = (a_arr, t_arr, v_arr)  with mult_bar[t, a, j] = v
      ms_by_a_arr[j]            = (b_arr, t_arr, v_arr)  with mult_bar[t, j, b] = v
      inv_mult_bar_arr[k]       = (a_arr, b_arr, v_arr)  with mult_bar[k, a, b] = v
      inv_delta_left_by_k_arr[k]  = (c_arr, al_arr, v_arr)  with delta_bar[c, al, k] = v
      inv_delta_right_by_k_arr[k] = (c_arr, be_arr, v_arr)  with delta_bar[c, k, be] = v
      delta_by_c_arr[j]         = (al_arr, be_arr, v_arr)  with delta_bar[j, al, be] = v
    """
    t_start = time.time()

    # Enumerate input columns (j, k) with (wt(j) - wt(k)) ≡ s (mod 3).
    cols_list = [(j, k) for j in B_bar for k in B_bar
                 if ((weights[j][0] - weights[k][0]) % ELL,
                     (weights[j][1] - weights[k][1]) % ELL) == s]
    col_map = {pair: i for i, pair in enumerate(cols_list)}
    n_cols = len(cols_list)

    # Enumerate ∂^h output rows (a, b, t) with (wt(t) - wt(a) - wt(b)) ≡ s.
    h_rows_list = [(a, b, t) for a in B_bar for b in B_bar for t in B_bar
                   if ((weights[t][0] - weights[a][0] - weights[b][0]) % ELL,
                       (weights[t][1] - weights[a][1] - weights[b][1]) % ELL) == s]
    n_rows_h = len(h_rows_list)
    h_row_map = np.full((dim, dim, dim), -1, dtype=np.int32)
    for i, (a, b, t) in enumerate(h_rows_list):
        h_row_map[a, b, t] = i

    # Enumerate ∂^c output rows (c, α, β) with (wt(α) + wt(β) - wt(c)) ≡ s.
    c_rows_list = [(c, al, be) for c in B_bar for al in B_bar for be in B_bar
                   if ((weights[al][0] + weights[be][0] - weights[c][0]) % ELL,
                       (weights[al][1] + weights[be][1] - weights[c][1]) % ELL) == s]
    n_rows_c = len(c_rows_list)
    c_row_map = np.full((dim, dim, dim), -1, dtype=np.int32)
    for i, (c, al, be) in enumerate(c_rows_list):
        c_row_map[c, al, be] = i

    t_enum = time.time() - t_start

    if n_cols == 0:
        return (0, n_rows_h, n_rows_c, 0, 0, (t_enum, 0.0, 0.0, 0.0))

    # Build sparse matrix entries (vectorized per column).
    rows_chunks = []
    cols_chunks = []
    vals_chunks = []
    n_entries = 0

    t_build_start = time.time()
    for col_idx, (j, k) in enumerate(cols_list):
        # ∂^h term 1: row (a, k, t), val = mult_bar[t, a, j].
        a_arr, t_arr, v_arr = ms_by_b_arr[j]
        if a_arr.size > 0:
            row_arr = h_row_map[a_arr, k, t_arr]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append(row_arr[valid].astype(np.int64))
                cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                vals_chunks.append(v_arr[valid])
                n_entries += int(np.sum(valid))

        # ∂^h term 2: row (a, b, j), val = -mult_bar[k, a, b].
        a_arr, b_arr, v_arr = inv_mult_bar_arr[k]
        if a_arr.size > 0:
            row_arr = h_row_map[a_arr, b_arr, j]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append(row_arr[valid].astype(np.int64))
                cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                vals_chunks.append(-v_arr[valid])
                n_entries += int(np.sum(valid))

        # ∂^h term 3: row (k, b, t), val = mult_bar[t, j, b].
        b_arr, t_arr, v_arr = ms_by_a_arr[j]
        if b_arr.size > 0:
            row_arr = h_row_map[k, b_arr, t_arr]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append(row_arr[valid].astype(np.int64))
                cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                vals_chunks.append(v_arr[valid])
                n_entries += int(np.sum(valid))

        # -∂^c term 1: row (c, α, j) in ∂^c output, val = -delta_bar[c, α, k].
        # inv_delta_left_by_k_arr[k] = (c_arr, al_arr, v_arr) with delta_bar[c, al, k] = v.
        c_arr, al_arr, v_arr = inv_delta_left_by_k_arr[k]
        if c_arr.size > 0:
            row_arr = c_row_map[c_arr, al_arr, j]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append((n_rows_h + row_arr[valid]).astype(np.int64))
                cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                vals_chunks.append(-v_arr[valid])
                n_entries += int(np.sum(valid))

        # -∂^c term 2: only c = k.  row (k, α, β), val = +delta_bar[j, α, β].
        al_arr, be_arr, v_arr = delta_by_c_arr[j]
        if al_arr.size > 0:
            row_arr = c_row_map[k, al_arr, be_arr]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append((n_rows_h + row_arr[valid]).astype(np.int64))
                cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                vals_chunks.append(v_arr[valid])
                n_entries += int(np.sum(valid))

        # -∂^c term 3: row (c, j, β) in ∂^c output, val = -delta_bar[c, k, β].
        # inv_delta_right_by_k_arr[k] = (c_arr, be_arr, v_arr) with delta_bar[c, k, be] = v.
        c_arr, be_arr, v_arr = inv_delta_right_by_k_arr[k]
        if c_arr.size > 0:
            row_arr = c_row_map[c_arr, j, be_arr]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append((n_rows_h + row_arr[valid]).astype(np.int64))
                cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                vals_chunks.append(-v_arr[valid])
                n_entries += int(np.sum(valid))

    t_build = time.time() - t_build_start

    if not rows_chunks:
        rank = 0
        return (n_cols, n_rows_h, n_rows_c, rank, n_cols, (t_enum, t_build, 0.0, 0.0))

    rows_arr = np.concatenate(rows_chunks)
    cols_arr = np.concatenate(cols_chunks)
    vals_arr = np.concatenate(vals_chunks)

    n_rows_total = n_rows_h + n_rows_c
    A = sparse.csr_matrix((vals_arr, (rows_arr, cols_arr)),
                          shape=(n_rows_total, n_cols), dtype=complex)

    t_gram_start = time.time()
    # Build G = A^* A as a SPARSE matrix (rather than .toarray()) so we can use
    # shift-invert eigsh (much faster and more memory-efficient than full eigvalsh
    # on the dense Gram matrix; also avoids the Cholesky-shortcut bug — see below).
    G_sparse = (A.conj().T @ A).tocsr()
    # Symmetrize (sparse)
    G_sparse = (G_sparse + G_sparse.conj().T) * 0.5
    t_gram = time.time() - t_gram_start

    t_eig_start = time.time()
    if n_cols == 0:
        rank = 0
    else:
        # Robust rank computation via shift-invert eigsh.
        #
        # NOTE: An earlier version of this code used a Cholesky-based shortcut:
        #   try: cholesky(G_dense); rank = n_cols
        #   except LinAlgError: <fallback to eigvalsh>
        # That shortcut is BUGGY: when G has small-but-positive "zero" eigenvalues
        # (numerical roundoff ~ 1e-13 instead of exactly 0), LAPACK's ?potrf will
        # succeed because it has no tolerance, reporting G as numerically positive
        # definite and giving rank = n_cols (nullity = 0). This is what happened
        # for sl_3 at ℓ=3: the true Gram matrix has 2 zero eigenvalues (~5e-14),
        # but Cholesky spuriously succeeded, masking the 2 cocycles and reporting
        # dim H̃¹_b = 0 (wrong; correct value is 2).
        #
        # The fix: skip Cholesky entirely. Use shift-invert eigsh on the sparse
        # Gram matrix to find the k smallest eigenvalues directly. eigsh with
        # sigma=0 uses shift-invert, which is reliable for finding eigenvalues
        # near zero. We then count how many are below tolerance.
        k_eigs = min(20, n_cols - 1) if n_cols > 1 else 1
        # Smallest k eigenvalues (by magnitude, via shift-invert)
        try:
            eigs_small = eigsh(G_sparse, k=k_eigs, sigma=0, which='LM',
                               return_eigenvectors=False, tol=1e-12, maxiter=20000)
            eigs_small = np.abs(eigs_small)
        except Exception:
            # Fallback: dense eigvalsh (slower but always works for small matrices)
            G_dense = G_sparse.toarray()
            G_dense = (G_dense + G_dense.conj().T) * 0.5
            all_eigs = np.sort(np.abs(eigvalsh(G_dense, check_finite=False)))[::-1]
            eigs_small = all_eigs[-k_eigs:] if len(all_eigs) >= k_eigs else all_eigs

        # Largest eigenvalue for tolerance
        try:
            eigs_large = eigsh(G_sparse, k=1, which='LM',
                               return_eigenvectors=False, tol=1e-6, maxiter=10000)
            largest = float(np.abs(eigs_large[0]))
        except Exception:
            largest = float(np.max(eigs_small)) if len(eigs_small) > 0 else 1.0

        # Count zero eigenvalues among the smallest k
        tol = max(n_cols, 1) * largest * 1e-10
        n_zero_small = int(np.sum(eigs_small < tol))

        # If all k smallest are zero, we may have undercounted; bump up k.
        # In practice this never triggers for our shifts (max nullity observed = 2).
        if n_zero_small == k_eigs and k_eigs < n_cols:
            # Increase k and retry (one-shot)
            k_eigs = min(k_eigs * 2, n_cols - 1)
            try:
                eigs_small = eigsh(G_sparse, k=k_eigs, sigma=0, which='LM',
                                   return_eigenvectors=False, tol=1e-12, maxiter=20000)
                eigs_small = np.abs(eigs_small)
            except Exception:
                pass
            n_zero_small = int(np.sum(eigs_small < tol))

        # Nullity = number of zero eigenvalues (we don't need exact rank, just nullity)
        rank = n_cols - n_zero_small
    t_eig = time.time() - t_eig_start

    nullity = n_cols - rank
    return (n_cols, n_rows_h, n_rows_c, rank, nullity, (t_enum, t_build, t_gram, t_eig))


def compute(dim, verbose=True):
    global ms_global

    if verbose:
        print(f"\n=== H̃¹_b(B⁺(u_q(sl_3)), C) at ℓ = {ELL} ===")
        print(f"dim B = {dim}, dim B̄ = {dim-1}")
        print(f"q = {Q}")

    t0 = time.time()
    print("Building multiplication table...")
    ms = build_mult(dim)
    ms_global = ms
    print(f"  Mult table: {time.time()-t0:.1f}s")

    t0 = time.time()
    print("Building comultiplication table...")
    Delta = build_delta(dim)
    print(f"  Delta table: {time.time()-t0:.1f}s")

    # Sanity checks.
    print("Verifying Δ(E12) = Δ(E1)Δ(E2) - q Δ(E2)Δ(E1)...")
    e12_err = verify_delta_E12(Delta, dim)
    print(f"  Δ(E12) check max error: {e12_err:.2e}")
    assert e12_err < 1e-10, "Δ(E12) formula inconsistent!"

    print("Verifying coassociativity on a sample...")
    coassoc_err, sample_idx = verify_coassoc(Delta, dim, n_samples=30)
    print(f"  Coassoc max error (sample of {len(sample_idx)}): {coassoc_err:.2e}")

    epsilon = build_epsilon(dim)
    print("Verifying counitality...")
    counital_err = verify_counital(Delta, dim, epsilon)
    print(f"  Counital max error: {counital_err:.2e}")
    assert counital_err < 1e-10, "Counitality failed!"

    # Build B̄ basis and modified structure constants.
    print("Building B̄ basis structure constants...")
    t0 = time.time()
    mult_bar = build_mult_bar(ms, epsilon, dim)
    delta_bar = build_delta_bar(Delta, dim)
    print(f"  mult_bar nnz: {len(mult_bar)}")
    print(f"  delta_bar nnz: {len(delta_bar)}")
    print(f"  B̄ basis: {time.time()-t0:.1f}s")

    B_bar = list(range(1, dim))
    weights = [weight(i) for i in range(dim)]

    # Precompute inverse tables as numpy arrays.
    print("Precomputing inverse tables...")
    t0 = time.time()
    inv_mult_bar = {l: [] for l in B_bar}
    ms_by_b = {b: [] for b in B_bar}
    ms_by_a = {a: [] for a in B_bar}
    for (l, a, b), v in mult_bar.items():
        inv_mult_bar[l].append((a, b, v))
        ms_by_b[b].append((a, l, v))
        ms_by_a[a].append((b, l, v))

    inv_delta_left = {}
    inv_delta_right = {}
    delta_by_c = {c: [] for c in B_bar}
    # Also build "by k" versions for vectorized ∂^c term 1 / term 3 loops.
    inv_delta_left_by_k = {k: [] for k in B_bar}     # k → list of (c, al, v) with delta_bar[c, al, k] = v
    inv_delta_right_by_k = {k: [] for k in B_bar}    # k → list of (c, be, v) with delta_bar[c, k, be] = v
    for (c, j, k), v in delta_bar.items():
        inv_delta_left.setdefault((c, k), []).append((j, v))
        inv_delta_right.setdefault((c, j), []).append((k, v))
        delta_by_c[c].append((j, k, v))
        inv_delta_left_by_k[k].append((c, j, v))     # here "al" = j
        inv_delta_right_by_k[k].append((c, j, v))    # here "be" = j, but we need (c, be) with be=right of (c,k,be)

    # Re-build inv_delta_right_by_k correctly: for fixed k (the LEFT tensor factor of delta_bar),
    # collect (c, be, v) with delta_bar[c, k, be] = v.
    inv_delta_right_by_k = {k: [] for k in B_bar}
    for (c, j, k), v in delta_bar.items():
        # delta_bar[c, j, k] = v: here j is the LEFT tensor factor, k is the RIGHT.
        # We want, for fixed LEFT factor = j, the list of (c, be=k, v) — i.e., inv_delta_right_by_k[j].
        inv_delta_right_by_k[j].append((c, k, v))

    # Convert to numpy arrays for vectorized access.
    def to_arr3(lst):
        if not lst:
            return (np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                    np.array([], dtype=complex))
        return (np.array([x[0] for x in lst], dtype=np.int32),
                np.array([x[1] for x in lst], dtype=np.int32),
                np.array([x[2] for x in lst], dtype=complex))

    def to_arr2(lst):
        if not lst:
            return (np.array([], dtype=np.int32), np.array([], dtype=complex))
        return (np.array([x[0] for x in lst], dtype=np.int32),
                np.array([x[1] for x in lst], dtype=complex))

    ms_by_b_arr = {b: to_arr3(ms_by_b[b]) for b in B_bar}
    ms_by_a_arr = {a: to_arr3(ms_by_a[a]) for a in B_bar}
    inv_mult_bar_arr = {k: to_arr3(inv_mult_bar[k]) for k in B_bar}
    inv_delta_left_by_k_arr = {k: to_arr3(inv_delta_left_by_k[k]) for k in B_bar}
    inv_delta_right_by_k_arr = {k: to_arr3(inv_delta_right_by_k[k]) for k in B_bar}
    delta_by_c_arr = {c: to_arr3(delta_by_c[c]) for c in B_bar}
    print(f"  Inverse tables: {time.time()-t0:.1f}s")

    # Compute per-shift rank.
    all_shifts = [(s1, s2) for s1 in range(ELL) for s2 in range(ELL)]
    total_rank = 0
    total_input = 0
    per_shift = []
    print(f"\nComputing ∂_b per shift (9 shifts)...")
    t_total = time.time()
    for s in all_shifts:
        ts = time.time()
        n_cols, n_rows_h, n_rows_c, rank, nullity, t_info = compute_shift_block(
            s, B_bar, weights, dim,
            ms_by_b_arr, ms_by_a_arr, inv_mult_bar_arr,
            inv_delta_left_by_k_arr, inv_delta_right_by_k_arr, delta_by_c_arr)
        total_rank += rank
        total_input += n_cols
        per_shift.append((s, n_cols, n_rows_h, n_rows_c, rank, nullity, t_info))
        if verbose:
            print(f"  shift {s}: cols={n_cols}, rows_h={n_rows_h}, rows_c={n_rows_c}, "
                  f"rank={rank}, nullity={nullity}  "
                  f"[enum={t_info[0]:.1f}s, build={t_info[1]:.1f}s, gram={t_info[2]:.1f}s, eig={t_info[3]:.1f}s]  "
                  f"total {time.time()-ts:.1f}s")

    dim_ker = total_input - total_rank
    if verbose:
        print(f"\nTotal input dim (dim Hom(B̄,B̄)): {total_input}")
        print(f"Total rank of ∂_b:                {total_rank}")
        print(f"dim H̃¹_b(B⁺(u_q(sl_3)), C)  = dim ker(∂_b) = {dim_ker}")
        print(f"Conjecture's structural prediction at A₂, ℓ=3:  3 (= C(3, 2))")
        if dim_ker == 3:
            print(f"STATUS: STRUCTURAL PREDICTION VERIFIED AT A₂ (modulo im(ῑ)=6)")
        elif dim_ker > 9:
            print(f"STATUS: CONJECTURE REFUTED AT A₂ (dim > 9 = C(3,2) + 2|Φ⁺|)")
        elif dim_ker == 2:
            print(f"STATUS: STRUCTURAL PREDICTION REFUTED AT A₂ (true dim = 2 ≠ 3).")
            print(f"        The full conjecture dim HH²(u_q(sl_3)) = 9 may still hold")
            print(f"        (e.g. with structural split dim im(δ)=2, dim im(ῑ)=7),")
            print(f"        but is not directly verifiable in this sandbox.")
        elif 3 < dim_ker <= 9:
            print(f"STATUS: STRUCTURAL DECOMPOSITION NEEDS REFINEMENT (3 < dim ≤ 9)")
        elif dim_ker < 2:
            print(f"STATUS: UNEXPECTED (dim < 2) — investigate")
        print(f"Total time: {time.time()-t_total:.1f}s")
    return dim_ker, per_shift


if __name__ == "__main__":
    result, per_shift = compute(DIM, verbose=True)
    sys.exit(0 if 0 <= result <= 20 else 1)
