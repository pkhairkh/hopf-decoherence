#!/usr/bin/env python3
"""
Braided Hochschild cohomology of B(V) for sl_2 at ℓ = 3 (proof-of-concept).

Reference: C. Negron, "Braided Hochschild cohomology and Hopf actions"
(arXiv:1511.07059).

Setup
-----
G = (Z/3)Z, with generator K and q = e^{2πi/3}, so q^3 = 1 and 1 + q + q^2 = 0.
The Nichols algebra B(V) is the braided Hopf algebra in YD_G^G

    B(V) = C[E] / (E^3),    dim = 3,

with right G-action  E . K = q^2 E   (so E^i . K^j = q^{2ij} E^i)
and  right G-coaction  δ(E) = E ⊗ K  (so δ(E^i) = E^i ⊗ K^i).

The braiding on B(V) ⊗ B(V) is
    c(E^i ⊗ E^j) = (E^j)_(0) ⊗ (E^i . (E^j)_(1)) = E^j ⊗ q^{2ij} E^i
                 = q^{2ij} (E^j ⊗ E^i).

The bosonization  B^+ = B(V) ⋊ C[G]  is the small quantum Borel u_q(b_+(sl_2)),
with basis {E^i K^j : 0 ≤ i, j ≤ 2}, dim = 9.  Its ordinary Hochschild
cohomology is known (paper §6, verified in scripts/verify_bplus_sl2_hh2.py):

    dim_C HH^2(B^+(sl_2), C) = 1   at ℓ = 3.

Negron's braided bar complex (Def 3.4 + Prop 3.3 of arXiv:1511.07059)
----------------------------------------------------------------------

For an algebra B in the braided monoidal category Z = H-mod (here
H = D(E), the Drinfeld double of E = C[G], so Z ≅ YD_E^E), the braided
bar complex is

    C^•_c(B) = 0 → Hom(k, B) → Hom(B, B) → Hom(B^{⊗2}, B) → ...

with differential  d^c : C^n_c → C^{n+1}_c  given by (Prop 3.3 / eq. 3.5):

  degree 0:    -d^c(f)(b) = (r^j · b) r_j · f(1) - f(1) b

  degree n ≥ 1:  (-1)^{|f|+1} d^c(f)(b ⊗ y ⊗ b')
                  = (r^j · b) r_j · f(y ⊗ b') + f(d_BB(b ⊗ y ⊗ b'))
                    + (-1)^{n+1} f(b ⊗ y) b'

where R = Σ_j r_j ⊗ r^j  is the R-matrix of H = D(E), y ∈ B^{⊗(n-1)},
and d_BB is the ordinary bar differential

    d_BB(b_1 ⊗ ... ⊗ b_l) = Σ_{i=1}^{l-1} (-1)^i b_1 ⊗ ... ⊗ b_i b_{i+1} ⊗ ... ⊗ b_l.

For our abelian G = (Z/3)Z, the R-matrix of D(C[G]) is
    R = Σ_{g ∈ G} g ⊗ δ_g  ∈  C[G] ⊗ C[G]^*,
and its flipped version is R_21 = Σ_g δ_g ⊗ g.  Acting on B(V):

    (r^j · b) r_j · m  =  Σ_{g ∈ G} (δ_g · b) (g · m)

where the left D(E)-action on the right YD-module B(V) is
    g · b   = b · g          (right action, no antipode)
    δ_g · b = b_(0) δ_g(b_(1))  (right coaction contracted with δ_g).

Substituting b = E^i, m = E^k:

    δ_g · E^i = E^i · δ_g(K^i) = δ_{g, K^i} E^i,
    K^i · E^k = E^k · K^i = q^{2ik} E^k,

so
    (r^j · E^i) r_j · E^k = (δ_{K^i} · E^i)(K^i · E^k) = E^i · q^{2ik} E^k
                          = q^{2ik} E^{i+k}    (zero if i+k ≥ 3).

This is precisely the braided multiplication  E^i ·_br E^k = q^{2ik} E^{i+k}.

Sign convention
---------------

With d_BB summed from i = 1 (as in Negron's displayed formula),
d^c(f) for f ∈ C^n_c is  (-1)^{n+1}  times the "standard-sign" braided
Hochschild differential d_br.  Since (-1)^{n+1} is a nonzero scalar,
ker d^c = ker d_br and im d^c = im d_br, so the cohomology is the same.
We use d_br (the standard-sign version) throughout:

  (d_br f)(b_0, ..., b_n) = b_0 ·_br f(b_1, ..., b_n)
                            + Σ_{i=0}^{n-1} (-1)^{i+1} f(b_0, ..., b_i b_{i+1}, ..., b_n)
                            + (-1)^{n+1} f(b_0, ..., b_{n-1}) · b_n.

The G-invariants
----------------

The right G-action on C^n_c = Hom(B^{⊗n}, B) is the diagonal action

    (f · K)(b_1, ..., b_n) = f(b_1 · K^{-1}, ..., b_n · K^{-1}) · K.

For the basis element f_{j_1, ..., j_n, m}  (f(E^{j_1}, ..., E^{j_n}) = E^m,
zero on other inputs), this gives weight

    (f · K) = q^{2(m - j_1 - ... - j_n)} f,

so f is G-invariant iff  m ≡ j_1 + ... + j_n  (mod 3).

Since d_br is a map in YD_G^G (Negron Prop 3.3 (2)), it preserves the
G-weight decomposition, and we may compute HH^2(B^+) = H^2_c(B)^G by
restricting d^1, d^2 to the G-invariant subspaces.

Expected dimensions (sl_2, ℓ = 3, dim B(V) = 3, |G| = 3)
---------------------------------------------------------

Braided bar complex of B(V) (full):
    C^0_c = B(V)                          dim 3
    C^1_c = Hom(B(V), B(V))               dim 9
    C^2_c = Hom(B(V)^{⊗2}, B(V))          dim 27
    C^3_c = Hom(B(V)^{⊗3}, B(V))          dim 81

G-invariant subspaces (one weight out of three):
    (C^0_c)^G  dim 1     (just the unit 1 ∈ B(V))
    (C^1_c)^G  dim 3
    (C^2_c)^G  dim 9
    (C^3_c)^G  dim 27

Negron's theorem (Cor 4.7) gives  HH^•(B^+) = H^•_c(B^+)^G  where B^+ is
the bosonization and H^•_c(B^+) is the braided HH of the *bosonization*.
The task's intended shortcut  HH^•(B^+) = H^•_c(B(V))^G  is NOT directly
Negron's theorem (it requires the *relative* braided complex and Prop 4.2
to identify it with HH^•(B(V), B^+)).  We compute both:

  (A)  dim H^2_c(B(V))^G  -- braided HH of Nichols algebra, then invariants.
                              (Task's literal ask; chain dims 3, 9, 27, 81.)

  (B)  dim HH^2(B(V), B^+)^G -- ordinary HH of B(V) with B^+-coefficients,
                                 then invariants.  This is the *correct*
                                 Negron computation (Prop 4.2 + Cor 4.7).
                                 Chain dims 9, 27, 81, 243.

The two computations share the same |G|-per-degree dimensional reduction
relative to the ordinary bar complex of B^+ (dims 9, 81, 729, 6561).
"""
import cmath
import math
import sys
import time
import numpy as np

# ============================================================
# Ring setup: q = e^{2πi/3}, q^3 = 1, 1 + q + q^2 = 0.
# ============================================================
ELL = 3
Q = cmath.exp(2j * math.pi / ELL)  # q
Q2 = Q * Q                          # q^2 = e^{4πi/3}, a primitive 3rd root.
Q_INV = Q ** (-1)
assert abs(Q ** ELL - 1.0) < 1e-12
assert abs(1 + Q + Q2) < 1e-12

# Nichols algebra  B(V) = C[E]/(E^3).  Basis {1, E, E^2} indexed 0, 1, 2.
N = 3  # dim B(V)


def E_mult(i, j):
    """E^i · E^j in B(V): E^{i+j} if i+j ≤ 2, else 0 (E^3 = 0)."""
    s = i + j
    if s >= N:
        return None
    return s


def E_br_mult(i, k):
    """Braided multiplication  E^i ·_br E^k = q^{2ik} E^{i+k},  0 if i+k ≥ 3."""
    s = i + k
    if s >= N:
        return None
    return (Q2 ** (i * k), s)


# ============================================================
# Indexing helpers for the braided bar complex of B(V).
# C^n_c has basis  {f_{j_1,...,j_n,m}}  with  f(E^{j_1},...,E^{j_n}) = E^m.
# ============================================================
def idx_c0(m):
    return m


def idx_c1(j, m):
    return j * N + m


def idx_c2(j1, j2, m):
    return (j1 * N + j2) * N + m


def idx_c3(j1, j2, j3, m):
    return ((j1 * N + j2) * N + j3) * N + m


# ============================================================
# Differential  d^0 : C^0_c → C^1_c.
# (d^0 f)(b) = b ·_br f - f · b        (f ∈ B(V), b ∈ B(V))
# For f = E^k, b = E^i:   d^0(E^k)(E^i) = q^{2ik} E^{i+k} - E^{k+i}
#                                            = (q^{2ik} - 1) E^{i+k}      (if i+k ≤ 2)
#                                            = 0                          (if i+k ≥ 3)
# Matrix element  M[idx_c1(i, m), k] = (q^{2ik} - 1)  if m = i+k ≤ 2  else 0.
# ============================================================
def build_d0():
    M = np.zeros((N * N, N), dtype=complex)  # 9 x 3
    for k in range(N):
        for i in range(N):
            res = E_br_mult(i, k)
            if res is None:
                continue
            coef, m = res
            M[idx_c1(i, m), k] = coef - 1.0
    return M


# ============================================================
# Differential  d^1 : C^1_c → C^2_c.
# (d^1 f)(b_0, b_1) = b_0 ·_br f(b_1) - f(b_0 b_1) + f(b_0) · b_1.
# For basis f with f(E^{i_in}) = E^{m_in}, evaluated at (E^i, E^j):
#   Term 1 (left, braided): E^i ·_br f(E^j)  — nonzero iff j == i_in, value
#                            E^i ·_br E^{m_in} = q^{2 i m_in} E^{i + m_in}
#                            (zero if i + m_in ≥ 3).
#   Term 2 (middle, -):     -f(E^i · E^j)    — nonzero iff i+j ≤ 2 and
#                            i+j == i_in, value -E^{m_in}.
#   Term 3 (right, +):      f(E^i) · E^j     — nonzero iff i == i_in, value
#                            E^{m_in} · E^j = E^{m_in + j}  (zero if m_in+j ≥ 3).
# ============================================================
def build_d1():
    M = np.zeros((N ** 3, N * N), dtype=complex)  # 27 x 9
    for i_in in range(N):
        for m_in in range(N):
            col = idx_c1(i_in, m_in)
            for i in range(N):
                for j in range(N):
                    # Term 1.
                    if j == i_in:
                        res = E_br_mult(i, m_in)
                        if res is not None:
                            coef, m = res
                            M[idx_c2(i, j, m), col] += coef
                    # Term 2.
                    s_ij = E_mult(i, j)
                    if s_ij is not None and s_ij == i_in:
                        M[idx_c2(i, j, m_in), col] += -1.0
                    # Term 3.
                    if i == i_in:
                        s = E_mult(m_in, j)
                        if s is not None:
                            M[idx_c2(i, j, s), col] += 1.0
    return M


# ============================================================
# Differential  d^2 : C^2_c → C^3_c.
# (d^2 f)(b_0, b_1, b_2) = b_0 ·_br f(b_1, b_2)
#                          - f(b_0 b_1, b_2) + f(b_0, b_1 b_2) - f(b_0, b_1) · b_2.
# For basis f with f(E^{i_in}, E^{j_in}) = E^{m_in}:
#   Term 1 (left, braided): E^i ·_br f(E^j, E^k)  — nonzero iff (j,k) == (i_in,j_in),
#                            value q^{2 i m_in} E^{i + m_in}  (zero if i + m_in ≥ 3).
#   Term 2 (middle, -):     -f(E^i E^j, E^k)      — nonzero iff i+j ≤ 2, (i+j, k) == (i_in, j_in),
#                            value -E^{m_in}.
#   Term 3 (middle, +):     f(E^i, E^j E^k)       — nonzero iff j+k ≤ 2, (i, j+k) == (i_in, j_in),
#                            value +E^{m_in}.
#   Term 4 (right, -):      -f(E^i, E^j) · E^k    — nonzero iff (i, j) == (i_in, j_in),
#                            value -E^{m_in + k}  (zero if m_in + k ≥ 3).
# ============================================================
def build_d2():
    M = np.zeros((N ** 4, N ** 3), dtype=complex)  # 81 x 27
    for i_in in range(N):
        for j_in in range(N):
            for m_in in range(N):
                col = idx_c2(i_in, j_in, m_in)
                for i in range(N):
                    for j in range(N):
                        for k in range(N):
                            row = idx_c3(i, j, k, 0)  # placeholder, will set per-term
                            # Term 1.
                            if j == i_in and k == j_in:
                                res = E_br_mult(i, m_in)
                                if res is not None:
                                    coef, m = res
                                    M[idx_c3(i, j, k, m), col] += coef
                            # Term 2.
                            s_ij = E_mult(i, j)
                            if s_ij is not None and s_ij == i_in and k == j_in:
                                M[idx_c3(i, j, k, m_in), col] += -1.0
                            # Term 3.
                            s_jk = E_mult(j, k)
                            if s_jk is not None and i == i_in and s_jk == j_in:
                                M[idx_c3(i, j, k, m_in), col] += 1.0
                            # Term 4.
                            if i == i_in and j == j_in:
                                s = E_mult(m_in, k)
                                if s is not None:
                                    M[idx_c3(i, j, k, s), col] += -1.0
    return M


# ============================================================
# G-invariant mask for C^n_c.
# A basis element  f_{j_1, ..., j_n, m}  has G-weight q^{2(m - j_1 - ... - j_n)}.
# It is G-invariant (weight 0) iff  m ≡ j_1 + ... + j_n  (mod 3).
# ============================================================
def ginv_mask_c0():
    """Indices in C^0 (basis E^m) with G-weight 0, i.e., m ≡ 0 (mod 3)."""
    return [m for m in range(N) if m % ELL == 0]


def ginv_mask_c1():
    """Indices in C^1 (basis (j, m)) with m ≡ j (mod 3)."""
    return [idx_c1(j, m) for j in range(N) for m in range(N) if (m - j) % ELL == 0]


def ginv_mask_c2():
    """Indices in C^2 (basis (j1, j2, m)) with m ≡ j1+j2 (mod 3)."""
    return [idx_c2(j1, j2, m)
            for j1 in range(N) for j2 in range(N) for m in range(N)
            if (m - j1 - j2) % ELL == 0]


def ginv_mask_c3():
    """Indices in C^3 (basis (j1, j2, j3, m)) with m ≡ j1+j2+j3 (mod 3)."""
    return [idx_c3(j1, j2, j3, m)
            for j1 in range(N) for j2 in range(N) for j3 in range(N)
            for m in range(N)
            if (m - j1 - j2 - j3) % ELL == 0]


# ============================================================
# Rank computation via SVD.
# ============================================================
def rank_complex(M, tol=None):
    """Numerical rank of a complex matrix via SVD."""
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    if len(s) == 0 or s[0] == 0:
        return 0
    if tol is None:
        tol = max(M.shape) * s[0] * 1e-12
    return int(np.sum(s > tol))


# ============================================================
# Approach (A):  Braided HH of B(V), then take G-invariants.
# ============================================================
def compute_braided_hh_bv():
    print("=" * 64)
    print("APPROACH (A): Braided HH of B(V) with B(V)-coefficients")
    print("=" * 64)
    d0 = build_d0()
    d1 = build_d1()
    d2 = build_d2()

    print(f"Chain dims: C^0={d0.shape[1]}, C^1={d0.shape[0]}={d1.shape[1]}, "
          f"C^2={d1.shape[0]}={d2.shape[1]}, C^3={d2.shape[0]}")

    # Sanity: d^1 ∘ d^0 = 0  and  d^2 ∘ d^1 = 0.
    err10 = np.linalg.norm(d1 @ d0)
    err21 = np.linalg.norm(d2 @ d1)
    print(f"||d^1 ∘ d^0||  = {err10:.2e}   (should be ~0)")
    print(f"||d^2 ∘ d^1||  = {err21:.2e}   (should be ~0)")

    r1 = rank_complex(d1)
    r2 = rank_complex(d2)
    print(f"rank(d^1) = {r1}")
    print(f"rank(d^2) = {r2}")
    print(f"dim ker(d^2) = {d2.shape[1] - r2}")
    print(f"dim im(d^1) = {r1}")
    dim_hh2_full = (d2.shape[1] - r2) - r1
    print(f"==> dim H^2_c(B(V)) [full] = {dim_hh2_full}")

    # G-invariant subspaces.
    mask0 = ginv_mask_c0()
    mask1 = ginv_mask_c1()
    mask2 = ginv_mask_c2()
    mask3 = ginv_mask_c3()
    print(f"\nG-invariant subspace dims: "
          f"(C^0)^G={len(mask0)}, (C^1)^G={len(mask1)}, "
          f"(C^2)^G={len(mask2)}, (C^3)^G={len(mask3)}")

    d1_ginv = d1[np.ix_(mask2, mask1)]
    d2_ginv = d2[np.ix_(mask3, mask2)]

    # Verify d^c preserves G-invariants (should hold by Negron Prop 3.3 (2)).
    # Quick check:  d^1 applied to a G-inv basis element should land in (C^2)^G.
    # We verify numerically by checking that the "off-block" pieces of d^1, d^2 are ~0.
    d1_off = np.delete(d1, mask2, axis=0)[:, mask1]
    d2_off = np.delete(d2, mask3, axis=0)[:, mask2]
    print(f"||d^1 off the G-inv block||  = {np.linalg.norm(d1_off):.2e}  (should be ~0)")
    print(f"||d^2 off the G-inv block||  = {np.linalg.norm(d2_off):.2e}  (should be ~0)")

    r1_g = rank_complex(d1_ginv)
    r2_g = rank_complex(d2_ginv)
    print(f"rank(d^1 |_(C^1)^G) = {r1_g}")
    print(f"rank(d^2 |_(C^2)^G) = {r2_g}")
    print(f"dim ker(d^2 |_(C^2)^G) = {len(mask2) - r2_g}")
    print(f"dim im(d^1 |_(C^1)^G) = {r1_g}")
    dim_hh2_ginv = (len(mask2) - r2_g) - r1_g
    print(f"==> dim H^2_c(B(V))^G  =  {dim_hh2_ginv}")
    return dim_hh2_full, dim_hh2_ginv


# ============================================================
# Approach (B):  Ordinary HH of B(V) with coefficients in B^+,
#                restricted to G-invariants.
# This is the *correct* Negron computation (Prop 4.2 + Cor 4.7):
#     C^•_{c,E}(B^+) = C^•(A, B^+) = Hom(A^{⊗•}, B^+),
#     HH^•(B^+) = H^•_{c,E}(B^+)^G = HH^•(B(V), B^+)^G.
# ============================================================

# B^+ = B(V) ⋊ C[G]  with basis  {E^i K^j : 0 ≤ i, j ≤ 2},  dim = 9.
# Multiplication:  (E^i K^j)(E^{i'} K^{j'}) = q^{2 j i'} E^{i+i'} K^{j+j'}
# (zero if i + i' ≥ 3).
DIM_BPLUS = N * N  # 9


def bplus_idx(i, j):
    """Index of E^i K^j in B^+."""
    return i * N + j


def bplus_mult(i, j, ip, jp):
    """(E^i K^j)(E^{i'} K^{j'}) = q^{2 j i'} E^{i+i'} K^{j+j'} (zero if i+i' ≥ 3)."""
    s_i = i + ip
    s_j = (j + jp) % ELL
    if s_i >= N:
        return None
    return (Q2 ** (j * ip), bplus_idx(s_i, s_j))


# Right adjoint G-action on B^+:  (E^i K^j) · K = S(K) (E^i K^j) K = K^{-1} (E^i K^j) K
#                                                  = q^{-2i} E^i K^j.
# (since K^{-1} E K = q^{-2} E, so K^{-1} E^i K = q^{-2i} E^i, and K^{-1} K^j K = K^j.)
# Thus  E^i K^j  has  G-weight  q^{-2i}  (independent of j) — equivalently
# the G-weight is indexed by i (mod 3) in the *additive* sense  -2 i ≡ i (mod 3)
# (since -2 ≡ 1 mod 3).  So  G-weight  =  q^{2 i}.

# Wait, let's double check:  -2 i mod 3  — if i=0, weight = 1; i=1, weight = q^{-2}=q;
# i=2, weight = q^{-4}=q^2.  And q^{2i} for i=0,1,2 gives 1, q^2, q.  These are
# complex conjugates.  The G-invariant subspace is weight 1, i.e., i ≡ 0 (mod 3).

# In B^+ the G-invariant (weight-1) subspace under the right adjoint action is
#     (B^+)^G = span{ E^0 K^j : j ∈ Z/3 } = span{1, K, K^2} = C[G].
# This has dim 3, NOT 9.  So the invariants pick out the C[G] subalgebra.
#
# But the chain groups C^n(B(V), B^+) = Hom(B(V)^{⊗n}, B^+) have the G-action
#     (f · K)(b_1, ..., b_n) = f(b_1 · K^{-1}, ..., b_n · K^{-1}) · K,
# where the G-action on B(V) (the source) is the FREE right action  E^i · K = q^{2i} E^i
# (NOT the adjoint action), and the G-action on B^+ (the target) is the ADJOINT action
# (E^i K^j) · K = q^{-2i} E^i K^j.  So for basis element  f_{j_1,...,j_n, (i, j)}:
#     f · K  has weight q^{2(-j_1 - ... - j_n - i)}  (source contributes -j's,
#                                                       target contributes -i).
# (In the multiplicative sense: f · K = q^{-2(j_1+...+j_n+i)} f.)
# So f is G-invariant iff  i + j_1 + ... + j_n ≡ 0 (mod 3).
# This gives dim  (C^n)^G = (3^n) * 3 = 3^{n+1}  (one j for each (j_1, ..., j_n, i)).
#
# Wait, that gives dim 9 for (C^1)^G.  Let me re-derive carefully.
#
# For n=1: basis {f_{j, (i, k)}} where f(E^j) = E^i K^k, dim 27.
#   G-action: (f · K)(E^j) = f(E^j · K^{-1}) · K = f(q^{-2j} E^j) · K
#                          = q^{-2j} f(E^j) · K
#                          = q^{-2j} (E^i K^k) · K
#                          = q^{-2j} q^{-2i} E^i K^k     (adjoint action on B^+)
#                          = q^{-2(j+i)} E^i K^k.
#   So f · K = q^{-2(j+i)} f.  G-invariant iff j + i ≡ 0 (mod 3).
#   For each (j, i) with j + i ≡ 0 (mod 3), and any k ∈ Z/3: dim = 3 * 3 = 9. ✓
#
# For n=2: basis {f_{j1, j2, (i, k)}} with dim 81.
#   f · K = q^{-2(j1 + j2 + i)} f.  G-inv iff j1 + j2 + i ≡ 0 (mod 3).
#   dim = 3^2 (for j1, j2) * 3 (for i = -(j1+j2) mod 3) * 3 (for k) = 27. ✓


def bplus_ginv_indices(n):
    """Return list of (source_indices, target_index) for G-invariant basis of
    C^n(B(V), B^+) = Hom(B(V)^{⊗n}, B^+).  G-invariant iff i + sum(j's) ≡ 0 mod 3."""
    # We construct the multi-index list explicitly.
    srcs = []
    if n == 0:
        srcs = [()]
    elif n == 1:
        srcs = [(j,) for j in range(N)]
    elif n == 2:
        srcs = [(j1, j2) for j1 in range(N) for j2 in range(N)]
    elif n == 3:
        srcs = [(j1, j2, j3)
                for j1 in range(N) for j2 in range(N) for j3 in range(N)]
    else:
        raise ValueError(f"n={n} not supported")

    inv = []
    for src in srcs:
        s = sum(src) % ELL
        for i in range(N):
            if (i + s) % ELL != 0:
                continue
            for k in range(N):
                inv.append((src, bplus_idx(i, k)))
    return inv


def build_ordinary_d1_bplus_ginv():
    """Build the G-invariant block of d^1: C^1(B(V), B^+) → C^2(B(V), B^+).
    This is the ordinary Hochschild differential restricted to G-invariants.
    (d^1 f)(b_0, b_1) = b_0 · f(b_1) - f(b_0 b_1) + f(b_0) · b_1.
    Here the multiplications are the ordinary multiplications in B(V) (for b_0 b_1)
    and in B^+ (for b_0 · f(b_1)  and  f(b_0) · b_1, viewing B(V) ⊂ B^+ as E^i K^0).

    Wait — the source B(V)^{⊗n} uses the Nichols algebra B(V); the coefficient
    B^+ is the bosonization.  We must be careful with how B(V) multiplies into B^+.
    The natural inclusion  B(V) ↪ B^+  sends  E^i ↦ E^i K^0,  and the multiplication
    in B^+ is  (E^i K^j)(E^{i'} K^{j'}) = q^{2 j i'} E^{i+i'} K^{j+j'}.  So
    E^i K^0 * E^{i'} K^0 = E^{i+i'} K^0  (with E^{i+i'} = 0 if i+i' ≥ 3), which is
    the same as the B(V) multiplication.  Good — B(V) is a subalgebra of B^+.
    """
    rows = bplus_ginv_indices(2)
    cols = bplus_ginv_indices(1)
    M = np.zeros((len(rows), len(cols)), dtype=complex)
    row_idx = {r: i for i, r in enumerate(rows)}
    col_idx = {c: i for i, c in enumerate(cols)}

    for ci, (src1, t1) in enumerate(cols):
        # t1 = bplus_idx(i1, k1) — f(E^{src1[0]}) = E^{i1} K^{k1}.
        i1 = t1 // N
        k1 = t1 % N
        j_src1 = src1[0]
        for ri, (src2, t2) in enumerate(rows):
            j0, j1 = src2
            # (d^1 f)(E^{j0}, E^{j1}) = E^{j0} · f(E^{j1}) - f(E^{j0} E^{j1}) + f(E^{j0}) · E^{j1}
            # Term 1: E^{j0} K^0 · f(E^{j1})  — nonzero iff j1 == j_src1, value
            #   (E^{j0} K^0)(E^{i1} K^{k1}) = q^{2*0*i1} E^{j0+i1} K^{k1} = E^{j0+i1} K^{k1}
            #   (zero if j0 + i1 ≥ 3).
            if j1 == j_src1:
                s = j0 + i1
                if s < N:
                    t_new = bplus_idx(s, k1)
                    key = ((j0, j1), t_new)
                    if key in row_idx:
                        M[row_idx[key], ci] += 1.0
            # Term 2: -f(E^{j0} E^{j1}) — nonzero iff j0+j1 < 3 and j0+j1 == j_src1,
            #   value -E^{i1} K^{k1}.
            s01 = j0 + j1
            if s01 < N and s01 == j_src1:
                key = ((j0, j1), t1)
                if key in row_idx:
                    M[row_idx[key], ci] += -1.0
            # Term 3: f(E^{j0}) · E^{j1}  — nonzero iff j0 == j_src1, value
            #   (E^{i1} K^{k1})(E^{j1} K^0) = q^{2 k1 j1} E^{i1+j1} K^{k1}
            #   (zero if i1 + j1 ≥ 3).
            if j0 == j_src1:
                s = i1 + j1
                if s < N:
                    coef = Q2 ** (k1 * j1)
                    t_new = bplus_idx(s, k1)
                    key = ((j0, j1), t_new)
                    if key in row_idx:
                        M[row_idx[key], ci] += coef
    return M


def build_ordinary_d2_bplus_ginv():
    """Build the G-invariant block of d^2: C^2(B(V), B^+) → C^3(B(V), B^+).
    (d^2 f)(b_0, b_1, b_2) = b_0 · f(b_1, b_2) - f(b_0 b_1, b_2) + f(b_0, b_1 b_2)
                              - f(b_0, b_1) · b_2.
    """
    rows = bplus_ginv_indices(3)
    cols = bplus_ginv_indices(2)
    M = np.zeros((len(rows), len(cols)), dtype=complex)
    row_idx = {r: i for i, r in enumerate(rows)}
    col_idx = {c: i for i, c in enumerate(cols)}

    for ci, (src2, t2) in enumerate(cols):
        i2 = t2 // N
        k2 = t2 % N
        j_src2_a, j_src2_b = src2
        for ri, (src3, t3) in enumerate(rows):
            j0, j1, j2 = src3
            # Term 1: b_0 · f(b_1, b_2) — nonzero iff (j1, j2) == (j_src2_a, j_src2_b),
            #   value (E^{j0} K^0)(E^{i2} K^{k2}) = E^{j0+i2} K^{k2}  (zero if j0+i2 ≥ 3).
            if j1 == j_src2_a and j2 == j_src2_b:
                s = j0 + i2
                if s < N:
                    t_new = bplus_idx(s, k2)
                    key = ((j0, j1, j2), t_new)
                    if key in row_idx:
                        M[row_idx[key], ci] += 1.0
            # Term 2: -f(b_0 b_1, b_2) — nonzero iff j0+j1 < 3, (j0+j1, j2) == (j_src2_a, j_src2_b),
            #   value -E^{i2} K^{k2}.
            s01 = j0 + j1
            if s01 < N and s01 == j_src2_a and j2 == j_src2_b:
                key = ((j0, j1, j2), t2)
                if key in row_idx:
                    M[row_idx[key], ci] += -1.0
            # Term 3: f(b_0, b_1 b_2) — nonzero iff j1+j2 < 3, (j0, j1+j2) == (j_src2_a, j_src2_b),
            #   value +E^{i2} K^{k2}.
            s12 = j1 + j2
            if s12 < N and j0 == j_src2_a and s12 == j_src2_b:
                key = ((j0, j1, j2), t2)
                if key in row_idx:
                    M[row_idx[key], ci] += 1.0
            # Term 4: -f(b_0, b_1) · b_2 — nonzero iff (j0, j1) == (j_src2_a, j_src2_b),
            #   value -(E^{i2} K^{k2})(E^{j2} K^0) = -q^{2 k2 j2} E^{i2+j2} K^{k2}
            #   (zero if i2+j2 ≥ 3).
            if j0 == j_src2_a and j1 == j_src2_b:
                s = i2 + j2
                if s < N:
                    coef = -Q2 ** (k2 * j2)
                    t_new = bplus_idx(s, k2)
                    key = ((j0, j1, j2), t_new)
                    if key in row_idx:
                        M[row_idx[key], ci] += coef
    return M


def compute_ordinary_hh_bv_bplus():
    print("\n" + "=" * 64)
    print("APPROACH (B): Ordinary HH of B(V) with B^+-coefficients, then G-inv")
    print("  (= the CORRECT Negron computation: Prop 4.2 + Cor 4.7)")
    print("=" * 64)
    d1 = build_ordinary_d1_bplus_ginv()
    d2 = build_ordinary_d2_bplus_ginv()

    print(f"G-inv chain dims: (C^1)^G={d1.shape[1]}, (C^2)^G={d1.shape[0]}={d2.shape[1]}, "
          f"(C^3)^G={d2.shape[0]}")

    err21 = np.linalg.norm(d2 @ d1)
    print(f"||d^2 ∘ d^1||  = {err21:.2e}   (should be ~0)")

    r1 = rank_complex(d1)
    r2 = rank_complex(d2)
    print(f"rank(d^1 |_(C^1)^G) = {r1}")
    print(f"rank(d^2 |_(C^2)^G) = {r2}")
    print(f"dim ker(d^2 |_(C^2)^G) = {d2.shape[1] - r2}")
    print(f"dim im(d^1 |_(C^1)^G) = {r1}")
    dim_hh2 = (d2.shape[1] - r2) - r1
    print(f"==> dim HH^2(B(V), B^+)^G  =  dim HH^2(B^+)  =  {dim_hh2}")
    return dim_hh2


# ============================================================
# Main driver.
# ============================================================
def main():
    print("Braided Hochschild cohomology of B(V) for sl_2 at ℓ = 3")
    print(f"q = e^(2πi/3) = {Q:.6f},  q^2 = {Q2:.6f},  1 + q + q^2 = {1 + Q + Q2:.2e}")
    print(f"B(V) = C[E]/(E^3),  dim = {N}")
    print(f"B^+ = B(V) ⋊ C[G],  dim = {DIM_BPLUS}")
    print(f"|G| = {ELL}")
    print()
    t0 = time.time()
    hhA_full, hhA_ginv = compute_braided_hh_bv()
    hhB = compute_ordinary_hh_bv_bplus()
    t1 = time.time()
    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"(A) dim H^2_c(B(V))       [full]            = {hhA_full}")
    print(f"(A) dim H^2_c(B(V)^G)     [G-invariant]     = {hhA_ginv}")
    print(f"(B) dim HH^2(B(V), B^+)^G [Negron's method] = {hhB}")
    print()
    print(f"Known: dim HH^2(B^+(sl_2), C) = 1   (paper §6, verify_bplus_sl2_hh2.py)")
    print()
    if hhB == 1:
        print(f"✓ Approach (B) MATCHES the known answer 1.")
        print(f"  This validates the Negron braided-HH method.")
    else:
        print(f"✗ Approach (B) does NOT match the known answer.")
    print()
    if hhA_ginv == 1:
        print(f"✓ Approach (A) ALSO matches 1 — the task's literal shortcut works!")
    else:
        print(f"⚠ Approach (A) gives {hhA_ginv}, NOT 1 — the task's literal shortcut")
        print(f"  (braided HH of B(V) alone, then G-invariants) does NOT compute")
        print(f"  HH^2(B^+) correctly.  The correct shortcut is Approach (B), which")
        print(f"  uses the *relative* braided complex (= ordinary HH of B(V) with")
        print(f"  B^+-coefficients) per Negron's Prop 4.2 + Cor 4.7.")
    print()
    print(f"Elapsed: {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
