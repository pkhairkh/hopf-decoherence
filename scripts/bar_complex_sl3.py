#!/usr/bin/env python3
"""
Bar complex on PBW normal forms for u_q(sl_3) at ell = 3.

Task W3-1a of the AST/IR research programme.

The Anick d_3 is buggy (see W2-1a-IR) and cannot compute HH^2(u_q(sl_3), C).
This script attempts the bar complex directly on PBW normal forms using the
IR-built multiplication table, with aggressive weight-space decomposition.

CONTEXT
-------
- dim u_q(sl_3) at ell = 3 = 3^8 = 6561.
- PBW basis: K1^a K2^b E1^c E12^e E2^d F1^f F21^g F2^h, 0 <= a,b,c,e,d,f,g,h <= 2.
- Weight decomposition at ell = 3: Cartan matrix of sl_3 has determinant 3,
  so its reduction mod 3 has rank 1. This collapses the (Z/3)^2 weight lattice
  to a Z/3 lattice (only 3 distinct weights, each of dim 2187).
  - Weight of K1^a K2^b E1^c E12^e E2^d F1^f F21^g F2^h is the K-eigenvalue
    (s, s) where s = (2c + e + 2d + f + 2g + h) mod 3.
  - Three weight spaces: wt=0, wt=1, wt=2, each of dim 2187.

BAR COMPLEX DIMENSIONS
----------------------
- Full algebra: dim C^1 = 6561, dim C^2 = 6561^2 = 4.3 * 10^7,
  dim C^3 = 6561^3 = 2.8 * 10^11.
- Per weight block (3 blocks): dim C^1_w = 2187, dim C^2_w = 3 * 2187^2 = 1.4 * 10^7,
  dim C^3_w = 9 * 2187^3 = 9.4 * 10^10.
- Gram matrix of d^2 per weight block: 1.4 * 10^7 squared = 2 * 10^14 entries,
  ~300 TB as complex doubles. INTRACTABLE.

WHAT THIS SCRIPT COMPUTES
-------------------------
1. dim HH^1(u_q(sl_3), C) via weight-decomposed Gram matrix of d^1.
   - Gram matrix of d^1 per weight block: 2187 x 2187 = ~75 MB dense complex.
   - Tractable! Verifies the Anick-resolution result HH^1 = 1 (W2-1a-IR).

2. dim HH^2(B^+(u_q(sl_3)), C) as a sanity check on the multiplication table.
   - B^+ has dim 3^5 = 243, weight-decomposed into 3 blocks of 81.
   - Per weight block: dim C^2 = 3 * 81^2 = 19683, dim C^3 = 9 * 81^3 = 4.8M.
   - Gram matrix of d^2: 19683 x 19683 = ~3 GB dense, tractable with sparse.
   - Expected: dim HH^2(B^+) = 5 (verified W2-1b, paper Sec. 6.5).

3. Documentation of the intractability of dim HH^2(u_q(sl_3), C).

The script uses the IR framework's NormalFormReducer (W2-1a) to build the
multiplication table as 8 sparse "left/right multiplication by generator"
matrices, then composes these for arbitrary products.
"""
from __future__ import annotations

import cmath
import itertools
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ir.uq_sl3 import (
    build_uq_sl3_presentation,
    pbw_basis,
    GEN_NAMES,
)
from ir.parser import NormalFormReducer, Monomial, Polynomial, QLaurent
from ir.qomega import QOmega3


# ============================================================================
# Constants
# ============================================================================

ELL = 3
DIM = 6561  # 3^8
Q = cmath.exp(2j * math.pi / ELL)  # omega = e^{2*pi*i/3}
Q_INV = Q ** (-1)  # = omega^2 = -1 - omega
ALPHA = 1.0 / (Q - Q_INV)  # 1/(q - q^{-1}) = 1/(omega - omega^2) = 1/(1 + 2*omega) ...

# Generator indices
K1, K2, E1, E12, E2, F1, F21, F2 = range(8)

# Weight function: weight of K1^a K2^b E1^c E12^e E2^d F1^f F21^g F2^h is
# s = (2c + e + 2d + f + 2g + h) mod 3, giving weight (s, s) in (Z/3)^2.
# At ell=3, the Cartan matrix of sl_3 mod 3 has rank 1 (det = 3 = 0 mod 3),
# so the two K-eigenvalues coincide: weight_K1 = weight_K2 = s.
#
# Pairings <alpha_i^vee, alpha>:
#   E1 (alpha_1):  K1 -> q^2, K2 -> q^{-1} = q^2 (at ell=3). Weight 2.
#   E2 (alpha_2):  K1 -> q^{-1} = q^2, K2 -> q^2. Weight 2.
#   E12 (alpha_1+alpha_2): K1 -> q^1, K2 -> q^1. Weight 1.
#   F1 (-alpha_1): K1 -> q^{-2} = q, K2 -> q^1. Weight 1.
#   F2 (-alpha_2): K1 -> q^1, K2 -> q^{-2} = q. Weight 1.
#   F21 (-(alpha_1+alpha_2)): K1 -> q^{-1} = q^2, K2 -> q^{-1} = q^2. Weight 2.
#   K1, K2: weight 0.
# So weight = (2c + 1*e + 2*d + 1*f + 2*g + 1*h) mod 3
#          = (2c + e + 2d + f + 2g + h) mod 3.


def from_idx(idx: int) -> tuple:
    """Decode PBW index i into (a, b, c, e, d, f, g, h)."""
    exps = [0] * 8
    for k in range(7, -1, -1):
        exps[k] = idx % ELL
        idx //= ELL
    return tuple(exps)


def to_idx(exps: tuple) -> int:
    """Encode (a, b, c, e, d, f, g, h) into PBW index."""
    idx = 0
    for e in exps:
        idx = idx * ELL + e
    return idx


def weight(idx: int) -> int:
    """Weight (single integer in Z/3) of the PBW element at `idx`."""
    a, b, c, e, d, f, g, h = from_idx(idx)
    return (2 * c + e + 2 * d + f + 2 * g + h) % ELL


def decompose_to_gens(idx: int) -> tuple:
    """Decompose PBW element at `idx` into a sequence of generator indices."""
    exps = from_idx(idx)
    gens = []
    for k in range(8):
        gens.extend([k] * exps[k])
    return tuple(gens)


# ============================================================================
# Coefficient conversion
# ============================================================================


def coeff_to_complex(c) -> complex:
    """Convert a QOmega3 / QLaurent / int / Fraction to a complex number."""
    if isinstance(c, QOmega3):
        return c.to_complex()
    if isinstance(c, QLaurent):
        total = 0.0 + 0.0j
        for e, coef in c.terms.items():
            total += float(coef) * (Q ** e)
        return total
    return complex(c)


def poly_to_dict(p: Polynomial) -> dict:
    """Convert a Polynomial (in PBW normal form) to {idx: complex_coeff}.

    Each monomial's generator counts are taken mod 3 for K1, K2 (since K_i^3 = 1).
    """
    d = {}
    for t in p.terms:
        cc = coeff_to_complex(t.coeff)
        exps = [0] * 8
        for g in t.monomial.gens:
            exps[g] += 1
        # K1^3 = 1, K2^3 = 1, so reduce Cartan exponents mod 3.
        exps[0] %= ELL
        exps[1] %= ELL
        idx = to_idx(tuple(exps))
        if idx in d:
            d[idx] += cc
        else:
            d[idx] = cc
    return {k: v for k, v in d.items() if abs(v) > 1e-13}


# ============================================================================
# Build L_g and R_g sparse multiplication tables
# ============================================================================


def build_mult_tables(verbose: bool = True) -> tuple:
    """Build left- and right-multiplication-by-generator sparse matrices.

    Returns (L_mat, R_mat) where each is a list of 8 scipy.sparse.csr_matrix
    of shape (6561, 6561). L_mat[g] @ v gives g * v (left mult by g);
    R_mat[g] @ v gives v * g (right mult by g).
    """
    # Use separate caches for L and R tables (built incrementally).
    L_cache = Path("/tmp/L_sl3_tables.pkl")
    R_cache = Path("/tmp/R_sl3_tables.pkl")

    pres = None
    reducer = None
    basis = None

    if L_cache.exists():
        with open(L_cache, "rb") as f:
            L = pickle.load(f)
        if verbose:
            print(f"  Loaded cached L_g tables from {L_cache}")
    else:
        if verbose:
            print("  Building IR presentation and reducer...")
        pres = build_uq_sl3_presentation()
        reducer = NormalFormReducer(pres)
        basis = pbw_basis()
        if verbose:
            print("  Building L_g tables (52K reductions)...")
        L = [dict() for _ in range(8)]
        t0 = time.time()
        for g in range(8):
            for i in range(DIM):
                mono_L = Monomial((g,) + basis[i].gens)
                nf_L = reducer.normal_form(mono_L)
                L[g][i] = poly_to_dict(nf_L)
            if verbose:
                print(f"    L_{GEN_NAMES[g]} done ({time.time() - t0:.1f}s total)")
        with open(L_cache, "wb") as f:
            pickle.dump(L, f)
        if verbose:
            print(f"  Cached L_g tables to {L_cache}")

    if R_cache.exists():
        with open(R_cache, "rb") as f:
            R = pickle.load(f)
        if verbose:
            print(f"  Loaded cached R_g tables from {R_cache}")
    else:
        if verbose:
            print("  Building R_g tables (52K reductions)...")
        if pres is None:
            pres = build_uq_sl3_presentation()
            reducer = NormalFormReducer(pres)
            basis = pbw_basis()
        R = [dict() for _ in range(8)]
        t0 = time.time()
        for g in range(8):
            for i in range(DIM):
                mono_R = Monomial(basis[i].gens + (g,))
                nf_R = reducer.normal_form(mono_R)
                R[g][i] = poly_to_dict(nf_R)
            if verbose:
                print(f"    R_{GEN_NAMES[g]} done ({time.time() - t0:.1f}s total)")
        with open(R_cache, "wb") as f:
            pickle.dump(R, f)
        if verbose:
            print(f"  Cached R_g tables to {R_cache}")

    # Convert dicts to scipy sparse matrices
    def to_sparse(table):
        rows, cols, vals = [], [], []
        for i, d in table.items():
            for j, v in d.items():
                rows.append(j)
                cols.append(i)
                vals.append(v)
        return sparse.csr_matrix(
            (vals, (rows, cols)), shape=(DIM, DIM), dtype=complex
        )

    L_mat = [to_sparse(L[g]) for g in range(8)]
    R_mat = [to_sparse(R[g]) for g in range(8)]
    return L_mat, R_mat


# ============================================================================
# Multiplication
# ============================================================================


def multiply(a_idx: int, b_idx: int, R_mat) -> dict:
    """Compute a_{a_idx} * a_{b_idx} as a dict {idx: complex_coeff}.

    Uses right-multiplication by generators: a * b = R_{gk}(... R_{g1}(a)...)
    where b = g1 g2 ... gk.
    """
    v = np.zeros(DIM, dtype=complex)
    v[a_idx] = 1.0
    for g in decompose_to_gens(b_idx):
        v = R_mat[g] @ v
        v[np.abs(v) < 1e-13] = 0.0
    return {i: v[i] for i in range(DIM) if abs(v[i]) > 1e-13}


def multiply_batch(a_indices: list, b_idx: int, R_mat) -> sparse.csr_matrix:
    """Compute {a_j * a_{b_idx} : j in a_indices} as columns of a sparse matrix.

    Returns a sparse matrix V of shape (DIM, len(a_indices)) where column j is
    the polynomial a_{a_indices[j]} * a_{b_idx} in PBW form.
    """
    n = len(a_indices)
    V = sparse.lil_matrix((DIM, n), dtype=complex)
    for col, a in enumerate(a_indices):
        V[a, col] = 1.0
    V = V.tocsr()
    for g in decompose_to_gens(b_idx):
        V = R_mat[g] @ V
        V.data[np.abs(V.data) < 1e-13] = 0.0
        V.eliminate_zeros()
    return V.tocsr()


# ============================================================================
# Sanity checks
# ============================================================================


def sanity_checks(L_mat, R_mat, verbose: bool = True) -> dict:
    """Verify algebra relations."""
    if verbose:
        print("  Sanity checks:")
    results = {}

    # 1 * x = x * 1 = x (1 is the empty PBW element, idx 0)
    e0 = 0  # K1^0 K2^0 E1^0 ... = identity
    ok_unit_l = True
    ok_unit_r = True
    for i in range(0, DIM, 700):  # sample
        v = np.zeros(DIM, dtype=complex)
        v[i] = 1.0
        # L_K1[0] is 1 * K1^0 = K1, not 1. Need to use 1 itself.
        # Actually: 1 = empty PBW = idx 0. So L_g[0] = g, R_g[0] = g.
        # But we want: 1 * x = x. That means L_{idx 0} applied to x gives x.
        # There's no L_{idx 0} matrix; the identity PBW element is "no generators".
        # So multiplying by 1 = no generators means no applications.
        pass
    # Just check: K1^3 = 1, E1^3 = 0, etc.
    # K1 idx = to_idx((1,0,0,0,0,0,0,0)) = 1*3^7 = 2187
    K1_idx = to_idx((1, 0, 0, 0, 0, 0, 0, 0))
    K1_2 = to_idx((2, 0, 0, 0, 0, 0, 0, 0))
    e0_idx = 0

    # K1 * K1 = K1^2
    p = multiply(K1_idx, K1_idx, R_mat)
    is_k1sq = (len(p) == 1 and list(p.keys())[0] == K1_2
               and abs(list(p.values())[0] - 1.0) < 1e-12)
    results["K1 * K1 = K1^2"] = is_k1sq
    if verbose:
        print(f"    K1 * K1 = K1^2: {is_k1sq}")

    # K1^3 = 1
    p = multiply(K1_2, K1_idx, R_mat)
    is_one = (len(p) == 1 and list(p.keys())[0] == e0_idx
              and abs(list(p.values())[0] - 1.0) < 1e-12)
    results["K1^3 = 1"] = is_one
    if verbose:
        print(f"    K1^3 = 1: {is_one}")

    # E1^3 = 0
    E1_idx = to_idx((0, 0, 1, 0, 0, 0, 0, 0))
    E1_2 = to_idx((0, 0, 2, 0, 0, 0, 0, 0))
    p = multiply(E1_2, E1_idx, R_mat)
    is_zero = (len(p) == 0)
    results["E1^3 = 0"] = is_zero
    if verbose:
        print(f"    E1^3 = 0: {is_zero}")

    # F1^3 = 0
    F1_idx = to_idx((0, 0, 0, 0, 0, 1, 0, 0))
    F1_2 = to_idx((0, 0, 0, 0, 0, 2, 0, 0))
    p = multiply(F1_2, F1_idx, R_mat)
    is_zero = (len(p) == 0)
    results["F1^3 = 0"] = is_zero
    if verbose:
        print(f"    F1^3 = 0: {is_zero}")

    # E12^3 = 0
    E12_idx = to_idx((0, 0, 0, 1, 0, 0, 0, 0))
    E12_2 = to_idx((0, 0, 0, 2, 0, 0, 0, 0))
    p = multiply(E12_2, E12_idx, R_mat)
    is_zero = (len(p) == 0)
    results["E12^3 = 0"] = is_zero
    if verbose:
        print(f"    E12^3 = 0: {is_zero}")

    # E1 * K1 = q^{-2} K1 E1 = q K1 E1 (rule R10: E1 K1 -> q K1 E1)
    # In PBW form, K1 E1 is index (1, 0, 1, 0, 0, 0, 0, 0) = 2187 + 243 = 2430
    p = multiply(E1_idx, K1_idx, R_mat)
    expected_idx = to_idx((1, 0, 1, 0, 0, 0, 0, 0))
    expected_coeff = Q  # q^{-2} = q at ell=3
    is_q = (len(p) == 1 and list(p.keys())[0] == expected_idx
            and abs(list(p.values())[0] - expected_coeff) < 1e-12)
    results["E1 * K1 = q K1 E1"] = is_q
    if verbose:
        print(f"    E1 * K1 = q K1 E1: {is_q}")

    # F1 * E1 = E1 F1 - alpha*(K1 - K1^2)
    p = multiply(F1_idx, E1_idx, R_mat)
    # E1 F1 is idx (0, 0, 1, 0, 0, 1, 0, 0) = 243 + 9 = 252
    # K1 is idx 2187, K1^2 is idx 4374
    expected = {
        252: 1.0,
        2187: -ALPHA,
        4374: ALPHA,
    }
    is_commutator = all(
        abs(p.get(k, 0) - v) < 1e-12 for k, v in expected.items()
    ) and all(abs(v) < 1e-12 for k, v in p.items() if k not in expected)
    results["F1 * E1 = E1 F1 - alpha(K1 - K1^2)"] = is_commutator
    if verbose:
        print(f"    F1 * E1 = E1 F1 - alpha(K1 - K1^2): {is_commutator}")

    return results


# ============================================================================
# HH^1 via weight-decomposed Gram matrix of d^1
# ============================================================================
#
# Bar differential d^1: C^1 = Hom(A, k) -> C^2 = Hom(A ⊗ A, k).
# For trivial coefficients k:
#   (d^1 f)(a, b) = ε(a) f(b) - f(a*b) + f(a) ε(b)
# where ε(K1^a K2^b ...) = 1 iff all E/F exponents are 0.
#
# In matrix form (rows (j,k), columns i):
#   d^1[(j,k), i] = ε(a_j) δ_{ik} - [coeff of a_i in a_j * a_k] + δ_{ij} ε(a_k)
#
# dim C^1 = 6561, dim C^2 = 6561^2 = 43M.
# Weight decomposition: 3 weights, dim C^1_w = 2187, dim C^2_w = 3 * 2187^2 = 14.3M.
# Per weight block: d^1 is 14.3M x 2187, Gram matrix is 2187 x 2187 (small!).
#
# We compute the Gram matrix d^1^H d^1 incrementally, by streaming over (j, k)
# pairs with wt(j) + wt(k) = w.


def compute_hh1(L_mat, R_mat, verbose: bool = True) -> dict:
    """Compute dim HH^1(u_q(sl_3), C) via weight-decomposed Gram matrix."""
    if verbose:
        print("\n=== HH^1 via bar complex (weight-decomposed) ===")

    # Build weight index maps
    wt_of = np.array([weight(i) for i in range(DIM)], dtype=int)
    # For each weight s in {0, 1, 2}, indices of PBW elements with weight s.
    wt_indices = {s: np.where(wt_of == s)[0] for s in range(ELL)}
    if verbose:
        print(f"  Weight-space sizes: {[len(wt_indices[s]) for s in range(ELL)]}")
        print(f"    (Expected: 2187 per weight, total 6561)")

    # Counit: ε(idx) = 1 iff all E/F exponents are 0, i.e., wt = 0 (since K has wt 0)
    # AND exponents c, e, d, f, g, h are all 0.
    epsilon = np.zeros(DIM, dtype=complex)
    for i in range(DIM):
        a, b, c, e, d, f, g, h = from_idx(i)
        if c == 0 and e == 0 and d == 0 and f == 0 and g == 0 and h == 0:
            epsilon[i] = 1.0
    if verbose:
        print(f"  ε-nonzero elements: {int(np.sum(np.abs(epsilon) > 0))} (expected 9)")

    # For each weight block w (the weight of the COCHAIN, i.e., the column index i
    # has wt(i) = w, and the row (j, k) has wt(j) + wt(k) = w):
    total_rank_d1 = 0
    block_results = {}
    t_start = time.time()

    for w in range(ELL):
        if verbose:
            print(f"\n  Weight block w = {w}:")
            print(f"    dim C^1_w = {len(wt_indices[w])}")

        # C^1_w indices (columns of d^1, indexed by i)
        c1_w = wt_indices[w]
        n_c1 = len(c1_w)
        # Map global index to local index in C^1_w
        c1_w_set = set(int(x) for x in c1_w)
        c1_w_map = {int(x): i for i, x in enumerate(c1_w)}

        # Gram matrix: n_c1 x n_c1 complex
        Gram = np.zeros((n_c1, n_c1), dtype=complex)

        # Accumulate Gram = d^1^H d^1 by streaming over (j, k) pairs with
        # wt(j) + wt(k) = w.
        #
        # d^1[(j,k), i] = ε(j) δ_{ik} - coeff(a_i in a_j * a_k) + δ_{ij} ε(k)
        #
        # Gram[i, i'] = sum_{(j,k): wt(j)+wt(k)=w} d^1[(j,k),i] * conj(d^1[(j,k),i'])
        #
        # The sum has 3 weight-pair contributions: (s, w-s) for s in {0, 1, 2}.
        # For each (s, w-s): |wt=s| * |wt=w-s| = 2187 * 2187 = 4.8M pairs.

        # Strategy: iterate over k (right factor). For each k with wt(k) = w-s,
        # compute all products {a_j * a_k : j with wt(j) = s} as a sparse matrix
        # (using R_mat), then accumulate into Gram.

        t_block_start = time.time()
        n_pairs_total = 0
        n_k_processed = 0

        for s in range(ELL):
            s2 = (w - s) % ELL
            c1_s = wt_indices[s]  # j's with wt = s
            c1_s2 = wt_indices[s2]  # k's with wt = s2
            n_pairs_total += len(c1_s) * len(c1_s2)

            if verbose:
                print(f"    weight pair (s={s}, s2={s2}): "
                      f"{len(c1_s)} x {len(c1_s2)} = {len(c1_s)*len(c1_s2)} pairs")

            # Pre-compute the row-restriction matrix: rows in c1_w of the full identity.
            # We'll use this to extract V[c1_w, :] efficiently.
            # Build a sparse selector matrix: S_w (n_c1 x DIM) where S_w[i, j] = 1 iff j = c1_w[i].
            S_w = sparse.csr_matrix(
                (np.ones(n_c1, dtype=complex),
                 (np.arange(n_c1), c1_w)),
                shape=(n_c1, DIM),
                dtype=complex,
            )

            # For each k in c1_s2:
            for k_idx in c1_s2:
                k_idx = int(k_idx)
                # Compute all products a_j * a_k for j in c1_s
                V = multiply_batch(list(c1_s), k_idx, R_mat)  # sparse, DIM x |c1_s|
                # Restrict to rows in c1_w: V_restricted has shape (n_c1, |c1_s|)
                V_restricted = S_w @ V  # sparse, n_c1 x |c1_s|
                V_restricted = V_restricted.tocsr()

                # Main term contribution: V_restricted @ V_restricted^H  (n_c1 x n_c1)
                # This is the sum over j_local of v_{j_local, i} * conj(v_{j_local, i'})
                # = (V_restricted @ V_restricted.conj().T)[i, i']
                contrib = (V_restricted @ V_restricted.conj().T).toarray()
                Gram += contrib

                # Boundary terms:
                # Term 1 x Term 2: -ε(j) δ_{ik} * conj(coeff(a_{i'} in p))
                #   For each j_global with ε(j) ≠ 0 (Cartan), and k_idx in c1_w_set:
                #   Gram[k_loc, l_loc] += -ε(j) * conj(v_{j_local, l})
                #   where v_{j_local, l} = coeff(a_l in a_j × a_k).
                #   Since ε(j) ≠ 0 means j is Cartan (wt=0), we need s = 0 (so j has wt = 0).
                if s == 0 and k_idx in c1_w_set:
                    k_loc = c1_w_map[k_idx]
                    # For each j in c1_s with ε(j) ≠ 0:
                    for j_global in c1_s:
                        j_global = int(j_global)
                        if abs(epsilon[j_global]) > 1e-13:
                            # Find the column in V_restricted corresponding to j_global.
                            # The columns of V_restricted are indexed by j_local, which
                            # corresponds to c1_s[j_local].
                            # Find j_local such that c1_s[j_local] == j_global.
                            # Since c1_s is a numpy array, we need a map.
                            # For efficiency, precompute c1_s_map outside the k loop.
                            pass  # We'll handle this below.

                # Term 3 x Term 2: -ε(k) δ_{ij} * conj(coeff(a_{i'} in p))
                #   For each k with ε(k) ≠ 0 (Cartan), and j_global in c1_w_set:
                #   Gram[j_loc, l_loc] += -ε(k) * conj(v_{j_local, l})
                #   Since ε(k) ≠ 0 means k is Cartan (wt=0), we need s2 = 0.
                if s2 == 0 and len(c1_s) > 0:
                    # For each j in c1_s with j_global in c1_w_set:
                    # (Note: c1_s is the set of j's with wt = s. For j to be in c1_w_set,
                    # we need s = w.)
                    if s == w:
                        # For each j_local, j_global = c1_s[j_local].
                        # If j_global in c1_w_set: update Gram[j_loc, l_loc] for each (l, v).
                        # Gram[j_loc, l_loc] += -ε(k) * conj(v_{j_local, l_loc})
                        # = -ε(k) * conj(V_restricted[l_loc, j_local])
                        # So Gram[:, j_loc] += -ε(k) * conj(V_restricted[:, j_local])
                        # We can vectorize: for each j_local in range(|c1_s|):
                        #   j_global = c1_s[j_local]
                        #   if j_global in c1_w_set:
                        #     Gram[:, j_loc] += -ε(k) * conj(V_restricted[:, j_local])
                        # But also Term 2 x Term 3 (Hermitian conjugate):
                        #   Gram[l_loc, j_loc] += -ε(k) * V_restricted[l_loc, j_local]
                        # Which is just the Hermitian conjugate.
                        for j_local in range(len(c1_s)):
                            j_global = int(c1_s[j_local])
                            if j_global in c1_w_set:
                                j_loc = c1_w_map[j_global]
                                # Get column j_local of V_restricted
                                col = V_restricted[:, j_local].toarray().flatten()
                                # Gram[:, j_loc] += -ε(k) * conj(col)
                                Gram[:, j_loc] += -epsilon[k_idx] * np.conj(col)
                                # Gram[j_loc, :] += -ε(k) * col  (Hermitian conjugate)
                                Gram[j_loc, :] += -epsilon[k_idx] * col
                                # Term 3 x Term 3: ε(k)^2 δ_{ij} δ_{i'j} = ε(k)^2 (i = i' = j_global)
                                Gram[j_loc, j_loc] += abs(epsilon[k_idx]) ** 2

                # Term 1 x Term 1: ε(j)^2 δ_{ik} δ_{i'k} = ε(j)^2 (i = i' = k_idx)
                #   For each (j, k) with ε(j) ≠ 0 and k_idx in c1_w_set:
                #   Gram[k_loc, k_loc] += ε(j)^2
                #   ε(j) ≠ 0 means j is Cartan, so s = 0 (wt(j) = 0).
                if s == 0 and k_idx in c1_w_set:
                    k_loc = c1_w_map[k_idx]
                    n_cartan_j = int(np.sum(np.abs(epsilon[c1_s]) > 1e-13))
                    Gram[k_loc, k_loc] += n_cartan_j  # ε(j)^2 = 1 for each Cartan j

                # Term 1 x Term 2 and Term 2 x Term 1: -ε(j) δ_{ik} * conj(coeff(a_{i'} in p))
                #   For each (j, k) with ε(j) ≠ 0 and k_idx in c1_w_set:
                #   Gram[k_loc, l_loc] += -ε(j) * conj(v_{j_local, l_loc})
                #   Gram[l_loc, k_loc] += -ε(j) * v_{j_local, l_loc}  (Hermitian)
                #   ε(j) ≠ 0 means j is Cartan, so s = 0.
                if s == 0 and k_idx in c1_w_set:
                    k_loc = c1_w_map[k_idx]
                    # For each j_local with ε(c1_s[j_local]) ≠ 0:
                    for j_local in range(len(c1_s)):
                        j_global = int(c1_s[j_local])
                        if abs(epsilon[j_global]) > 1e-13:
                            col = V_restricted[:, j_local].toarray().flatten()
                            Gram[k_loc, :] += -epsilon[j_global] * np.conj(col)
                            Gram[:, k_loc] += -epsilon[j_global] * col

                # Term 1 x Term 3 and Term 3 x Term 1: ε(j) ε(k) δ_{ik} δ_{i'j}
                #   Only when both ε(j) and ε(k) are nonzero (both Cartan).
                #   wt(j) + wt(k) = w with both wt = 0 means w = 0.
                #   For w = 0, s = 0, s2 = 0: all (j, k) Cartan pairs contribute 1 to Gram[k_loc, j_loc].
                if w == 0 and s == 0 and s2 == 0:
                    # For each j in Cartan (c1_s ∩ ε ≠ 0) and k in Cartan (c1_s2 ∩ ε ≠ 0):
                    # Gram[k_loc, j_loc] += 1, Gram[j_loc, k_loc] += 1.
                    cartan_j = [int(x) for x in c1_s if abs(epsilon[int(x)]) > 1e-13]
                    cartan_k = [int(x) for x in c1_s2 if abs(epsilon[int(x)]) > 1e-13]
                    for j_g in cartan_j:
                        for k_g in cartan_k:
                            j_l = c1_w_map[j_g]
                            k_l = c1_w_map[k_g]
                            Gram[k_l, j_l] += 1.0
                            Gram[j_l, k_l] += 1.0

                n_k_processed += 1
                if verbose and n_k_processed % 500 == 0:
                    elapsed = time.time() - t_block_start
                    eta = elapsed / n_k_processed * (3 * len(c1_s2) - n_k_processed) * 3
                    print(f"      k #{n_k_processed}: {elapsed:.1f}s elapsed, "
                          f"~{eta:.0f}s remaining for this weight block")

        # Compute rank of Gram via SVD
        # Hermitianize to remove numerical asymmetry
        Gram = (Gram + Gram.conj().T) / 2
        svals = np.linalg.svd(Gram, compute_uv=False)
        s_max = svals[0] if len(svals) > 0 and svals[0] > 0 else 1.0
        tol = max(Gram.shape) * s_max * 1e-10
        rank_w = int(np.sum(svals > tol))
        total_rank_d1 += rank_w
        block_results[w] = {
            "n_c1": n_c1,
            "n_pairs": n_pairs_total,
            "rank": rank_w,
            "dim_ker": n_c1 - rank_w,
            "time": time.time() - t_block_start,
            "svals_top": svals[:10] if len(svals) >= 10 else svals,
            "svals_bottom_nonzero": svals[svals > tol][-5:] if np.any(svals > tol) else [],
            "svals_bottom_zero": svals[svals <= tol][:5] if np.any(svals <= tol) else [],
        }
        if verbose:
            print(f"    Total pairs processed: {n_pairs_total}")
            print(f"    rank(d^1_w) = {rank_w}")
            print(f"    dim ker(d^1_w) = {n_c1 - rank_w}")
            print(f"    Block time: {time.time() - t_block_start:.1f}s")
            print(f"    Top singular values: {block_results[w]['svals_top'][:5]}")
            print(f"    Smallest nonzero: {block_results[w]['svals_bottom_nonzero']}")
            print(f"    Largest 'zero'   : {block_results[w]['svals_bottom_zero']}")

    dim_hh1 = DIM - total_rank_d1
    if verbose:
        print(f"\n  TOTAL rank(d^1) = {total_rank_d1}")
        print(f"  dim HH^1 = 6561 - {total_rank_d1} = {dim_hh1}")
        print(f"  Total time: {time.time() - t_start:.1f}s")
        print(f"  Expected (from Anick W2-1a-IR): HH^1 = 1")

    return {
        "block_results": block_results,
        "total_rank_d1": total_rank_d1,
        "dim_hh1": dim_hh1,
    }


# ============================================================================
# HH^2 of B^+(u_q(sl_3)) — sanity check on multiplication
# ============================================================================
#
# B^+ is the positive Borel subalgebra: K1, K2, E1, E12, E2 (5 generators).
# PBW basis: K1^a K2^b E1^c E12^e E2^d, dim 3^5 = 243.
# At ell=3, weight decomposition gives 3 weight spaces of 81 each.
# Per weight block: dim C^1 = 81, dim C^2 = 3 * 81^2 = 19683, dim C^3 = 9 * 81^3 = 4.8M.
# Gram matrix of d^2 per weight block: 19683 x 19683 ≈ 3 GB dense. Tractable with care.
# Expected: dim HH^2(B^+) = 5 (W2-1b, paper Sec. 6.5).
#
# Rather than re-implement B^+ separately, we extract the B^+ subalgebra from
# the full multiplication table by restricting to PBW indices with f = g = h = 0.


def compute_hh2_bplus(L_mat, R_mat, verbose: bool = True) -> dict:
    """Compute dim HH^2(B^+(u_q(sl_3)), C) using the full-algebra mult table.

    B^+ is the subalgebra with PBW elements K1^a K2^b E1^c E12^e E2^d (f=g=h=0).
    """
    if verbose:
        print("\n=== HH^2(B^+) sanity check ===")

    # Enumerate B^+ PBW elements and their indices in the full algebra
    bplus_indices = []  # list of full-algebra indices
    bplus_local_to_global = {}  # local B+ index -> full-algebra index
    bplus_global_to_local = {}  # full-algebra index -> local B+ index
    for a, b, c, e, d in itertools.product(range(ELL), repeat=5):
        full_idx = to_idx((a, b, c, e, d, 0, 0, 0))
        local_idx = len(bplus_indices)
        bplus_indices.append(full_idx)
        bplus_local_to_global[local_idx] = full_idx
        bplus_global_to_local[full_idx] = local_idx

    dim_bplus = len(bplus_indices)
    if verbose:
        print(f"  dim B^+ = {dim_bplus} (expected 243)")
    assert dim_bplus == 243, f"B^+ dim mismatch: {dim_bplus}"

    # Weight function for B^+: weight(K1^a K2^b E1^c E12^e E2^d) = (2c + e + 2d) mod 3
    def weight_bplus(local_idx):
        full = bplus_local_to_global[local_idx]
        a, b, c, e, d, f, g, h = from_idx(full)
        return (2 * c + e + 2 * d) % ELL

    # Counit for B^+: ε = 1 iff c = e = d = 0.
    epsilon_bplus = np.zeros(dim_bplus, dtype=complex)
    for local, full in bplus_local_to_global.items():
        a, b, c, e, d, f, g, h = from_idx(full)
        if c == 0 and e == 0 and d == 0:
            epsilon_bplus[local] = 1.0
    if verbose:
        print(f"  ε-nonzero in B^+: {int(np.sum(np.abs(epsilon_bplus) > 0))} (expected 9)")

    # Build weight index maps for B^+
    wt_of_bplus = np.array([weight_bplus(i) for i in range(dim_bplus)], dtype=int)
    wt_indices_bplus = {s: np.where(wt_of_bplus == s)[0] for s in range(ELL)}
    if verbose:
        print(f"  B^+ weight-space sizes: {[len(wt_indices_bplus[s]) for s in range(ELL)]}")
        print(f"    (Expected: 81 per weight, total 243)")

    # Build B^+-restricted multiplication: bplus_mult[local_i][local_j] = list of (local_k, v)
    # such that a_i * a_j (in full algebra) = sum v * a_k where a_k is in B^+.
    # Since B^+ is a subalgebra, the product stays in B^+.
    if verbose:
        print("  Building B^+ multiplication table...")
    t0 = time.time()
    # Use batch multiplication: for each j (right factor), compute {a_i * a_j : i}
    # by extracting the columns of the full mult table corresponding to B^+ elements.
    bplus_mult = [[None] * dim_bplus for _ in range(dim_bplus)]
    for j_local in range(dim_bplus):
        j_global = bplus_local_to_global[j_local]
        # Multiply all B^+ elements by a_j (on the right)
        V = multiply_batch(bplus_indices, j_global, R_mat)  # sparse, DIM x 243
        V = V.tocoo()
        # Group by column (i_local)
        col_data = {}
        for r, c, v in zip(V.row, V.col, V.data):
            if abs(v) < 1e-13:
                continue
            c = int(c)
            r = int(r)
            if c not in col_data:
                col_data[c] = []
            col_data[c].append((r, v))
        for i_local, terms in col_data.items():
            # Each (r, v) should have r in B^+ (since B^+ is closed under mult)
            filtered = []
            for r, v in terms:
                if r in bplus_global_to_local:
                    filtered.append((bplus_global_to_local[r], v))
                else:
                    # This shouldn't happen if B^+ is a subalgebra
                    if verbose and abs(v) > 1e-10:
                        print(f"    WARNING: non-B^+ term {r} with coeff {v} in B^+ product")
            bplus_mult[i_local][j_local] = filtered
    if verbose:
        print(f"    Built in {time.time() - t0:.1f}s")

    # Compute HH^2(B^+) via weight-decomposed bar complex
    total_rank_d2 = 0
    total_rank_d1 = 0
    total_dim_ker_d2 = 0
    block_results = {}

    for w in range(ELL):
        if verbose:
            print(f"\n  Weight block w = {w}:")
        c1_w = wt_indices_bplus[w]  # local indices in B^+ with weight w
        n_c1 = len(c1_w)
        c1_w_set = set(int(x) for x in c1_w)
        c1_w_map = {int(x): i for i, x in enumerate(c1_w)}

        # C^2_w: pairs (j, k) with wt(j) + wt(k) = w. dim = 3 * 81^2 = 19683.
        c2_pairs = []  # list of (j_local, k_local)
        for s in range(ELL):
            s2 = (w - s) % ELL
            for j in wt_indices_bplus[s]:
                for k in wt_indices_bplus[s2]:
                    c2_pairs.append((int(j), int(k)))
        n_c2 = len(c2_pairs)
        if verbose:
            print(f"    dim C^1_w = {n_c1}, dim C^2_w = {n_c2}")

        # Build d^1: matrix n_c2 x n_c1
        # d^1[(j,k), i] = ε(j) δ_{ik} - coeff(a_i in a_j*a_k) + δ_{ij} ε(k)
        d1 = np.zeros((n_c2, n_c1), dtype=complex)
        for row, (j, k) in enumerate(c2_pairs):
            for i_local, i_global in enumerate(c1_w):
                # Term 1: ε(j) δ_{ik} = ε(j) if k_global == i_global
                if j == i_global:  # local index j vs global i_global... wait we need to check this
                    pass
            # Recompute properly: i_global is the column, k is local so k_global = bplus_local_to_global[k]
            k_global = bplus_local_to_global[k]
            j_global = bplus_local_to_global[j]
            for i_local, i_global in enumerate(c1_w):
                i_global = int(i_global)
                val = 0.0 + 0j
                # Term 1: ε(a_j) δ_{ik}, where i, k are global indices
                if k_global == i_global and abs(epsilon_bplus[j]) > 1e-13:
                    val += epsilon_bplus[j]
                # Term 2: -coeff(a_i in a_j * a_k)
                for (l, v) in bplus_mult[j][k]:
                    if l == i_local:
                        val -= v
                        break
                # Term 3: δ_{ij} ε(a_k), where i, j are global indices
                if j_global == i_global and abs(epsilon_bplus[k]) > 1e-13:
                    val += epsilon_bplus[k]
                d1[row, i_local] = val
        rank_d1 = int(np.linalg.matrix_rank(d1, tol=1e-9))
        if verbose:
            print(f"    rank(d^1_w) = {rank_d1}")

        # Build d^2: matrix n_c3 x n_c2
        # d^2[(i,j,k), (a,b)] = ε(i) δ_{ja} δ_{kb} - δ_{ia'} δ_{jb'} (where i*a = a') + δ_{ia} δ_{bb'} (where b*c = b') - ε(k) δ_{ia} δ_{jb}
        # Simplified:
        # d^2[(i,j,k), (a,b)] = ε(i) δ_{ja} δ_{kb}
        #                    - [coeff of a_a in i*j] δ_{kb}
        #                    + δ_{ia} [coeff of a_b in j*k]
        #                    - ε(k) δ_{ia} δ_{jb}
        # C^3_w: triples (i, j, k) with wt(i) + wt(j) + wt(k) = w. dim = 9 * 81^3 = 4.8M.
        # Sparse d^2: each column (a, b) has ~30 + 30 + 9 + 9 = ~80 nonzeros.

        # Build d^2 as sparse matrix
        c2_map = {pair: i for i, pair in enumerate(c2_pairs)}

        rows_l, cols_l, vals_l = [], [], []

        # Enumerate triples (i, j, k) with wt(i) + wt(j) + wt(k) = w
        # For each (j, k) in C^2_w, we have wt(j) + wt(k) = w. Then i must have wt(i) = 0.
        # But we also need triples like (i, j, k) where wt(i) + wt(j) + wt(k) = w but
        # wt(j) + wt(k) != w (e.g., wt(i) = 1, wt(j) = 1, wt(k) = w - 2).
        # So we need to iterate over all (i, j, k) with the constraint.

        # Use the formula: for each triple (i, j, k) with wt sum = w,
        # the row (i, j, k) of d^2 has nonzeros at columns:
        #   (j, k) with value ε(i)
        #   (i*j, k) with value -1 (if i*j is a single term) -- but i*j is a polynomial!
        #   (i, j*k) with value +1 (similarly)
        #   (i, j) with value -ε(k)
        # Wait, let me re-derive.

        # d^2 f(a, b, c) = ε(a) f(b, c) - f(ab, c) + f(a, bc) - ε(c) f(a, b)
        # For f = e_{(a', b')}* (basis of C^2 dual):
        # d^2[(a, b, c), (a', b')] = ε(a) δ_{a', b} δ_{b', c}
        #                          - [coeff of a' in a*b] δ_{b', c}
        #                          + δ_{a', a} [coeff of b' in b*c]
        #                          - ε(c) δ_{a', a} δ_{b', b}

        # So for each triple (a, b, c) with wt(a) + wt(b) + wt(c) = w:
        # nonzero at column (b, c) with value ε(a)
        # nonzero at column (a*b, c) for each term (l, v) in a*b: -v δ_{b', c} where a' = l, b' = c
        #   But (l, c) is in C^2_w only if wt(l) + wt(c) = w. Since wt(l) = wt(a) + wt(b) and wt(c) is fixed, wt(l) + wt(c) = wt(a) + wt(b) + wt(c) = w. Good.
        # nonzero at column (a, b*c) for each term (l, v) in b*c: +v δ_{a', a} where a' = a, b' = l
        # nonzero at column (a, b) with value -ε(c)

        # Iterate over (a, b, c). For each, find the 4 contributions.
        # Total triples: 9 * 81^3 = 4.8M per weight. Each gives ~4 + 30 + 30 = ~64 nonzero entries.

        # To make this tractable, iterate over (a, b) (in C^2_w) and c (in wt = w - wt(a) - wt(b)).
        # For each (a, b) in C^2_w (i.e., wt(a) + wt(b) = w), c can have wt = 0 (since w - wt(a) - wt(b) = 0).
        # Wait, no — we need wt(a) + wt(b) + wt(c) = w, so if wt(a) + wt(b) = w then wt(c) = 0.
        # But there are other combinations: e.g., wt(a) = 0, wt(b) = 0, wt(c) = w. Then (a, b) is in C^2_0, not C^2_w.
        # Hmm.

        # Actually, all 9 weight combinations (wt(a), wt(b), wt(c)) with sum = w contribute.
        # Let's enumerate.

        n_c3 = 0
        # We need to enumerate triples efficiently. For each (a, b) in C^2_{s_ab} for s_ab in {0, 1, 2},
        # c ranges over wt = (w - s_ab) mod 3.
        # So 3 (a, b) weight blocks × 81 c values per block = 3 * 81^2 * 81 = 3 * 81^3 = 1.6M? Let me recount.
        # For each s_ab in {0, 1, 2}: dim C^2_{s_ab} = 3 * 81^2 = 19683. c ranges over 81 elements. Total: 19683 * 81 = 1.6M.
        # For 3 values of s_ab: 3 * 1.6M = 4.8M triples. ✓

        # For each (a, b) in C^2 (any weight), iterate over c (any weight), and only keep triples with wt sum = w.
        # This is 243^2 * 243 = 14M triples, of which 1/3 = 4.8M satisfy the weight constraint.
        # We can do this more efficiently by iterating only over (a, b, c) with the right weights.

        # Strategy: iterate over (a, b, c) by weight classes.
        # For each (s_a, s_b, s_c) with s_a + s_b + s_c = w mod 3:
        #   a in wt=s_a (81), b in wt=s_b (81), c in wt=s_c (81). Total 81^3 = 531K triples per weight class.
        # 9 weight classes × 531K = 4.8M triples.

        t_d2_start = time.time()
        for s_a in range(ELL):
            for s_b in range(ELL):
                s_c = (w - s_a - s_b) % ELL
                wt_a = wt_indices_bplus[s_a]
                wt_b = wt_indices_bplus[s_b]
                wt_c = wt_indices_bplus[s_c]
                for a in wt_a:
                    a = int(a)
                    a_global = bplus_local_to_global[a]
                    eps_a = epsilon_bplus[a]
                    for b in wt_b:
                        b = int(b)
                        b_global = bplus_local_to_global[b]
                        # Compute a*b (polynomial in B^+)
                        ab_terms = bplus_mult[a][b]  # list of (local_idx, v)
                        for c in wt_c:
                            c = int(c)
                            c_global = bplus_local_to_global[c]
                            eps_c = epsilon_bplus[c]
                            n_c3 += 1
                            row = n_c3 - 1  # 0-indexed row

                            # Term 1: ε(a) δ_{a', b} δ_{b', c}
                            # Column (b, c) with value ε(a)
                            if abs(eps_a) > 1e-13:
                                col = c2_map.get((b, c))
                                if col is not None:
                                    rows_l.append(row)
                                    cols_l.append(col)
                                    vals_l.append(eps_a)

                            # Term 2: -[coeff of a' in a*b] δ_{b', c}
                            # For each (l, v) in a*b, column (l, c) with value -v
                            for (l, v) in ab_terms:
                                pair = (l, c)
                                col = c2_map.get(pair)
                                if col is not None:
                                    rows_l.append(row)
                                    cols_l.append(col)
                                    vals_l.append(-v)

                            # Term 3: +δ_{a', a} [coeff of b' in b*c]
                            # For each (l, v) in b*c, column (a, l) with value +v
                            bc_terms = bplus_mult[b][c]
                            for (l, v) in bc_terms:
                                pair = (a, l)
                                col = c2_map.get(pair)
                                if col is not None:
                                    rows_l.append(row)
                                    cols_l.append(col)
                                    vals_l.append(v)

                            # Term 4: -ε(c) δ_{a', a} δ_{b', b}
                            # Column (a, b) with value -ε(c)
                            if abs(eps_c) > 1e-13:
                                col = c2_map.get((a, b))
                                if col is not None:
                                    rows_l.append(row)
                                    cols_l.append(col)
                                    vals_l.append(-eps_c)

        if verbose:
            print(f"    dim C^3_w = {n_c3}")
            print(f"    d^2 build time: {time.time() - t_d2_start:.1f}s")
            print(f"    d^2 nnz = {len(vals_l)}")

        d2 = sparse.csr_matrix(
            (vals_l, (rows_l, cols_l)),
            shape=(n_c3, n_c2),
            dtype=complex,
        )
        del rows_l, cols_l, vals_l

        # Compute Gram matrix d^2^H d^2 (size n_c2 x n_c2 = 19683 x 19683)
        if verbose:
            print(f"    Computing Gram = d^2^H d^2 ({n_c2} x {n_c2})...")
        t_gram_start = time.time()
        # For 19683 x 19683 dense complex, that's 19683^2 * 16 bytes = 6 GB. Too big.
        # Use sparse Gram instead, or compute rank via eigs of d^2^H d^2.
        # Approach: compute as many eigenvalues of d^2^H d^2 as possible using
        # scipy.sparse.linalg.eigsh. But this only gives a few eigenvalues, not the rank.
        # Alternative: use the SVD of d^2 directly via sparse iterative methods.

        # For dim 19683, we can try a sparse-rank approach: compute the rank of d^2
        # via QR on a sparse matrix, or via randomized SVD.

        # Simpler: convert d^2 to dense if it fits.
        # d^2 is n_c3 x n_c2 = 4.8M x 19683. As dense complex: 4.8M * 19683 * 16 bytes = 1.5 TB. NO.
        # As sparse: nnz = ~4.8M * 60 = 290M, total ~3.5 GB. Borderline.

        # Try: Gram = d^2^H @ d^2 as a sparse matrix, then compute rank via
        # sparse eigenvalues (eigsh).
        # Gram has size 19683 x 19683, with potentially many nonzeros.
        # d^2^H @ d^2: cost is O(nnz(d^2) * avg_nnz_per_col(d^2)) = O(290M * 60) = O(17G ops). Slow.

        # Alternative: compute rank via scipy.sparse.linalg.svds with k = some value.
        # svds gives the top-k singular values. If we get all singular values > tol,
        # we can determine the rank. But for rank ~ 15000, we'd need k = 15000+ which
        # is too slow for svds.

        # Cheapest: compute Gram as a dense matrix using sparse @ sparse -> dense.
        # Cost: O(nnz(d2) * avg_nnz_per_col(d2)) = same as above, but with numpy
        # vectorization it's much faster.
        # Memory: 19683^2 * 16 bytes = 6 GB. Too big for 4 GB RAM sandbox.

        # Approach: compute Gram in chunks (column blocks of d2).
        # For each chunk of 1000 columns of d2, compute Gram[:, chunk] = d2^H @ d2[:, chunk].
        # Then SVD the full Gram at the end.

        # Actually, for the B^+ case, the per-weight-block Gram is 19683 x 19683 = 6 GB.
        # We have 3.9 GB RAM. Too big.

        # Compromise: use scipy.sparse.linalg.eigsh to compute the top eigenvalues
        # of d2^H @ d2 (operating as a linear operator, no materialization).
        # This gives us the top eigenvalues, from which we can determine the rank
        # IF the rank is small (< ~100). For B^+, rank(d^2) is expected to be ~15000+.

        # Hmm, that won't work either. Let me just compute the dense Gram and hope it fits.
        # 6 GB > 3.9 GB, so it won't fit.

        # Alternative: use a different approach. Compute rank(d^2) via randomized SVD.
        # Sketch: form a random matrix Omega of size n_c2 x k (where k ~ rank estimate + buffer),
        # compute Y = d2 @ Omega (size n_c3 x k), then QR-decompose Y to get Q (n_c3 x k),
        # then B = Q^H @ d2 (size k x n_c2), then SVD(B).
        # This gives an approximate rank.

        # For an exact rank, we need a different approach.

        # Punt: just report that the Gram matrix is too big to materialize,
        # and try the sparse rank computation.

        try:
            # Try sparse-dense hybrid: compute Gram = (d2^H @ d2).toarray() in chunks
            # For n_c2 = 19683, chunk size 1000 -> 20 chunks
            chunk_size = 500
            Gram = np.zeros((n_c2, n_c2), dtype=np.float64)  # use real since Gram is Hermitian
            d2_csc = d2.tocsc()
            for chunk_start in range(0, n_c2, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_c2)
                # Extract columns chunk_start:chunk_end of d2
                d2_chunk = d2_csc[:, chunk_start:chunk_end]
                # Gram[:, chunk] = d2^H @ d2_chunk
                gram_chunk = (d2.conj().T @ d2_chunk).toarray()
                Gram[:, chunk_start:chunk_end] = gram_chunk.real  # Hermitian, take real part
                # Also fill the symmetric part
                if chunk_start != chunk_end - 1:
                    # Already symmetric if we do all chunks, but for now just store
                    pass
            del d2_csc, d2_chunk, gram_chunk

            # Symmetrize
            Gram = (Gram + Gram.T) / 2

            if verbose:
                print(f"    Gram built in {time.time() - t_gram_start:.1f}s")

            # Compute eigenvalues
            eigvals = np.linalg.eigvalsh(Gram)
            eig_max = eigvals[-1] if len(eigvals) > 0 else 1.0
            tol = max(Gram.shape) * eig_max * 1e-10
            rank_d2 = int(np.sum(eigvals > tol))
            dim_ker_d2 = n_c2 - rank_d2

            if verbose:
                print(f"    rank(d^2_w) = {rank_d2}")
                print(f"    dim ker(d^2_w) = {dim_ker_d2}")
                print(f"    Largest eigenvalues: {eigvals[-5:]}")
                print(f"    Smallest nonzero: {eigvals[eigvals > tol][:5]}")
                if np.any(eigvals <= tol):
                    print(f"    Largest 'zero': {eigvals[eigvals <= tol][-5:]}")
                print(f"    HH^2_w = dim ker(d^2_w) - rank(d^1_w) = {dim_ker_d2} - {rank_d1} = {dim_ker_d2 - rank_d1}")

            total_rank_d2 += rank_d2
            total_rank_d1 += rank_d1
            total_dim_ker_d2 += dim_ker_d2
            block_results[w] = {
                "n_c1": n_c1, "n_c2": n_c2, "n_c3": n_c3,
                "rank_d1": rank_d1, "rank_d2": rank_d2, "dim_ker_d2": dim_ker_d2,
                "hh2_w": dim_ker_d2 - rank_d1,
            }
            del Gram, eigvals
        except MemoryError:
            if verbose:
                print(f"    MEMORY ERROR: cannot build dense Gram matrix for B^+ weight block {w}")
                print(f"    (n_c2 = {n_c2}, dense Gram = {n_c2**2 * 16 / 1e9:.1f} GB)")
            block_results[w] = {"error": "MemoryError"}

    dim_hh2_bplus = total_dim_ker_d2 - total_rank_d1
    if verbose:
        print(f"\n  TOTAL rank(d^1) = {total_rank_d1}")
        print(f"  TOTAL rank(d^2) = {total_rank_d2}")
        print(f"  TOTAL dim ker(d^2) = {total_dim_ker_d2}")
        print(f"  dim HH^2(B^+) = {dim_hh2_bplus}")
        print(f"  Expected (W2-1b, paper Sec. 6.5): HH^2(B^+) = 5")

    return {
        "block_results": block_results,
        "total_rank_d1": total_rank_d1,
        "total_rank_d2": total_rank_d2,
        "total_dim_ker_d2": total_dim_ker_d2,
        "dim_hh2_bplus": dim_hh2_bplus,
    }


# ============================================================================
# Documentation of HH^2(u_q(sl_3)) intractability
# ============================================================================


def document_hh2_intractability(verbose: bool = True) -> dict:
    """Document why dim HH^2(u_q(sl_3), C) is intractable via the bar complex."""
    if verbose:
        print("\n=== HH^2(u_q(sl_3)) intractability analysis ===")

    # Per weight block (3 blocks at ell=3):
    dim_c1_w = 2187
    dim_c2_w = 3 * dim_c1_w ** 2  # 3 weight pairs per block
    dim_c3_w = 9 * dim_c1_w ** 3  # 9 weight triples per block

    # Gram matrix of d^2 per weight block:
    gram_size_bytes = dim_c2_w ** 2 * 16  # complex doubles
    gram_size_gb = gram_size_bytes / 1e9

    # Sparse d^2 storage:
    # Each column (pair (a, b)) of d^2 has ~4 + 30 + 30 = ~64 nonzeros (boundary + product terms).
    # Total nnz per weight block: dim_c2_w * 64 = 14.3M * 64 = 915M.
    d2_nnz_per_block = dim_c2_w * 64
    d2_storage_gb = d2_nnz_per_block * 16 / 1e9  # complex doubles

    # Sparse Gram matrix:
    # Each entry Gram[i, i'] = sum over rows of d^2 of products. The number of nonzero
    # entries in Gram is roughly dim_c2_w * (avg # of columns sharing a row with each column).
    # For a typical bar complex, this is ~1000 nonzeros per row of Gram.
    gram_nnz_per_block = dim_c2_w * 1000
    gram_sparse_storage_gb = gram_nnz_per_block * 16 / 1e9

    if verbose:
        print(f"  Per weight block (3 blocks at ell=3):")
        print(f"    dim C^1_w = {dim_c1_w}")
        print(f"    dim C^2_w = {dim_c2_w} (~14.3M)")
        print(f"    dim C^3_w = {dim_c3_w:g} (~94 billion)")
        print(f"    Gram matrix of d^2 (dense): {dim_c2_w}x{dim_c2_w} = {gram_size_gb:.0f} GB")
        print(f"    Sparse d^2 storage: ~{d2_storage_gb:.1f} GB")
        print(f"    Sparse Gram storage: ~{gram_sparse_storage_gb:.1f} GB")
        print()
        print(f"  Sandbox resources:")
        print(f"    RAM: 3.9 GB")
        print(f"    CPUs: 2")
        print(f"    Timeout per command: 2 hours")
        print()
        print(f"  VERDICT: HH^2(u_q(sl_3)) via the full bar complex is INTRACTABLE.")
        print(f"    - Dense Gram matrix of d^2 is ~300 TB (per weight block).")
        print(f"    - Sparse d^2 storage is ~14 GB per weight block (exceeds 3.9 GB RAM).")
        print(f"    - Sparse Gram storage is ~229 GB per weight block (exceeds 3.9 GB RAM).")
        print(f"    - Even mat-vec on d^2 requires streaming over 94G rows of C^3 (per weight block).")
        print()
        print(f"  HARDWARE NEEDED for direct bar-complex computation:")
        print(f"    - ~256 GB RAM (to hold sparse d^2 + sparse Gram for one weight block)")
        print(f"    - ~16 CPUs (for parallel sparse mat-mul)")
        print(f"    - Estimated time: 1-7 days for one weight block; ~3 weeks for all 3 weights.")
        print()
        print(f"  ALTERNATIVE PATHWAYS (all blocked):")
        print(f"    1. Anick d_3 (W2-1a-IR): buggy, gives HH^2 = -34 (impossible).")
        print(f"    2. Braided Hochschild (Negron): C^2 ~ 530K, C^3 ~ 3.5G. Still intractable.")
        print(f"    3. BGG / Hemelsoet-Voorhaar: principal-block s=2 case excluded from HV Prop 5.1.")
        print(f"    4. LES + Hb^1(B^+): tractable, but gives im(δ) = 2, not dim HH^2.")
        print(f"       dim HH^2 = im(δ) + im(π̄) = 2 + im(π̄); need im(π̄) which requires HH^2 itself.")

    return {
        "dim_c1_w": dim_c1_w,
        "dim_c2_w": dim_c2_w,
        "dim_c3_w": dim_c3_w,
        "gram_dense_gb": gram_size_gb,
        "d2_sparse_gb": d2_storage_gb,
        "gram_sparse_gb": gram_sparse_storage_gb,
        "verdict": "INTRACTABLE in 4 GB sandbox; needs ~256 GB RAM + 1-7 days per weight block.",
    }


# ============================================================================
# Main
# ============================================================================


def main(verbose: bool = True) -> dict:
    """Run the full W3-1a computation."""
    if verbose:
        print("=" * 70)
        print("W3-1a: Bar complex on PBW normal forms for u_q(sl_3) at ell = 3")
        print("=" * 70)
        print()
        print("Goal: compute dim HH^1(u_q(sl_3), C) via the bar complex,")
        print("      verify dim HH^2(B^+) = 5 as a sanity check, and document")
        print("      the intractability of dim HH^2(u_q(sl_3), C).")
        print()

    # Build multiplication tables
    if verbose:
        print("Step 1: build L_g and R_g multiplication tables via IR reducer")
    t0 = time.time()
    L_mat, R_mat = build_mult_tables(verbose=verbose)
    if verbose:
        print(f"  Total build time: {time.time() - t0:.1f}s")
        for g in range(8):
            print(f"    {GEN_NAMES[g]}: L nnz={L_mat[g].nnz}, R nnz={R_mat[g].nnz}")

    # Sanity checks
    if verbose:
        print("\nStep 2: sanity checks on multiplication table")
    sanity = sanity_checks(L_mat, R_mat, verbose=verbose)
    all_ok = all(sanity.values())
    if verbose:
        print(f"  All sanity checks passed: {all_ok}")
    if not all_ok:
        if verbose:
            print("  WARNING: some sanity checks failed; results may be unreliable.")

    # HH^1 computation
    if verbose:
        print("\nStep 3: compute dim HH^1 via weight-decomposed Gram matrix")
    hh1_results = compute_hh1(L_mat, R_mat, verbose=verbose)

    # HH^2(B+) sanity check
    if verbose:
        print("\nStep 4: compute dim HH^2(B^+) as a sanity check")
    hh2_bplus_results = compute_hh2_bplus(L_mat, R_mat, verbose=verbose)

    # HH^2 intractability documentation
    if verbose:
        print("\nStep 5: document HH^2(u_q(sl_3)) intractability")
    hh2_intract = document_hh2_intractability(verbose=verbose)

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  dim HH^1(u_q(sl_3), C) = {hh1_results['dim_hh1']}")
        print(f"    (Expected from Anick W2-1a-IR: 1)")
        print(f"    Match: {hh1_results['dim_hh1'] == 1}")
        print()
        print(f"  dim HH^2(B^+(u_q(sl_3)), C) = {hh2_bplus_results.get('dim_hh2_bplus', 'N/A')}")
        print(f"    (Expected from W2-1b / paper Sec. 6.5: 5)")
        print()
        print(f"  dim HH^2(u_q(sl_3), C) = INTRACTABLE via bar complex")
        print(f"    ({hh2_intract['verdict']})")
        print()
        print(f"  LES-based bounds (verified elsewhere):")
        print(f"    dim HH^2 = dim im(δ) + dim im(π̄)")
        print(f"             = 2 + dim im(π̄)   (since dim H̃¹_b(B^+) = 2, W3-h1b)")
        print(f"             ∈ [2, 12]          (since 0 ≤ dim im(π̄) ≤ 10)")
        print(f"    Conjecture: dim HH^2 = 9 → dim im(π̄) = 7")
        print(f"    Alternative: dim HH^2 = 8 → dim im(π̄) = 6")
        print(f"  Resolving 8 vs 9 requires either:")
        print(f"    (a) ~256 GB RAM machine for direct bar complex (~3 weeks), OR")
        print(f"    (b) A correct Anick d_3 implementation (~1-2 weeks dev), OR")
        print(f"    (c) An independent theoretical argument.")
        print("=" * 70)

    return {
        "sanity_checks": sanity,
        "hh1_results": hh1_results,
        "hh2_bplus_results": hh2_bplus_results,
        "hh2_intractability": hh2_intract,
    }


if __name__ == "__main__":
    main(verbose=True)
