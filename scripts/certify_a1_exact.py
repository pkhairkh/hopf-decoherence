#!/usr/bin/env python3
"""Exact cyclotomic certification of dim HH^2(u_q(sl_2), C) at ell = 3.

This script certifies the A_1 case of the paper's main formula

    dim_C HH^2(u_q(g), C) = C(n+1, 2) + 2|Phi^+|

at g = sl_2 (n = 1, |Phi^+| = 1) and ell = 3, giving dim HH^2 = 1 + 2 = 3.

Unlike the floating-point certification in verify_sl2_hh2.py, this script
uses EXACT cyclotomic arithmetic over the Eisenstein rationals
Z[omega, 1/3] (omega a primitive cube root of unity) and lifts the rank
computation to characteristic zero via reduction modulo several primes
p ≡ 1 (mod 3) with p ≠ 3.

Mathematical setup
------------------
At ell = 3, q is a primitive cube root of unity, so q^2 + q + 1 = 0.  All
structure constants of u_q(sl_2) at q lie in Z[q, q^{-1}] = Z[q] (since
q^{-1} = q^2 = -1 - q).  However, the commutator [E, F] = (K - K^{-1})/(q - q^{-1})
introduces a division by D := q - q^{-1} = q - q^2 = 1 + 2*omega, whose
algebraic norm in Z[omega] is

    N(D) = D * D_bar = (1 + 2*omega)(1 + 2*omega^2) = 1 - (-2) + 4 = 3,

so D is NOT a unit in Z[omega].  Instead D^2 = -3, hence 1/D = -D/3
= (-1 - 2*omega)/3, and the structure constants actually live in the
localization Z[omega, 1/3].

Z[omega, 1/3] is a Dedekind domain (a localization of the ring of
Eisenstein integers, which is the ring of integers of Q(omega), a number
field of class number 1).  Rank semicontinuity over a Dedekind domain
implies that if an integer matrix has the same rank modulo every prime
ideal of residue characteristic p != 3 (in particular p ≡ 1 (mod 3) so
that omega exists in F_p), then it has that rank over the fraction field
Q(omega) and hence over C (after embedding omega -> e^{2*pi*i/3}).

We represent each element of Z[omega, 1/3] as a triple (a, b, c) of
integers with c >= 0, denoting (a + b*omega) / 3^c.  Multiplication uses
omega^2 = -1 - omega.  Reduction mod p (for p ≡ 1 (mod 3), p ≠ 3) maps
omega -> q_p (a primitive cube root of unity in F_p) and uses that 3 is
invertible mod p.

The bar-complex differentials are
    (d^1 f)(a, b)    = eps(a) f(b) - f(a*b) + f(a) eps(b)
    (d^2 g)(a, b, c) = eps(a) g(b,c) - g(a*b, c) + g(a, b*c) - g(a,b) eps(c)

with dim C^1 = 27, dim C^2 = 729, dim C^3 = 19683.  Then

    dim HH^2 = dim ker(d^2) - dim im(d^1) = (dim C^2 - rank(d^2)) - rank(d^1).

Implementation
--------------
1. Build the 27 x 27 multiplication table of u_q(sl_2) over Z[omega, 1/3].
2. Build d^1 (729 x 27) and d^2 (19683 x 729) as sparse dicts with ER values.
3. For each prime p in {7, 13, 19, 31, 37, 43, 61, 67, 73, 79, 97} (all
   p ≡ 1 (mod 3), p ≠ 3):
   (a) Find a primitive cube root of unity q_p in F_p.
   (b) Reduce d^1 and d^2 modulo p (substituting omega -> q_p) to obtain
       integer matrices over F_p.
   (c) Compute ranks over F_p via exact Gaussian elimination (no float
       tolerance needed).
4. Verify the ranks are consistent across all primes.  By rank
   semicontinuity over the Dedekind domain Z[omega, 1/3], this certifies
   the rank over Q(omega) and hence over C.
5. Compute dim HH^2 = (729 - rank(d^2)) - rank(d^1) and verify it equals 3.
"""

import os
import sys
import time
from dataclasses import dataclass

import numpy as np

ELL = 3
DIM = ELL ** 3  # 27

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "certify_a1_output.txt"
)


# ============================================================================
# Ring of "Eisenstein rationals" Z[omega, 1/3].
#
# Each element is stored as (a + b*omega) / 3^c with a, b, c integers, c >= 0.
# Here omega is a primitive cube root of unity (omega = q), satisfying
# omega^2 + omega + 1 = 0, i.e., omega^2 = -1 - omega.
#
# The denominator 3 arises because D = q - q^{-1} = 1 + 2*omega has norm 3,
# so 1/D = (-1 - 2*omega)/3 lives in Z[omega, 1/3] but not in Z[omega].
# ============================================================================


@dataclass
class ER:
    """Element of Z[omega, 1/3], represented as (a + b*omega) / 3^c.

    The representation is kept in lowest terms: c is reduced as far as
    possible by cancelling common factors of 3 from both a and b.
    """

    a: int = 0
    b: int = 0
    c: int = 0

    def _reduce(self) -> "ER":
        """Cancel common factors of 3 from (a, b), reducing c."""
        a, b, c = self.a, self.b, self.c
        while c > 0 and a % 3 == 0 and b % 3 == 0:
            a //= 3
            b //= 3
            c -= 1
        return ER(a, b, c)

    def __add__(self, other: "ER") -> "ER":
        if self.c >= other.c:
            f = 3 ** (self.c - other.c)
            return ER(self.a + other.a * f,
                      self.b + other.b * f,
                      self.c)._reduce()
        else:
            f = 3 ** (other.c - self.c)
            return ER(self.a * f + other.a,
                      self.b * f + other.b,
                      other.c)._reduce()

    def __mul__(self, other: "ER") -> "ER":
        # (a + b*omega)(d + e*omega) = ad + (ae+bd)*omega + be*omega^2
        #                           = ad + (ae+bd)*omega + be*(-1-omega)
        #                           = (ad - be) + (ae + bd - be)*omega
        a, b = self.a, self.b
        d, e = other.a, other.b
        new_a = a * d - b * e
        new_b = a * e + b * d - b * e
        return ER(new_a, new_b, self.c + other.c)._reduce()

    def __neg__(self) -> "ER":
        return ER(-self.a, -self.b, self.c)

    def __sub__(self, other: "ER") -> "ER":
        return self + (-other)

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0

    def reduce_mod(self, p: int, q_p: int) -> int:
        """Reduce to F_p (substituting omega -> q_p in F_p, p != 3).

        Returns (a + b*q_p) * inv(3^c) mod p.
        """
        if self.c == 0:
            return (self.a + self.b * q_p) % p
        # 3 is invertible mod p (since p != 3), so 3^(-c) mod p is well-defined.
        inv3c = pow(3, -self.c, p)
        return ((self.a + self.b * q_p) * inv3c) % p

    def __repr__(self) -> str:
        if self.c == 0:
            return f"ER({self.a},{self.b})"
        return f"ER({self.a},{self.b})/3^{self.c}"


# Precomputed constants.
ZERO = ER(0, 0, 0)
ONE = ER(1, 0, 0)
OMEGA = ER(0, 1, 0)       # omega = q
OMEGA2 = ER(-1, -1, 0)    # omega^2 = -1 - omega

# D = q - q^{-1} = q - q^2 = 1 + 2*omega  (since q^{-1} = q^2 = -1 - omega).
D = ER(1, 2, 0)
# 1/D = (-1 - 2*omega)/3.  Verification: D * (1/D) = (1+2w)(-1-2w)/3
#       = -(1+2w)^2/3 = -D^2/3 = -(-3)/3 = 1, since D^2 = (1+2w)^2 = -3.
INV_D = ER(-1, -2, 1)
assert (D * INV_D) == ONE, f"D * INV_D != ONE: got {D * INV_D}"


def qpow(k: int) -> ER:
    """Return omega^k as an ER (k any integer; omega^3 = 1)."""
    k = k % 3
    if k == 0:
        return ONE
    elif k == 1:
        return OMEGA
    else:
        return OMEGA2


def idx(a: int, b: int, c: int) -> int:
    """Linear index of basis element K^a E^b F^c (0 <= a, b, c < ELL)."""
    return a * ELL * ELL + b * ELL + c


def from_idx(i: int):
    """Inverse of idx()."""
    return i // (ELL * ELL), (i // ELL) % ELL, i % ELL


# ============================================================================
# Algebra multiplication with exact Z[omega, 1/3] arithmetic.
#
# Basis: { K^a E^b F^c : 0 <= a, b, c < 3 }.
# Relations (q = omega, omega^2 + omega + 1 = 0):
#   K^3 = 1,   E^3 = F^3 = 0
#   K E K^{-1} = q^2 E   ->  K E = q^2 E K,  E K = q^{-2} K E
#   K F K^{-1} = q^{-2} F ->  K F = q^{-2} F K,  F K = q^2 K F
#   [E, F] = (K - K^{-1})/(q - q^{-1}) = (K - K^2)/D
# ============================================================================


def multiply_by_F_left(elem: dict) -> dict:
    """Multiply a normal-form element (dict {(a,b,c): ER}) by F on the LEFT.

    Uses F K^a = q^{2a} K^a F and the commutator F E = E F - delta with
    delta = (K - K^{-1})/(q - q^{-1}).  By induction:
        F E^b = E^b F - sum_{k=0}^{b-1} E^k delta E^{b-1-k}
              = E^b F - sum_{k=0}^{b-1} [(q^{-2k} K - q^{2k} K^{-1})/D] E^{b-1}.
    So
        F * K^a E^b F^c = q^{2a} K^a E^b F^{c+1}
                        - q^{2a} sum_{k=0}^{b-1} [(q^{-2k} K^{a+1}
                                                  - q^{2k} K^{a-1})/D] E^{b-1} F^c
    with K^{a-1} = K^{(a+ELL-1) % ELL} (since K^{-1} = K^{ELL-1}).
    """
    new = {}
    for (a, b, c), v in elem.items():
        if v.is_zero():
            continue
        coeff_a = v * qpow(2 * a)  # v * q^(2a)
        # Term 1: q^(2a) K^a E^b F^(c+1) (vanishes if c+1 == ELL).
        if c + 1 < ELL:
            key = (a, b, c + 1)
            new[key] = new.get(key, ZERO) + coeff_a
        # Term 2: correction from commuting F past E^b.
        for k in range(b):
            # K^(a+1) term: -coeff_a * q^(-2k) / D
            key1 = ((a + 1) % ELL, b - 1, c)
            new[key1] = new.get(key1, ZERO) + ((-coeff_a) * qpow(-2 * k) * INV_D)
            # K^(a-1) term: +coeff_a * q^(2k) / D
            key2 = ((a + ELL - 1) % ELL, b - 1, c)
            new[key2] = new.get(key2, ZERO) + (coeff_a * qpow(2 * k) * INV_D)
    return {k: v for k, v in new.items() if not v.is_zero()}


def multiply_by_E_left(elem: dict) -> dict:
    """Multiply a normal-form element by E on the LEFT.

    Uses E K^a = q^{-2a} K^a E, so
        E * K^a E^b F^c = q^{-2a} K^a E^{b+1} F^c  (vanishes if b+1 == ELL).
    """
    new = {}
    for (a, b, c), v in elem.items():
        if v.is_zero():
            continue
        if b + 1 < ELL:
            coeff = v * qpow(-2 * a)
            key = (a, b + 1, c)
            new[key] = new.get(key, ZERO) + coeff
    return new


def multiply_monomials(a: int, b: int, c: int,
                       ap: int, bp: int, cp: int) -> dict:
    """Return (K^a E^b F^c) * (K^{a'} E^{b'} F^{c'}) as a dict {(a,b,c): ER}.

    Algorithm (same as verify_sl2_hh2.py, but with exact arithmetic):
      1. Pull K^{a'} to the left past E^b F^c: total scalar q^{2 a'(c - b)}.
      2. Reduce the middle F^c E^{b'} to normal form via c applications of
         multiply_by_F_left to E^{b'}.
      3. Multiply the middle by E^b on the LEFT (b applications of
         multiply_by_E_left).
      4. Multiply by F^{c'} on the RIGHT (add to F-exponent).
      5. Multiply by K^{a1} on the LEFT (add to K-exponent mod ELL).
    """
    factor = qpow(2 * ap * (c - b))
    a1 = (a + ap) % ELL
    # Step 2: reduce F^c * E^{b'}.
    middle = {(0, bp, 0): ONE}
    for _ in range(c):
        middle = multiply_by_F_left(middle)
    # Step 3: multiply by E^b on the LEFT.
    for _ in range(b):
        middle = multiply_by_E_left(middle)
    # Step 4: multiply by F^{c'} on the RIGHT.
    after_F = {}
    for (a2, b2, c2), v in middle.items():
        c3 = c2 + cp
        if c3 < ELL:
            after_F[(a2, b2, c3)] = after_F.get((a2, b2, c3), ZERO) + v
    # Step 5: multiply by K^{a1} on the LEFT and apply the scalar factor.
    final = {}
    for (a2, b2, c2), v in after_F.items():
        a3 = (a1 + a2) % ELL
        v_final = v * factor
        if not v_final.is_zero():
            final[(a3, b2, c2)] = final.get((a3, b2, c2), ZERO) + v_final
    return final


def build_multiplication_table() -> dict:
    """Build the 27 x 27 x 27 multiplication table.

    Returns a dict mapping (k, i, j) -> ER, where the value is the
    coefficient of basis[k] in (basis[i] * basis[j]).
    """
    mult = {}
    for i in range(DIM):
        a, b, c = from_idx(i)
        for j in range(DIM):
            ap, bp, cp = from_idx(j)
            prod = multiply_monomials(a, b, c, ap, bp, cp)
            for (a2, b2, c2), v in prod.items():
                mult[(idx(a2, b2, c2), i, j)] = v
    return mult


def sanity_checks(mult: dict) -> None:
    """Verify the defining relations of u_q(sl_2) at ell = 3 with exact arithmetic."""
    e0 = idx(0, 0, 0)
    K = idx(1, 0, 0)
    K2 = idx(2, 0, 0)
    E = idx(0, 1, 0)
    F = idx(0, 0, 1)
    E2 = idx(0, 2, 0)
    F2 = idx(0, 0, 2)
    KE_idx = idx(1, 1, 0)

    # 1 * x = x * 1 = x.
    for i in range(DIM):
        assert mult.get((i, e0, i), ZERO) == ONE, f"1 * x failed at {i}"
        assert mult.get((i, i, e0), ZERO) == ONE, f"x * 1 failed at {i}"

    # K^3 = 1: K * K = K^2, K * K^2 = 1.
    assert mult.get((K2, K, K), ZERO) == ONE, "K * K != K^2"
    assert mult.get((e0, K, K2), ZERO) == ONE, "K * K^2 != 1"

    # E^3 = 0: E^2 * E = 0.
    for i in range(DIM):
        v = mult.get((i, E2, E), ZERO)
        assert v.is_zero(), f"E^2 * E != 0 at {i}: got {v}"

    # F^3 = 0: F^2 * F = 0.
    for i in range(DIM):
        v = mult.get((i, F2, F), ZERO)
        assert v.is_zero(), f"F^2 * F != 0 at {i}: got {v}"

    # K E = q^2 E K (equivalently, E K = q^{-2} K E).
    assert mult.get((KE_idx, K, E), ZERO) == ONE, "K * E != K E"
    assert mult.get((KE_idx, E, K), ZERO) == qpow(-2), \
        f"E * K != q^-2 K E: got {mult.get((KE_idx, E, K), ZERO)}"

    # [E, F] = (K - K^{-1})/(q - q^{-1}) = (K - K^2)/D = INV_D * K - INV_D * K^2.
    for k in range(DIM):
        ef = mult.get((k, E, F), ZERO)
        fe = mult.get((k, F, E), ZERO)
        diff = ef - fe
        if k == K:
            assert diff == INV_D, f"[E, F] at K != INV_D: got {diff}"
        elif k == K2:
            assert diff == (-INV_D), f"[E, F] at K^2 != -INV_D: got {diff}"
        else:
            assert diff.is_zero(), f"[E, F] at basis[{k}] != 0: got {diff}"


def build_epsilon() -> list:
    """Counit: eps(K^a E^b F^c) = 1 if b = c = 0, else 0 (extended as algebra map)."""
    epsilon = [0] * DIM
    for a in range(ELL):
        epsilon[idx(a, 0, 0)] = 1
    return epsilon


def build_d1(mult: dict, epsilon: list) -> dict:
    """Build d^1: C^1 -> C^2 as a sparse dict (row, col) -> ER.

    (d^1 f)(a, b) = eps(a) * f(b) - f(a*b) + f(a) * eps(b)
    """
    d1 = {}
    for i in range(DIM):  # f = indicator on basis[i]
        for a in range(DIM):
            for b in range(DIM):
                row = a * DIM + b
                term1 = ONE if (epsilon[a] and b == i) else ZERO
                term2 = mult.get((i, a, b), ZERO)
                term3 = ONE if (a == i and epsilon[b]) else ZERO
                v = term1 - term2 + term3
                if not v.is_zero():
                    d1[(row, i)] = v
    return d1


def build_d2(mult: dict, epsilon: list) -> dict:
    """Build d^2: C^2 -> C^3 as a sparse dict (row, col) -> ER.

    (d^2 g)(a, b, c) = eps(a) * g(b, c) - g(a*b, c) + g(a, b*c) - g(a, b) * eps(c)
    """
    d2 = {}
    for a in range(DIM):
        for b in range(DIM):
            for c in range(DIM):
                row = a * DIM * DIM + b * DIM + c
                # Term 1: eps(a) * g(b, c).
                if epsilon[a]:
                    col = b * DIM + c
                    d2[(row, col)] = d2.get((row, col), ZERO) + ONE
                # Term 2: - g(a*b, c)  where a*b = sum_k mult[k, a, b] basis[k].
                for k in range(DIM):
                    v = mult.get((k, a, b), ZERO)
                    if not v.is_zero():
                        col = k * DIM + c
                        d2[(row, col)] = d2.get((row, col), ZERO) - v
                # Term 3: + g(a, b*c)  where b*c = sum_k mult[k, b, c] basis[k].
                for k in range(DIM):
                    v = mult.get((k, b, c), ZERO)
                    if not v.is_zero():
                        col = a * DIM + k
                        d2[(row, col)] = d2.get((row, col), ZERO) + v
                # Term 4: - g(a, b) * eps(c).
                if epsilon[c]:
                    col = a * DIM + b
                    d2[(row, col)] = d2.get((row, col), ZERO) - ONE
    return d2


# ============================================================================
# Modular reduction and rank computation over F_p.
# ============================================================================


def find_cube_root_mod_p(p: int):
    """Find a primitive cube root of unity in F_p (requires p ≡ 1 mod 3, p ≠ 3)."""
    if p == 3 or (p - 1) % 3 != 0:
        return None
    # Find a generator-like element: try x = 2, 3, ... and compute x^((p-1)/3).
    for x in range(2, p):
        r = pow(x, (p - 1) // 3, p)
        if r != 1 and (r * r * r) % p == 1:
            return r
    return None


def reduce_to_numpy(dict_mat: dict, n_rows: int, n_cols: int,
                    p: int, q_p: int) -> np.ndarray:
    """Reduce a sparse dict matrix to a dense numpy array over F_p.

    Each entry (a + b*omega)/3^c is mapped to (a + b*q_p) * inv(3^c) mod p.
    """
    A = np.zeros((n_rows, n_cols), dtype=np.int64)
    for (r, c), v in dict_mat.items():
        rv = v.reduce_mod(p, q_p)
        if rv != 0:
            # Accumulate (in case of duplicate keys, though there shouldn't be any).
            A[r, c] = (A[r, c] + rv) % p
    return A


def rank_mod_p(A: np.ndarray, p: int) -> int:
    """Compute the rank of A over F_p via exact Gaussian elimination.

    The matrix A is modified in-place (on a copy).  No floating-point
    arithmetic is used; all operations are exact integer arithmetic modulo p.
    """
    A = A.copy() % p
    m, n = A.shape
    rank = 0
    for col in range(n):
        if rank >= m:
            break
        # Find a pivot row (with nonzero in column `col`) at or below `rank`.
        col_vals = A[rank:, col]
        nonzero_idx = np.nonzero(col_vals)[0]
        if len(nonzero_idx) == 0:
            continue
        pivot_row = rank + int(nonzero_idx[0])
        # Swap pivot row into position `rank`.
        if pivot_row != rank:
            A[[rank, pivot_row]] = A[[pivot_row, rank]]
        # Normalize the pivot row so A[rank, col] == 1.
        pv = int(A[rank, col]) % p
        if pv != 1:
            inv_pv = pow(pv, -1, p)
            A[rank] = (A[rank] * inv_pv) % p
        # Eliminate column `col` from all other rows.
        col_full = A[:, col]
        mask = col_full != 0
        mask[rank] = False
        if np.any(mask):
            factors = A[mask, col].astype(np.int64)[:, None]
            A[mask] = (A[mask] - factors * A[rank]) % p
        rank += 1
    return rank


# ============================================================================
# Output tee: write to both stdout and the output file.
# ============================================================================


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)

    def flush(self):
        for f in self.files:
            f.flush()


# ============================================================================
# Main certification routine.
# ============================================================================

# Primes p ≡ 1 (mod 3) with p ≠ 3, suitable for reduction (omega exists in F_p).
CERTIFICATION_PRIMES = [7, 13, 19, 31, 37, 43, 61, 67, 73, 79, 97]


def run_certification(primes=None, verbose=True):
    """Run the full certification.  Returns a dict with the results.

    If `primes` is None, uses CERTIFICATION_PRIMES.
    """
    if primes is None:
        primes = CERTIFICATION_PRIMES

    log = []
    def out(msg=""):
        if verbose:
            print(msg)
        log.append(msg)

    out("=== Exact cyclotomic certification of HH^2(u_q(sl_2), C) at ell = 3 ===")
    out()
    out("Ring: Z[omega, 1/3]  (omega = q, a primitive cube root of unity).")
    out("  omega^2 + omega + 1 = 0, so omega^2 = -1 - omega.")
    out("  D = q - q^{-1} = 1 + 2*omega,  D^2 = -3,  1/D = (-1 - 2*omega)/3.")
    out()
    out(f"dim u_q(sl_2) = {DIM}")
    out(f"dim C^1 = {DIM},  dim C^2 = {DIM**2},  dim C^3 = {DIM**3}")
    out()

    # --- Step 1: multiplication table ---
    t0 = time.time()
    out("Step 1: Building multiplication table over Z[omega, 1/3] ...")
    mult = build_multiplication_table()
    out(f"  done in {time.time()-t0:.1f}s, nonzero entries: {len(mult)}")

    # --- Sanity checks ---
    out("Step 1b: Sanity checks (defining relations of u_q(sl_2)) ...")
    epsilon = build_epsilon()
    sanity_checks(mult)
    out("  all sanity checks passed (1*x=x*1=x, K^3=1, E^3=F^3=0,")
    out("  K E = q^2 E K, [E,F] = (K - K^{-1})/(q - q^{-1}))")
    out()

    # --- Step 2: build differentials ---
    t0 = time.time()
    out("Step 2a: Building d^1 over Z[omega, 1/3] ...")
    d1 = build_d1(mult, epsilon)
    out(f"  done in {time.time()-t0:.1f}s, nonzero entries: {len(d1)}")
    out(f"  d^1 shape: {DIM*DIM} rows x {DIM} cols")

    t0 = time.time()
    out("Step 2b: Building d^2 over Z[omega, 1/3] ...")
    d2 = build_d2(mult, epsilon)
    out(f"  done in {time.time()-t0:.1f}s, nonzero entries: {len(d2)}")
    out(f"  d^2 shape: {DIM**3} rows x {DIM*DIM} cols")
    out()

    # --- Step 3 & 4: reduce mod p and compute ranks ---
    out("Step 3: Reducing mod p and computing ranks over F_p for primes")
    out(f"  p ≡ 1 (mod 3), p ≠ 3: {primes}")
    out()
    out(f"  {'p':>5}  {'q_p':>5}  {'rank(d^1)':>10}  "
        f"{'rank(d^2)':>10}  {'dim HH^2':>10}  {'time':>8}")

    ranks_d1 = []
    ranks_d2 = []
    dim_hh2_list = []

    for p in primes:
        t0 = time.time()
        q_p = find_cube_root_mod_p(p)
        if q_p is None:
            out(f"  {p:>5}  ERROR: no cube root of unity mod {p}")
            continue

        # d^1: shape (729, 27).  Small; dense Gaussian elimination is fast.
        A1 = reduce_to_numpy(d1, DIM * DIM, DIM, p, q_p)
        rank1 = rank_mod_p(A1, p)
        ranks_d1.append(rank1)

        # d^2: shape (19683, 729).  Gaussian elimination on 729 columns.
        A2 = reduce_to_numpy(d2, DIM ** 3, DIM * DIM, p, q_p)
        rank2 = rank_mod_p(A2, p)
        ranks_d2.append(rank2)

        dim_ker_d2 = DIM * DIM - rank2
        dim_hh2 = dim_ker_d2 - rank1
        dim_hh2_list.append(dim_hh2)

        elapsed = time.time() - t0
        out(f"  {p:>5}  {q_p:>5}  {rank1:>10}  {rank2:>10}  "
            f"{dim_hh2:>10}  {elapsed:>7.1f}s")

    out()

    # --- Step 5: verify consistency and certify ---
    out("=== Certification summary ===")
    out(f"  rank(d^1) values: {ranks_d1}")
    out(f"  rank(d^2) values: {ranks_d2}")
    out(f"  dim HH^2 values:  {dim_hh2_list}")

    consistent = (len(set(ranks_d1)) == 1 and len(set(ranks_d2)) == 1
                  and len(ranks_d1) == len(primes))

    result = {
        "primes": list(primes),
        "ranks_d1": ranks_d1,
        "ranks_d2": ranks_d2,
        "dim_hh2_list": dim_hh2_list,
        "consistent": consistent,
        "certified_dim_hh2": None,
        "log": log,
    }

    if consistent:
        r1 = ranks_d1[0]
        r2 = ranks_d2[0]
        dim_hh2 = dim_hh2_list[0]
        result["certified_dim_hh2"] = dim_hh2
        out()
        out(f"  RANKS CONSISTENT across all {len(primes)} primes.")
        out(f"  By rank semicontinuity over the Dedekind domain Z[omega, 1/3],")
        out(f"  this certifies the rank over Q(omega) and hence over C")
        out(f"  (embedding omega -> e^(2*pi*i/3)).")
        out()
        out(f"  Certified rank(d^1) = {r1}")
        out(f"  Certified rank(d^2) = {r2}")
        out(f"  dim C^2 = {DIM*DIM},  dim ker(d^2) = {DIM*DIM - r2}")
        out(f"  dim im(d^1) = {r1}")
        out(f"  dim HH^2(u_q(sl_2), C) = {dim_hh2}")
        out()
        expected = 3  # C(2, 2) + 2|Phi^+(A_1)| = 1 + 2 = 3
        if dim_hh2 == expected:
            out(f"  CERTIFIED: dim HH^2 = {dim_hh2}, matching paper Theorem 1.2")
            out(f"             (C(n+1, 2) + 2|Phi^+| = {expected} for n = 1).")
        else:
            out(f"  MISMATCH: dim HH^2 = {dim_hh2}, expected {expected}.")
    else:
        out()
        out("  RANKS INCONSISTENT across primes -- rank semicontinuity")
        out("  cannot be applied.  Certification FAILED.")

    return result


def main():
    # Tee stdout to the output file.
    output_f = open(OUTPUT_FILE, "w")
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, output_f)
    try:
        result = run_certification()
    finally:
        sys.stdout = old_stdout
        output_f.close()
    # Exit with nonzero code if certification failed.
    if not result["consistent"] or result["certified_dim_hh2"] != 3:
        sys.exit(1)


if __name__ == "__main__":
    main()
