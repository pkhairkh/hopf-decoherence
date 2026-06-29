"""
Quantum algebra module for u_q(sl_n) at roots of unity.

Canonical Weyl module constructions with:
  - Highest-weight-first basis ordering
  - Decreasing K-eigenvalue order
  - Balanced sqrt-normalization for E/F coefficients

Supports:
  - sl_2: L(j) for arbitrary j
  - sl_3: L(a,b) for (a,b) in {(0,0), (1,0), (0,1), (2,0), (0,2), (1,1)}
"""

import numpy as np
from typing import Tuple, Union

# Rank computation tolerance
RANK_TOL = 1e-7

# Generic q (far from any low-order root of unity)
Q_GENERIC = np.exp(1.7j)


# ============================================================
# q-Number Utilities
# ============================================================

def q_number(n: int, q: complex) -> complex:
    """Compute [n]_q = (q^n - q^{-n}) / (q - q^{-1}).

    At q = 1 (limiting case), returns n.
    """
    if n == 0:
        return 0.0 + 0j
    denom = q - q ** (-1)
    if abs(denom) < 1e-14:
        return float(n) + 0j
    return (q ** n - q ** (-n)) / denom


def q_factorial(n: int, q: complex) -> complex:
    """Compute [n]_q! = [1]_q [2]_q ... [n]_q."""
    result = 1.0 + 0j
    for k in range(1, n + 1):
        result *= q_number(k, q)
    return result


def q_binomial(n: int, k: int, q: complex) -> complex:
    """Compute the q-binomial coefficient [n choose k]_q."""
    if k < 0 or k > n:
        return 0.0 + 0j
    if k == 0 or k == n:
        return 1.0 + 0j
    denom = q_factorial(k, q) * q_factorial(n - k, q)
    if abs(denom) < 1e-14:
        return 0.0 + 0j
    return q_factorial(n, q) / denom


# ============================================================
# sl_2 Weyl Module Construction
# ============================================================

def build_weyl_module_sl2(j: int, q: complex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build L(j) for u_q(sl_2) with canonical normalization.

    Basis: |j>, |j-2>, ..., |-j>  (decreasing weight)
    Basis index m = 0, 1, ..., j corresponds to weight j - 2m.

    Generators:
        K |j-2m> = q^{j-2m} |j-2m>
        F |j-2m> = sqrt([m+1]_q [j-m]_q) |j-2(m+1)>   (lowering)
        E |j-2m> = sqrt([m]_q [j-m+1]_q) |j-2(m-1)>   (raising)

    Parameters
    ----------
    j : int
        Highest weight (dimension = j + 1).
    q : complex
        Deformation parameter.

    Returns
    -------
    K, E, F : np.ndarray of shape (j+1, j+1)
    """
    dim = j + 1
    K = np.zeros((dim, dim), dtype=complex)
    E = np.zeros((dim, dim), dtype=complex)
    F = np.zeros((dim, dim), dtype=complex)

    for m in range(dim):
        # Weight = j - 2*m
        K[m, m] = q ** (j - 2 * m)

        # F: lowering. F|m> -> |m+1>
        if m + 1 < dim:
            val = q_number(m + 1, q) * q_number(j - m, q)
            F[m + 1, m] = np.sqrt(val) if abs(val) > 1e-30 else 0.0

        # E: raising. E|m> -> |m-1>
        if m - 1 >= 0:
            val = q_number(m, q) * q_number(j - m + 1, q)
            E[m - 1, m] = np.sqrt(val) if abs(val) > 1e-30 else 0.0

    return K, E, F


# ============================================================
# sl_3 Weyl Module Construction
# ============================================================

def _build_sl3_L00(q: complex):
    """L(0,0) -- trivial 1-dimensional representation."""
    dim = 1
    K1 = np.eye(dim, dtype=complex)
    K2 = np.eye(dim, dtype=complex)
    E1 = np.zeros((dim, dim), dtype=complex)
    E2 = np.zeros((dim, dim), dtype=complex)
    F1 = np.zeros((dim, dim), dtype=complex)
    F2 = np.zeros((dim, dim), dtype=complex)
    return K1, K2, E1, E2, F1, F2


def _build_sl3_L10(q: complex):
    """L(1,0) = L(omega_1) -- fundamental 3-dimensional representation.

    Canonical basis (highest-weight-first):
      v0: weight (1,0)  -- highest weight
      v1: weight (-1,1)
      v2: weight (0,-1)
    """
    dim = 3
    K1 = np.diag([q, q ** (-1), 1.0 + 0j])
    K2 = np.diag([1.0 + 0j, q, q ** (-1)])

    E1 = np.zeros((dim, dim), dtype=complex)
    E1[0, 1] = 1.0

    E2 = np.zeros((dim, dim), dtype=complex)
    E2[1, 2] = 1.0

    F1 = np.zeros((dim, dim), dtype=complex)
    F1[1, 0] = 1.0

    F2 = np.zeros((dim, dim), dtype=complex)
    F2[2, 1] = 1.0

    return K1, K2, E1, E2, F1, F2


def _build_sl3_L01(q: complex):
    """L(0,1) = L(omega_2) -- dual fundamental 3-dimensional representation.

    Canonical basis (highest-weight-first):
      w0: weight (0,1)  -- highest weight
      w1: weight (1,-1)
      w2: weight (-1,0)
    """
    dim = 3
    K1 = np.diag([1.0 + 0j, q, q ** (-1)])
    K2 = np.diag([q, q ** (-1), 1.0 + 0j])

    E1 = np.zeros((dim, dim), dtype=complex)
    E1[1, 2] = 1.0

    E2 = np.zeros((dim, dim), dtype=complex)
    E2[0, 1] = 1.0

    F1 = np.zeros((dim, dim), dtype=complex)
    F1[2, 1] = 1.0

    F2 = np.zeros((dim, dim), dtype=complex)
    F2[1, 0] = 1.0

    return K1, K2, E1, E2, F1, F2


def _build_sl3_from_tensor_product(q, rep1, rep2, target_weight):
    """Build an sl_3 CG component from a tensor product via HWV generation.

    Constructs L(target_weight) as a summand of rep1 tensor rep2 by:
    1. Building the coproduct action on rep1 tensor rep2
    2. Finding the highest-weight vector with the target weight
    3. Generating the full module by applying F1, F2

    Parameters
    ----------
    q : complex
    rep1, rep2 : tuple of (K1, K2, E1, E2, F1, F2)
    target_weight : tuple (a, b) -- Dynkin labels of the target component
    """
    from .coproduct import coproduct_matrices_sl3

    K1_a, K2_a, E1_a, E2_a, F1_a, F2_a = rep1
    K1_b, K2_b, E1_b, E2_b, F1_b, F2_b = rep2

    dim_a, dim_b = K1_a.shape[0], K1_b.shape[0]
    tensor_dim = dim_a * dim_b

    dK1, dK2, dE1, dE2, dF1, dF2, dE12, dF12 = coproduct_matrices_sl3(
        K1_a, K2_a, E1_a, E2_a, F1_a, F2_a, q, dim_a
    )
    # We need the cross-representation coproduct, not self-coproduct
    # For rep1 tensor rep2, coproduct acts as:
    I_a = np.eye(dim_a, dtype=complex)
    I_b = np.eye(dim_b, dtype=complex)
    K1_ainv = np.linalg.inv(K1_a)
    K2_ainv = np.linalg.inv(K2_a)

    dE1 = np.kron(E1_a, I_b) + np.kron(K1_a, E1_b)
    dE2 = np.kron(E2_a, I_b) + np.kron(K2_a, E2_b)
    dF1 = np.kron(F1_a, K1_ainv) + np.kron(I_a, F1_b)  # wrong: should use K2_b inv
    dF1 = np.kron(F1_a, np.linalg.inv(K1_b)) + np.kron(I_a, F1_b)
    dF2 = np.kron(F2_a, np.linalg.inv(K2_b)) + np.kron(I_a, F2_b)
    dK1 = np.kron(K1_a, K1_b)
    dK2 = np.kron(K2_a, K2_b)

    # Weyl dimension formula for target
    a_t, b_t = target_weight
    target_dim = (a_t + 1) * (b_t + 1) * (a_t + b_t + 2) // 2

    # Find HWV with the target weight
    target_K1_eig = q ** a_t
    target_K2_eig = q ** b_t

    hw_vecs = []
    for i in range(tensor_dim):
        if (abs(dK1[i, i] - target_K1_eig) < 1e-8 and
                abs(dK2[i, i] - target_K2_eig) < 1e-8):
            v = np.zeros(tensor_dim, dtype=complex)
            v[i] = 1.0
            e1_v = dE1 @ v
            e2_v = dE2 @ v
            if np.linalg.norm(e1_v) < 1e-8 and np.linalg.norm(e2_v) < 1e-8:
                hw_vecs.append(v)

    if not hw_vecs:
        raise ValueError(f"No HWV with weight {target_weight} found in tensor product.")

    # Generate module from first HWV
    hwv = hw_vecs[0]
    basis = [hwv.copy()]
    changed = True
    while changed:
        changed = False
        new_basis = []
        for v in basis:
            for dF in [dF1, dF2]:
                w = dF @ v
                if np.linalg.norm(w) < 1e-10:
                    continue
                for b in basis + new_basis:
                    w -= np.dot(b.conj(), w) * b
                if np.linalg.norm(w) > 1e-10:
                    w /= np.linalg.norm(w)
                    new_basis.append(w)
                    changed = True
        basis.extend(new_basis)

    P = np.column_stack(basis)

    K1 = P.conj().T @ dK1 @ P
    K2 = P.conj().T @ dK2 @ P
    E1 = P.conj().T @ dE1 @ P
    E2 = P.conj().T @ dE2 @ P
    F1 = P.conj().T @ dF1 @ P
    F2 = P.conj().T @ dF2 @ P

    return K1, K2, E1, E2, F1, F2


def _build_sl3_L20(q):
    """L(2,0) -- 6-dimensional representation from L(1,0) tensor L(1,0)."""
    rep1 = _build_sl3_L10(q)
    rep2 = _build_sl3_L10(q)
    return _build_sl3_from_tensor_product(q, rep1, rep2, (2, 0))


def _build_sl3_L02(q):
    """L(0,2) -- 6-dimensional dual representation from L(0,1) tensor L(0,1)."""
    rep1 = _build_sl3_L01(q)
    rep2 = _build_sl3_L01(q)
    return _build_sl3_from_tensor_product(q, rep1, rep2, (0, 2))


def _build_sl3_L11(q):
    """L(1,1) -- 8-dimensional adjoint-type from L(1,0) tensor L(0,1)."""
    rep1 = _build_sl3_L10(q)
    rep2 = _build_sl3_L01(q)
    return _build_sl3_from_tensor_product(q, rep1, rep2, (1, 1))


def build_weyl_module(
    n: int,
    lam: Union[int, Tuple[int, ...]],
    q: complex,
):
    """Build the Weyl module L(lam) for u_q(sl_n).

    Parameters
    ----------
    n : int
        Rank of sl_n (2 for sl_2, 3 for sl_3).
    lam : int or tuple of int
        For n=2: integer j (highest weight, dim = j+1).
        For n=3: tuple (a, b) of Dynkin labels.
    q : complex
        Deformation parameter.

    Returns
    -------
    For n=2:  (K, E, F) each of shape (j+1, j+1)
    For n=3:  (K1, K2, E1, E2, F1, F2)
    """
    if n == 2:
        return build_weyl_module_sl2(lam, q)
    elif n == 3:
        a, b = lam[0], lam[1]
        builders = {
            (0, 0): _build_sl3_L00,
            (1, 0): _build_sl3_L10,
            (0, 1): _build_sl3_L01,
            (2, 0): _build_sl3_L20,
            (0, 2): _build_sl3_L02,
            (1, 1): _build_sl3_L11,
        }
        if (a, b) not in builders:
            raise NotImplementedError(f"L({a},{b}) for sl_3 not yet implemented.")
        return builders[(a, b)](q)
    else:
        raise NotImplementedError(f"sl_{n} not yet supported.")


# ============================================================
# Algebra Verification
# ============================================================

def verify_algebra_relations(
    K, E, F, q, tol=1e-8
) -> Tuple[bool, Tuple[float, float, float]]:
    """Verify u_q(sl_2) relations: KE=q^2*EK, KF=q^{-2}*FK, [E,F]=(K-K^{-1})/(q-q^{-1}).

    Returns (ok, (err_KE, err_KF, err_EF)).
    """
    K_inv = np.linalg.inv(K)
    err1 = np.max(np.abs(K @ E - q ** 2 * E @ K))
    err2 = np.max(np.abs(K @ F - q ** (-2) * F @ K))
    comm = E @ F - F @ E
    expected = (K - K_inv) / (q - q ** (-1))
    err3 = np.max(np.abs(comm - expected))
    return err1 < tol and err2 < tol and err3 < tol, (err1, err2, err3)


def verify_sl3_relations(
    K1, K2, E1, E2, F1, F2, q, tol=1e-8
) -> Tuple[bool, dict]:
    """Verify u_q(sl_3) relations on a representation.

    Checks K-E/F commutation, [E_i, F_j] relations, and quantum Serre relations.
    """
    K1inv = np.linalg.inv(K1)
    K2inv = np.linalg.inv(K2)
    q2 = q + q ** (-1)  # [2]_q

    errs = {}
    # K-E commutation
    errs['K1E1'] = np.max(np.abs(K1 @ E1 - q ** 2 * E1 @ K1))
    errs['K1E2'] = np.max(np.abs(K1 @ E2 - q ** (-1) * E2 @ K1))
    errs['K2E1'] = np.max(np.abs(K2 @ E1 - q ** (-1) * E1 @ K2))
    errs['K2E2'] = np.max(np.abs(K2 @ E2 - q ** 2 * E2 @ K2))
    # K-F commutation
    errs['K1F1'] = np.max(np.abs(K1 @ F1 - q ** (-2) * F1 @ K1))
    errs['K1F2'] = np.max(np.abs(K1 @ F2 - q * F2 @ K1))
    errs['K2F1'] = np.max(np.abs(K2 @ F1 - q * F1 @ K2))
    errs['K2F2'] = np.max(np.abs(K2 @ F2 - q ** (-2) * F2 @ K2))
    # E-F relations
    errs['EF1'] = np.max(np.abs(E1 @ F1 - F1 @ E1 - (K1 - K1inv) / (q - q ** (-1))))
    errs['EF2'] = np.max(np.abs(E2 @ F2 - F2 @ E2 - (K2 - K2inv) / (q - q ** (-1))))
    errs['E1F2'] = np.max(np.abs(E1 @ F2 - F2 @ E1))
    errs['E2F1'] = np.max(np.abs(E2 @ F1 - F1 @ E2))
    # Quantum Serre relations
    errs['Serre1'] = np.max(np.abs(E1 @ E1 @ E2 - q2 * E1 @ E2 @ E1 + E2 @ E1 @ E1))
    errs['Serre2'] = np.max(np.abs(E2 @ E2 @ E1 - q2 * E2 @ E1 @ E2 + E1 @ E2 @ E2))

    ok = all(v < tol for v in errs.values())
    return ok, errs


# ============================================================
# Rank Computation
# ============================================================

def compute_rank(M: np.ndarray, tol: float = RANK_TOL) -> int:
    """Compute numerical rank via SVD."""
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol))
