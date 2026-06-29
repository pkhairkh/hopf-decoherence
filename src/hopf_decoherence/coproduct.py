"""
Coproduct representation matrices for u_q(sl_n).

Computes the action of Delta: u_q -> u_q tensor u_q on tensor product
representations, and the rank of the representation Phi: u_q -> End(V tensor V).

Key formula:
  For u_q(sl_2) at root of unity q = exp(2*pi*i/ell),
  rank(Phi on St tensor St) = ell^3 - D_2(ell) where D_2(ell) = (ell^3 - ell)/6.
"""

import numpy as np
from .q_algebra import (
    build_weyl_module_sl2,
    q_number,
    compute_rank,
    RANK_TOL,
    Q_GENERIC,
)


def coproduct_matrices_sl2(K, E, F, dim=None):
    """Build coproduct matrices Delta(K), Delta(E), Delta(F) on V tensor V for sl_2.

    Coproduct:
        Delta(E) = E tensor 1 + K tensor E
        Delta(F) = F tensor K^{-1} + 1 tensor F
        Delta(K) = K tensor K

    Parameters
    ----------
    K, E, F : np.ndarray
        Representation matrices on V.
    dim : int, optional
        Dimension of V. Inferred from K if not given.

    Returns
    -------
    dK, dE, dF : np.ndarray on V tensor V
    """
    if dim is None:
        dim = K.shape[0]
    I = np.eye(dim, dtype=complex)
    K_inv = np.linalg.inv(K)
    dE = np.kron(E, I) + np.kron(K, E)
    dF = np.kron(F, K_inv) + np.kron(I, F)
    dK = np.kron(K, K)
    return dK, dE, dF


def coproduct_matrices_sl3(K1, K2, E1, E2, F1, F2, q, dim=None):
    """Build coproduct matrices on V tensor V for sl_3.

    Returns dK1, dK2, dE1, dE2, dF1, dF2, dE12, dF12 where
    E12 = E1*E2 - q*E2*E1 and F12 = F2*F1 - q^{-1}*F1*F2 are non-simple root vectors.
    """
    if dim is None:
        dim = K1.shape[0]
    I = np.eye(dim, dtype=complex)
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    dE1 = np.kron(E1, I) + np.kron(K1, E1)
    dE2 = np.kron(E2, I) + np.kron(K2, E2)
    dF1 = np.kron(F1, K1_inv) + np.kron(I, F1)
    dF2 = np.kron(F2, K2_inv) + np.kron(I, F2)
    dK1 = np.kron(K1, K1)
    dK2 = np.kron(K2, K2)

    # Non-simple root vectors
    dE12 = dE1 @ dE2 - q * dE2 @ dE1
    dF12 = dF2 @ dF1 - q ** (-1) * dF1 @ dF2

    return dK1, dK2, dE1, dE2, dF1, dF2, dE12, dF12


def compute_phi_rank(j, q, ell, tol=RANK_TOL):
    """Compute rank of Phi: u_q(sl_2) -> End(V tensor V).

    V = Weyl module L(j) with highest weight j (dim = j+1).
    The map Phi sends each PBW basis element E^a K^c F^b to its
    coproduct action on V tensor V.

    For the Steinberg module at ell-th root of unity, use j = ell-1.

    Returns (rank, algebra_dim, tensor_dim, phi_matrix).
    """
    K, E, F = build_weyl_module_sl2(j, q)
    dim = j + 1  # dim(L(j)) = j + 1
    dK, dE, dF = coproduct_matrices_sl2(K, E, F, dim)

    tensor_dim = dim ** 2

    # Precompute powers of coproduct generators
    dE_pow = [np.eye(tensor_dim, dtype=complex)]
    for a in range(1, ell):
        dE_pow.append(dE_pow[-1] @ dE)

    dF_pow = [np.eye(tensor_dim, dtype=complex)]
    for b in range(1, ell):
        dF_pow.append(dF_pow[-1] @ dF)

    dK_pow = [np.eye(tensor_dim, dtype=complex)]
    for c in range(1, ell):
        dK_pow.append(dK_pow[-1] @ dK)

    # Collect all image vectors (PBW basis: E^a K^c F^b)
    num_basis = ell ** 3
    all_vecs = np.zeros((tensor_dim ** 2, num_basis), dtype=complex)

    idx = 0
    for a in range(ell):
        Ea = dE_pow[a]
        for b in range(ell):
            Fb = dF_pow[b]
            for c in range(ell):
                mat = Ea @ dK_pow[c] @ Fb
                all_vecs[:, idx] = mat.ravel()
                idx += 1

    rank = compute_rank(all_vecs, tol)
    return rank, num_basis, tensor_dim, all_vecs


def compute_cross_rep_coproduct_rank(k1, k2, q, ell, tol=RANK_TOL):
    """Compute the rank of Phi_{k1,k2} = (rho_{k1} tensor rho_{k2}) circ Delta.

    Uses DIFFERENT representations on the two tensor factors.
    """
    K1, E1, F1 = build_weyl_module_sl2(k1, q)
    K2, E2, F2 = build_weyl_module_sl2(k2, q)
    dim1, dim2 = k1 + 1, k2 + 1

    I1 = np.eye(dim1, dtype=complex)
    I2 = np.eye(dim2, dtype=complex)
    K2_inv = np.linalg.inv(K2)

    dE = np.kron(E1, I2) + np.kron(K1, E2)
    dF = np.kron(F1, K2_inv) + np.kron(I1, F2)
    dK = np.kron(K1, K2)

    tensor_dim = dim1 * dim2
    num_basis = ell ** 3

    dE_pow = [np.eye(tensor_dim, dtype=complex)]
    for a in range(1, ell):
        dE_pow.append(dE_pow[-1] @ dE)

    dF_pow = [np.eye(tensor_dim, dtype=complex)]
    for b in range(1, ell):
        dF_pow.append(dF_pow[-1] @ dF)

    dK_pow = [np.eye(tensor_dim, dtype=complex)]
    for c in range(1, ell):
        dK_pow.append(dK_pow[-1] @ dK)

    all_vecs = np.zeros((tensor_dim ** 2, num_basis), dtype=complex)
    idx = 0
    for a in range(ell):
        for b in range(ell):
            for c in range(ell):
                mat = dE_pow[a] @ dK_pow[c] @ dF_pow[b]
                all_vecs[:, idx] = mat.ravel()
                idx += 1

    return compute_rank(all_vecs, tol)


def compute_sl3_coproduct_rank_algebra_closure(K1, K2, E1, E2, F1, F2, q,
                                                 max_iter=60, tol=1e-10):
    """Compute coproduct rank on V tensor V for sl_3 using algebra closure.

    More efficient than PBW enumeration for large algebra dimensions.

    Returns the dimension of the subalgebra generated by the coproduct images
    acting on End(V tensor V).
    """
    dim = K1.shape[0]
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
    gens = [np.eye(dim_rr, dtype=complex), dE1, dE2, dE12, dK1, dK2, dF1, dF2, dF12]

    basis = []
    for g in gens:
        v = g.ravel()
        for bv in basis:
            v -= np.vdot(bv, v) / np.vdot(bv, bv) * bv
        if np.linalg.norm(v) > tol:
            basis.append(v / np.linalg.norm(v))

    for _ in range(max_iter):
        new = []
        old = len(basis)
        for bv in list(basis):
            bm = bv.reshape(dim_rr, dim_rr)
            for g in gens:
                v = (g @ bm).ravel()
                for ex in basis + new:
                    v -= np.vdot(ex, v) / np.vdot(ex, ex) * ex
                if np.linalg.norm(v) > tol:
                    new.append(v / np.linalg.norm(v))
        basis.extend(new)
        if len(basis) == old:
            break

    return len(basis)
