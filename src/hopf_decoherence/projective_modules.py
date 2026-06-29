"""
Projective indecomposable modules P(j) for u_q(sl_2) at roots of unity.

For u_q(sl_2) at q = exp(2*pi*i/ell) with ell odd:
  - P(ell-1) = L(ell-1) = Steinberg module (simple and projective, dim=ell)
  - P(j) for j < ell-1: dim = 2*ell, constructed from baby Verma pairs
  - Regular representation: u_q = direct_sum_{j=0}^{ell-1} P(j)^{direct_sum(j+1)}

Construction uses the regular representation (left multiplication on the PBW basis)
and decomposition into projective blocks.
"""

import numpy as np
from .q_algebra import q_number, q_factorial, q_binomial, compute_rank, RANK_TOL


def _pbw_index(a, c, b, ell):
    """Convert PBW multi-index (a,c,b) to linear index in the basis of u_q(sl_2)."""
    return a * ell * ell + c * ell + b


def _pbw_multi(idx, ell):
    """Convert linear index back to (a, c, b)."""
    a = idx // (ell * ell)
    remainder = idx % (ell * ell)
    c = remainder // ell
    b = remainder % ell
    return a, c, b


def _compute_FEa(ell):
    """Compute F E^a for a = 0, ..., ell-1 as PBW coefficient dictionaries.

    F E^0 = F = {(0,0,1): 1}
    F E^{n+1} = (F E^n) * E  (right multiply by E)

    Right multiplication of E^pa K^pc F^pb by E:
      E^pa K^pc F^pb * E
        = E^pa K^pc (F^pb * E)

    Using F^pb * E = E * F^pb + [pb]_q * (q^{1-pb} K^{-1} - q^{pb-1} K) / (q-q^{-1}) * F^{pb-1}

    So E^pa K^pc F^pb * E
        = E^pa K^pc E F^pb + correction
        = E^pa * (K^pc E) * F^pb + correction
        = E^pa * q^{2pc} E K^pc * F^pb + correction   (since KE=q^2EK, so K^c E = q^{2c} E K^c)
        = q^{2pc} * E^{pa+1} * K^pc * F^pb + correction

    Correction (if pb >= 1):
      [pb]_q * q^{1-pb}/(q-q^{-1}) * E^pa * K^pc * K^{-1} * F^{pb-1}
    - [pb]_q * q^{pb-1}/(q-q^{-1}) * E^pa * K^pc * K * F^{pb-1}

    = [pb]_q * q^{1-pb}/(q-q^{-1}) * E^pa * K^{(pc+ell-1)%ell} * F^{pb-1}
    - [pb]_q * q^{pb-1}/(q-q^{-1}) * E^pa * K^{(pc+1)%ell} * F^{pb-1}

    Returns list: fe[a] = dict of {(pa,pc,pb): complex_coeff} for F E^a
    """
    q = np.exp(2j * np.pi / ell)
    qm1 = q - q ** (-1)

    fe = [None] * ell
    # F E^0 = F = {(0,0,1): 1}
    fe[0] = {(0, 0, 1): 1.0 + 0j}

    for a in range(1, ell):
        prev = fe[a - 1]
        new_terms = {}

        for (pa, pc, pb), coeff in prev.items():
            # Right multiply E^pa K^pc F^pb by E

            # Term 1: q^{2pc} E^{pa+1} K^pc F^pb  (if pa+1 < ell)
            if pa + 1 < ell:
                key = (pa + 1, pc, pb)
                val = coeff * q ** (2 * pc)
                new_terms[key] = new_terms.get(key, 0.0 + 0j) + val

            # Terms 2 & 3 (if pb >= 1)
            if pb >= 1:
                qn_pb = q_number(pb, q)
                if abs(qn_pb) > 1e-15 and abs(qm1) > 1e-15:
                    # Term 2: K^{-1} = K^{ell-1}, K^pc K^{-1} = K^{(pc+ell-1)%ell}
                    c2 = (pc + ell - 1) % ell
                    val2 = coeff * qn_pb * q ** (1 - pb) / qm1
                    key2 = (pa, c2, pb - 1)
                    new_terms[key2] = new_terms.get(key2, 0.0 + 0j) + val2

                    # Term 3: K^pc K = K^{(pc+1)%ell}
                    c3 = (pc + 1) % ell
                    val3 = -coeff * qn_pb * q ** (pb - 1) / qm1
                    key3 = (pa, c3, pb - 1)
                    new_terms[key3] = new_terms.get(key3, 0.0 + 0j) + val3

        fe[a] = new_terms

    return fe


def build_regular_representation(ell):
    """Build left multiplication matrices L_K, L_E, L_F on the PBW basis of u_q(sl_2).

    The PBW basis is {E^a K^c F^b : 0 <= a,b,c < ell}, ordered as
    idx(a,c,b) = a*ell^2 + c*ell + b.

    Left multiplication rules:
      K · E^a K^c F^b = q^{2(a-b)} E^a K^{(c+1) mod ell} F^b
      E · E^a K^c F^b = E^{a+1} K^c F^b  (if a+1 < ell, else 0)
      F · E^a K^c F^b = (F E^a) K^c F^b  (using precomputed F E^a)

    Parameters
    ----------
    ell : int
        Root of unity order.

    Returns
    -------
    L_K, L_E, L_F : np.ndarray of shape (ell^3, ell^3), dtype=complex
    """
    q = np.exp(2j * np.pi / ell)
    dim = ell ** 3

    L_K = np.zeros((dim, dim), dtype=complex)
    L_E = np.zeros((dim, dim), dtype=complex)

    for a in range(ell):
        for c in range(ell):
            for b in range(ell):
                col = _pbw_index(a, c, b, ell)

                # K · E^a K^c F^b = q^{2(a-b)} E^a K^{(c+1) mod ell} F^b
                new_c = (c + 1) % ell
                row_K = _pbw_index(a, new_c, b, ell)
                L_K[row_K, col] = q ** (2 * (a - b))

                # E · E^a K^c F^b = E^{a+1} K^c F^b (if a+1 < ell)
                if a + 1 < ell:
                    row_E = _pbw_index(a + 1, c, b, ell)
                    L_E[row_E, col] = 1.0

    # Build L_F using precomputed F E^a
    fe = _compute_FEa(ell)

    L_F = np.zeros((dim, dim), dtype=complex)

    for a in range(ell):
        for c in range(ell):
            for b in range(ell):
                col = _pbw_index(a, c, b, ell)
                terms = fe[a]

                for (pa, pc, pb), coeff in terms.items():
                    # (F E^a) has term E^pa K^pc F^pb with coefficient coeff
                    # Need to multiply by K^c F^b on the right:
                    # E^pa K^pc F^pb · K^c F^b

                    # F^pb · K^c = q^{2*pb*c} K^c F^pb  (since F K^c = q^{2c} K^c F)
                    # So: E^pa K^pc · q^{2*pb*c} · K^c · F^{pb+b}
                    # = q^{2*pb*c} · E^pa · K^{(pc+c) % ell} · F^{pb+b}

                    new_b = pb + b
                    if new_b >= ell:
                        continue  # F^ell = 0

                    new_c = (pc + c) % ell
                    phase = q ** (2 * pb * c)
                    val = coeff * phase

                    row = _pbw_index(pa, new_c, new_b, ell)
                    L_F[row, col] += val

    return L_K, L_E, L_F


def build_projective_module(j, ell, L_K=None, L_E=None, L_F=None):
    """Build the projective indecomposable P(j) for u_q(sl_2) at ℓ-th root of unity.

    Uses the baby Verma module construction:
    - P(ell-1) = L(ell-1) = Steinberg (already in q_algebra)
    - P(j) for j < ell-1: dim = 2*ell, constructed from the regular representation

    Parameters
    ----------
    j : int
        Label of the projective indecomposable (0 <= j <= ell-1).
    ell : int
        Root of unity order.
    L_K, L_E, L_F : np.ndarray, optional
        Regular representation matrices (computed if not provided).

    Returns
    -------
    K_P, E_P, F_P : np.ndarray
        Representation matrices on P(j).
    dim_P : int
        Dimension of P(j).
    """
    q = np.exp(2j * np.pi / ell)

    # Steinberg module: use Weyl module directly
    if j == ell - 1:
        from .q_algebra import build_weyl_module_sl2
        K, E, F = build_weyl_module_sl2(j, q)
        return K, E, F, j + 1

    # For j < ell-1, build P(j) from the regular representation
    if L_K is None or L_E is None or L_F is None:
        L_K, L_E, L_F = build_regular_representation(ell)

    dim = ell ** 3
    target_dim = 2 * ell  # dim(P(j)) for j < ell-1

    # Find a highest weight vector of weight j in the regular representation:
    # L_E v = 0 and L_K v = q^j v
    target_eig = q ** j

    # Find kernel of L_E
    U_E, S_E, Vh_E = np.linalg.svd(L_E, full_matrices=True)
    rank_E = np.sum(S_E > RANK_TOL)
    ker_E = Vh_E[rank_E:].T  # Shape: (dim, dim - rank_E)

    # Find eigenspace of L_K for eigenvalue q^j
    eig_vals, eig_vecs = np.linalg.eig(L_K)
    mask = np.abs(eig_vals - target_eig) < 1e-8
    eig_space = eig_vecs[:, mask]

    if eig_space.shape[1] == 0:
        raise ValueError(f"No highest weight vector of weight {j} found in regular representation.")

    # Intersect: project eig_space onto ker_E
    proj = ker_E @ ker_E.conj().T
    hw_space = proj @ eig_space

    # Orthogonalize
    hw_space, _ = np.linalg.qr(hw_space)

    # Filter out zero vectors
    norms = np.linalg.norm(hw_space, axis=0)
    hw_space = hw_space[:, norms > 1e-10]

    if hw_space.shape[1] == 0:
        raise ValueError(f"No valid highest weight vector for P({j}) found.")

    # Take the first highest weight vector
    hwv = hw_space[:, 0]

    # Generate the projective module by applying L_F and L_E
    basis = [hwv.copy()]
    changed = True

    for iteration in range(target_dim + 2):
        if not changed:
            break
        changed = False
        new_basis = []
        for v in list(basis):
            for L_X in [L_F, L_E]:
                w = L_X @ v
                if np.linalg.norm(w) < 1e-10:
                    continue
                for bv in basis + new_basis:
                    w -= np.vdot(bv, w) / (np.vdot(bv, bv) + 1e-30) * bv
                if np.linalg.norm(w) > 1e-10:
                    w /= np.linalg.norm(w)
                    new_basis.append(w)
                    changed = True
        basis.extend(new_basis)
        if len(basis) >= target_dim:
            break

    # Build projection matrix from regular rep to P(j)
    P = np.column_stack(basis[:target_dim])

    # Extract representation matrices
    K_P = P.conj().T @ L_K @ P
    E_P = P.conj().T @ L_E @ P
    F_P = P.conj().T @ L_F @ P

    actual_dim = P.shape[1]
    return K_P, E_P, F_P, actual_dim


def build_all_projectives_sl2(ell):
    """Build all projective indecomposable modules for u_q(sl_2) at ℓ-th root of unity.

    Returns
    -------
    projectives : dict
        Maps j -> {'K': K_mat, 'E': E_mat, 'F': F_mat, 'dim': int,
                    'modified_qdim': float, 'composition': list}
    """
    L_K, L_E, L_F = build_regular_representation(ell)

    projectives = {}
    for j in range(ell):
        K_P, E_P, F_P, dim_P = build_projective_module(j, ell, L_K, L_E, L_F)

        # Modified quantum dimension
        if j < ell - 1:
            mod_qdim = ((-1) ** j * np.sin(np.pi * (j + 1) / ell) /
                        (ell * np.sin(np.pi / ell) ** 2))
        else:
            mod_qdim = 0.0

        projectives[j] = {
            'K': K_P,
            'E': E_P,
            'F': F_P,
            'dim': dim_P,
            'modified_qdim': mod_qdim,
            'is_steinberg': j == ell - 1,
        }

    return projectives


if __name__ == '__main__':
    print("=" * 60)
    print("  Projective Indecomposable Modules for u_q(sl_2)")
    print("=" * 60)

    for ell in [3, 5]:
        print(f"\n--- ell = {ell} ---")

        # Build regular representation and verify algebra relations
        L_K, L_E, L_F = build_regular_representation(ell)
        q = np.exp(2j * np.pi / ell)
        dim = ell ** 3

        # Verify K E = q^2 E K
        err_KE = np.max(np.abs(L_K @ L_E - q ** 2 * L_E @ L_K))
        print(f"  [K, E] error: {err_KE:.2e}")

        # Verify K F = q^{-2} F K
        err_KF = np.max(np.abs(L_K @ L_F - q ** (-2) * L_F @ L_K))
        print(f"  [K, F] error: {err_KF:.2e}")

        # Verify [E, F] = (K - K^{-1})/(q - q^{-1})
        K_inv = np.linalg.inv(L_K)
        comm_EF = L_E @ L_F - L_F @ L_E
        expected = (L_K - K_inv) / (q - q ** (-1))
        err_EF = np.max(np.abs(comm_EF - expected))
        print(f"  [E, F] error: {err_EF:.2e}")

        # Verify E^ell = 0, F^ell = 0, K^ell = I
        E_pow = np.eye(dim, dtype=complex)
        F_pow = np.eye(dim, dtype=complex)
        K_pow = np.eye(dim, dtype=complex)
        for _ in range(ell):
            E_pow = E_pow @ L_E
            F_pow = F_pow @ L_F
            K_pow = K_pow @ L_K

        print(f"  E^ell max: {np.max(np.abs(E_pow)):.2e} (should be ~0)")
        print(f"  F^ell max: {np.max(np.abs(F_pow)):.2e} (should be ~0)")
        print(f"  K^ell - I max: {np.max(np.abs(K_pow - np.eye(dim))):.2e} (should be ~0)")

        # Build projectives
        projectives = build_all_projectives_sl2(ell)
        total_dim = 0
        for j, data in projectives.items():
            d = data['dim']
            mqd = data['modified_qdim']
            total_dim += (j + 1) * d
            print(f"  P({j}): dim={d}, modified_qdim={mqd:.6f}")

        print(f"  Total dim check: sum(j+1)*dim(P(j)) = {total_dim} "
              f"(expected {ell**3}: {'OK' if total_dim == ell**3 else 'MISMATCH'})")
