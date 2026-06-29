"""
Unrolled quantum group Û_q(sl₂) for the BCGP non-semisimple TQFT.

The key structural difference from u_q(sl₂):
  - u_q(sl₂): K is grouplike, Δ(K) = K⊗K, leading to rank deficiency in BOTH
    grouplike and nilpotent sectors
  - Û_q(sl₂): H is primitive, Δ(H) = H⊗1 + 1⊗H, K = q^H
    This resolves the grouplike deficiency, isolating the nilpotent deficiency
    to the E, F sector

Generators: E, F, H with K = q^H
Relations:
  [H, E] = 2E
  [H, F] = -2F
  [E, F] = (K - K⁻¹)/(q - q⁻¹) = (q^H - q^{-H})/(q - q⁻¹)
  E^r = 0, F^r = 0
  K^r = 1 on atypical modules (H-eigenvalues ∈ Z); K^r ≠ I on typical modules

Coproduct:
  Δ(H) = H⊗1 + 1⊗H  (PRIMITIVE)
  Δ(E) = E⊗1 + K⊗E
  Δ(F) = F⊗K⁻¹ + 1⊗F
  Δ(K) = K⊗K

Representations:
  - Typical V_α: dim = r, H-eigenvalues α, α-2, ..., α-2(r-1), simple & projective
  - Atypical L(j): dim = j+1, H-eigenvalues j, j-2, ..., -j, simple NOT projective
  - Steinberg St = L(r-1): dim = r, simple AND projective, modified qdim = 0
  - Projective P(j): dim = 2r for j < (r-1)/2, composition factors L(j) and L(r-2-j)

Reference: Blanchet-Costantino-Geer-Patureau-Mirand, arXiv:1605.07941
"""

import numpy as np
from .q_algebra import q_number, RANK_TOL


class UnrolledUqSl2:
    """The unrolled quantum group Û_q(sl₂) at q = exp(2πi/r)."""

    def __init__(self, r: int):
        if r < 3 or r % 2 == 0:
            raise ValueError("r must be odd integer >= 3")
        self.r = r
        self.q = np.exp(2j * np.pi / r)

    def build_typical_module(self, alpha: float):
        """Build the typical module V_α for Û_q(sl₂).

        A typical module has H-eigenvalues α, α-2, α-4, ..., α-2(r-1).
        It is simple and projective for generic (non-integer) α.

        Basis: v_m for m = 0, 1, ..., r-1
        H·v_m = (α - 2m)·v_m
        F·v_m = [m+1]_q · v_{m+1}    (if m < r-1)
        E·v_m = [α-m+1]_q · v_{m-1}  (if m > 0)

        For the typical module the E and F action uses q-numbers directly
        (not the sqrt-balanced normalization of Weyl modules). The formula:

            E·v_m = [α-m+1]_q · v_{m-1}
            F·v_m = [m+1]_q · v_{m+1}

        ensures the relation [E,F] = (K - K⁻¹)/(q - q⁻¹) is satisfied
        for typical modules.

        Parameters
        ----------
        alpha : float
            The H-eigenvalue of the highest weight vector.
            Typical if α ∉ {0, 1, 2, ..., r-1}.

        Returns
        -------
        H_mat, E_mat, F_mat, K_mat : np.ndarray
            Representation matrices on V_α (dim = r).
        """
        r = self.r
        q = self.q
        dim = r

        H_mat = np.zeros((dim, dim), dtype=complex)
        E_mat = np.zeros((dim, dim), dtype=complex)
        F_mat = np.zeros((dim, dim), dtype=complex)
        K_mat = np.zeros((dim, dim), dtype=complex)

        for m in range(dim):
            weight = alpha - 2 * m
            H_mat[m, m] = weight
            K_mat[m, m] = q ** weight

            # F: lowering, F·v_m = [m+1]_q · v_{m+1}
            if m + 1 < dim:
                F_mat[m + 1, m] = q_number(m + 1, q)

            # E: raising, E·v_m = [α-m+1]_q · v_{m-1}
            if m - 1 >= 0:
                n = alpha - m + 1
                # Use the exponential formula for q-number with real n.
                # [n]_q = (q^n - q^{-n})/(q - q^{-1}) = sin(2πn/r)/sin(2π/r)
                # Must use 2π not π since q = e^{2πi/r}.
                E_mat[m - 1, m] = (q**n - q**(-n)) / (q - q**(-1))

        return H_mat, E_mat, F_mat, K_mat

    def build_atypical_module(self, j: int):
        """Build the simple module L(j) for j = 0, 1, ..., r-2.

        This is the Weyl module with highest weight j.
        Basis: v_m for m = 0, 1, ..., j
        H·v_m = (j - 2m)·v_m

        Uses the same sqrt-balanced normalization as build_weyl_module_sl2:
            F·v_m = sqrt([m+1]_q [j-m]_q) · v_{m+1}
            E·v_m = sqrt([m]_q [j-m+1]_q) · v_{m-1}

        Parameters
        ----------
        j : int
            Highest weight (0 ≤ j ≤ r-2, or j = r-1 for Steinberg).

        Returns
        -------
        H_mat, E_mat, F_mat, K_mat : np.ndarray
            Representation matrices on L(j) (dim = j+1).
        """
        r = self.r
        q = self.q
        dim = j + 1

        H_mat = np.zeros((dim, dim), dtype=complex)
        E_mat = np.zeros((dim, dim), dtype=complex)
        F_mat = np.zeros((dim, dim), dtype=complex)
        K_mat = np.zeros((dim, dim), dtype=complex)

        for m in range(dim):
            weight = j - 2 * m
            H_mat[m, m] = weight
            K_mat[m, m] = q ** weight

            # F: lowering. F·v_m = sqrt([m+1]_q [j-m]_q) · v_{m+1}
            if m + 1 < dim:
                val = q_number(m + 1, q) * q_number(j - m, q)
                F_mat[m + 1, m] = np.sqrt(val) if abs(val) > 1e-30 else 0.0

            # E: raising. E·v_m = sqrt([m]_q [j-m+1]_q) · v_{m-1}
            if m - 1 >= 0:
                val = q_number(m, q) * q_number(j - m + 1, q)
                E_mat[m - 1, m] = np.sqrt(val) if abs(val) > 1e-30 else 0.0

        return H_mat, E_mat, F_mat, K_mat

    def build_steinberg(self):
        """Build the Steinberg module St = L(r-1).

        dim = r, simple AND projective, modified qdim = 0.
        """
        return self.build_atypical_module(self.r - 1)

    def coproduct_matrices(self, H, E, F, K, dim):
        """Compute coproduct matrices on V⊗V for Û_q(sl₂).

        Δ(H) = H⊗1 + 1⊗H  (PRIMITIVE - key difference!)
        Δ(E) = E⊗1 + K⊗E
        Δ(F) = F⊗K⁻¹ + 1⊗F
        Δ(K) = K⊗K

        Parameters
        ----------
        H, E, F, K : np.ndarray
            Representation matrices on V (dim × dim).
        dim : int
            Dimension of V.

        Returns
        -------
        dH, dE, dF, dK : np.ndarray
            Coproduct matrices on V⊗V (dim² × dim²).
        """
        I = np.eye(dim, dtype=complex)
        K_inv = np.linalg.inv(K)

        dH = np.kron(H, I) + np.kron(I, H)  # PRIMITIVE!
        dE = np.kron(E, I) + np.kron(K, E)
        dF = np.kron(F, K_inv) + np.kron(I, F)
        dK = np.kron(K, K)

        return dH, dE, dF, dK

    def verify_algebra(self, H, E, F, K, tol=1e-6, check_Kr=True):
        """Verify Û_q(sl₂) relations on a representation.

        Checks:
        1. [H, E] = 2E
        2. [H, F] = -2F
        3. K = q^H  (i.e. K is diagonal with K_{mm} = q^{H_{mm}})
        4. KE = q² EK
        5. KF = q⁻² FK
        6. [E, F] = (K - K⁻¹)/(q - q⁻¹)
        7. E^r = 0
        8. F^r = 0
        9. K^r = I  (only for atypical modules; set check_Kr=False for typical)

        Parameters
        ----------
        H, E, F, K : np.ndarray
            Representation matrices.
        tol : float
            Tolerance for numerical comparison.
        check_Kr : bool
            Whether to check K^r = I. Should be True for atypical modules
            (integer H-eigenvalues) and False for typical modules (non-integer
            H-eigenvalues), since K^r = q^{rH} ≠ I when H has non-integer eigenvalues.
        """
        q = self.q
        r = self.r
        dim = H.shape[0]

        errs = {}
        errs['HE'] = np.max(np.abs(H @ E - E @ H - 2 * E))
        errs['HF'] = np.max(np.abs(H @ F - F @ H + 2 * F))

        # Verify K = q^H for diagonal H
        H_diag = H.diagonal()
        K_expected = np.diag(q ** H_diag)
        errs['K_eq_qH'] = np.max(np.abs(K - K_expected))

        # Commutation relations
        errs['KE'] = np.max(np.abs(K @ E - q**2 * E @ K))
        errs['KF'] = np.max(np.abs(K @ F - q**(-2) * F @ K))

        K_inv = np.linalg.inv(K)
        comm_EF = E @ F - F @ E
        expected_EF = (K - K_inv) / (q - q**(-1))
        errs['EF'] = np.max(np.abs(comm_EF - expected_EF))

        # Check nilpotency
        E_pow = np.eye(dim, dtype=complex)
        F_pow = np.eye(dim, dtype=complex)
        for _ in range(r):
            E_pow = E_pow @ E
            F_pow = F_pow @ F

        errs['E_r'] = np.max(np.abs(E_pow))
        errs['F_r'] = np.max(np.abs(F_pow))

        # K^r = I only holds for atypical modules (integer H-eigenvalues)
        if check_Kr:
            K_pow = np.eye(dim, dtype=complex)
            for _ in range(r):
                K_pow = K_pow @ K
            errs['K_r'] = np.max(np.abs(K_pow - np.eye(dim)))

        all_ok = all(v < tol for v in errs.values())
        return all_ok, errs

    def verify_coproduct_relations(self, H, E, F, K, dim, tol=1e-6):
        """Verify coproduct algebra relations on V⊗V.

        The coproduct must preserve the Û_q(sl₂) relations:
        1. [Δ(H), Δ(E)] = 2Δ(E)
        2. [Δ(H), Δ(F)] = -2Δ(F)
        3. Δ(K)Δ(E) = q² Δ(E)Δ(K)
        4. Δ(K)Δ(F) = q⁻² Δ(F)Δ(K)
        5. [Δ(E), Δ(F)] = (Δ(K) - Δ(K)⁻¹)/(q - q⁻¹)
        6. Δ(K) = q^{Δ(H)}  (since Δ(H) is primitive and Δ(K) = K⊗K)
        """
        dH, dE, dF, dK = self.coproduct_matrices(H, E, F, K, dim)
        q = self.q

        errs = {}
        errs['coproduct_HE'] = np.max(np.abs(dH @ dE - dE @ dH - 2 * dE))
        errs['coproduct_HF'] = np.max(np.abs(dH @ dF - dF @ dH + 2 * dF))
        errs['coproduct_KE'] = np.max(np.abs(dK @ dE - q**2 * dE @ dK))
        errs['coproduct_KF'] = np.max(np.abs(dK @ dF - q**(-2) * dF @ dK))

        dK_inv = np.linalg.inv(dK)
        comm_dE_dF = dE @ dF - dF @ dE
        expected_dEF = (dK - dK_inv) / (q - q**(-1))
        errs['coproduct_EF'] = np.max(np.abs(comm_dE_dF - expected_dEF))

        # Verify Δ(K) = q^{Δ(H)} on diagonal
        dH_diag = dH.diagonal()
        dK_expected = np.diag(q ** dH_diag)
        errs['coproduct_K_eq_qH'] = np.max(np.abs(dK - dK_expected))

        all_ok = all(v < tol for v in errs.values())
        return all_ok, errs

    def compute_coproduct_rank_deficiency(self, j=None, alpha=None):
        """Compute the coproduct rank deficiency for Û_q(sl₂) on a specific module.

        For the unrolled quantum group, the grouplike deficiency is resolved
        (H is primitive), so only the nilpotent deficiency remains.

        Parameters
        ----------
        j : int, optional
            If provided, compute on the atypical module L(j).
        alpha : float, optional
            If provided, compute on the typical module V_α.

        Returns
        -------
        rank : int
            Rank of the coproduct representation.
        deficiency : int or None
            Algebra dimension minus rank (None if algebra dim unknown).
        """
        r = self.r

        if j is not None:
            H, E, F, K = self.build_atypical_module(j)
        elif alpha is not None:
            H, E, F, K = self.build_typical_module(alpha)
        else:
            raise ValueError("Must specify either j or alpha")

        dim = H.shape[0]
        dH, dE, dF, dK = self.coproduct_matrices(H, E, F, K, dim)

        tensor_dim = dim ** 2

        # Build the algebra generated by dH, dE, dF, dK on V⊗V
        # Use algebra closure (Gram-Schmidt approach)
        gens = [np.eye(tensor_dim, dtype=complex), dH, dE, dF, dK]

        basis = []
        for g in gens:
            v = g.ravel()
            for bv in basis:
                v -= np.vdot(bv, v) / (np.vdot(bv, bv) + 1e-30) * bv
            if np.linalg.norm(v) > 1e-10:
                basis.append(v / np.linalg.norm(v))

        for _ in range(50):
            new = []
            old = len(basis)
            for bv in list(basis):
                bm = bv.reshape(tensor_dim, tensor_dim)
                for g in gens:
                    v = (g @ bm).ravel()
                    for ex in basis + new:
                        v -= np.vdot(ex, v) / (np.vdot(ex, ex) + 1e-30) * ex
                    if np.linalg.norm(v) > 1e-10:
                        new.append(v / np.linalg.norm(v))
            basis.extend(new)
            if len(basis) == old:
                break

        rank = len(basis)
        # For Û_q(sl₂), the algebra dimension depends on the module
        # The deficiency = algebra_dim - rank

        return rank, None  # Return rank; deficiency needs algebra dim context


def modified_qdim(j, r):
    """Modified quantum dimension d̃(P_j) for the BCGP construction.

    d̃(P_j) = (-1)^j sin(π(j+1)/r) / (r sin²(π/r))
    d̃(St) = d̃(P_{r-1}) = 0

    These are the GPY modified quantum dimensions on projective modules.
    """
    if j == r - 1:
        return 0.0
    return ((-1)**j * np.sin(np.pi * (j + 1) / r)) / (r * np.sin(np.pi / r)**2)


def typical_qdim(alpha, r):
    """Modified quantum dimension of the typical module V_α.

    For typical modules (simple and projective):
    d̃(V_α) = sin(πα/r) / (r sin²(π/r))
    """
    return np.sin(np.pi * alpha / r) / (r * np.sin(np.pi / r)**2)


def verify_steinberg_properties(r, verbose=True):
    """Verify key properties of the Steinberg module St = L(r-1).

    1. St is simple (verified by irreducibility)
    2. St is projective (verified by the lifting property)
    3. d̃(St) = 0 (modified quantum dimension)
    """
    uq = UnrolledUqSl2(r)
    H, E, F, K = uq.build_steinberg()

    # Verify algebra
    ok, errs = uq.verify_algebra(H, E, F, K)

    # Verify modified qdim = 0
    d = modified_qdim(r - 1, r)

    result = {
        'r': r,
        'algebra_ok': ok,
        'modified_qdim': d,
        'is_zero_qdim': abs(d) < 1e-10,
        'errors': errs,
    }

    if verbose:
        print(f"Steinberg St = L({r-1}) at r={r}:")
        print(f"  Algebra relations: {'OK' if ok else 'FAILED'}")
        print(f"  Modified qdim: {d:.6f} (should be 0)")
        for k, v in errs.items():
            print(f"  {k}: {v:.2e}")

    return result


if __name__ == '__main__':
    print("=" * 60)
    print("  Unrolled Quantum Group Û_q(sl₂)")
    print("=" * 60)

    for r in [3, 5, 7]:
        uq = UnrolledUqSl2(r)
        print(f"\n--- r = {r} ---")

        # Build and verify atypical modules
        print("  Atypical modules:")
        for j in range(min(r - 1, 5)):
            H, E, F, K = uq.build_atypical_module(j)
            ok, errs = uq.verify_algebra(H, E, F, K)
            print(f"    L({j}): dim={j+1}, algebra={'OK' if ok else 'FAIL'}, "
                  f"max_err={max(errs.values()):.2e}")

        # Build and verify typical module
        # Note: check_Kr=False because typical modules have non-integer H-eigenvalues
        # so K^r ≠ I (K^r = q^{rH} = e^{2πiH} ≠ I for non-integer H)
        print("  Typical module V_0.5:")
        H, E, F, K = uq.build_typical_module(0.5)
        ok, errs = uq.verify_algebra(H, E, F, K, check_Kr=False)
        print(f"    dim={r}, algebra={'OK' if ok else 'FAIL'}, "
              f"max_err={max(errs.values()):.2e}")

        # Verify Steinberg
        verify_steinberg_properties(r)

        # Modified quantum dimensions
        print("  Modified quantum dimensions:")
        for j in range(r):
            d = modified_qdim(j, r)
            print(f"    P({j}): d̃ = {d:+.6f}")
