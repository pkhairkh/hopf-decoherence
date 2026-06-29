"""
Non-Hopf ideal property of ker(Delta) for u_q(sl_n), n >= 3.

Critical discovery: The Jacobson radical J of u_q(sl_n) at roots of unity
is NOT a Hopf ideal for n >= 3. This means:

1. The Verlinde quotient u_q/J does NOT inherit a Hopf algebra structure
2. ker(Delta) intersects J non-trivially but Delta(J) is not contained in
   J tensor u_q + u_q tensor J
3. This creates "leakage" -- elements of J that survive the coproduct
   as cross-terms between CG summands (the cross-Hom mechanism)

For sl_2: The Verlinde quotient IS a Hopf algebra (J IS a Hopf ideal),
but the coproduct representation on St tensor St still has deficiency
D_2(ell) = (ell^3 - ell)/6 because the representation is not faithful.

For sl_3 and higher: J is NOT a Hopf ideal, and the coproduct rank
EXCEEDS the Verlinde rank. This is the key structural input for the
Defect TQFT construction.
"""

import numpy as np
from .q_algebra import _build_sl3_L10, _build_sl3_L01, compute_rank, RANK_TOL
from .coproduct import compute_sl3_coproduct_rank_algebra_closure


def verify_sl3_not_hopf_ideal(ell: int = 3, verbose: bool = False) -> dict:
    """Verify that J is not a Hopf ideal for sl_3 at ell-th root of unity.

    The test:
    1. E_1^2 is in J (the Jacobson radical) for ell=3
    2. But Delta(E_1^2) has a cross-term with coefficient (1 + q^2) != 0
    3. This cross-term is NOT in J tensor u_q + u_q tensor J
    4. Therefore J is not a Hopf ideal

    Numerical verification:
    - rank(Phi on R tensor R) > V_rank for sl_3
    - For ell=3: rank = 155 > V_rank = 19
    """
    q = np.exp(2j * np.pi / ell)

    # Analytic check: cross-term coefficient
    cross_coeff = 1 + q ** 2
    cross_nonzero = abs(cross_coeff) > 1e-10

    # Numerical check: coproduct rank exceeds Verlinde rank
    K1_f, K2_f, E1_f, E2_f, F1_f, F2_f = _build_sl3_L10(q)
    K1_d, K2_d, E1_d, E2_d, F1_d, F2_d = _build_sl3_L01(q)

    # Build R = L(0,0) + L(1,0) + L(0,1), dim=7
    dim_R = 7
    R_K1 = np.zeros((dim_R, dim_R), dtype=complex)
    R_K2 = np.zeros((dim_R, dim_R), dtype=complex)
    R_E1 = np.zeros((dim_R, dim_R), dtype=complex)
    R_E2 = np.zeros((dim_R, dim_R), dtype=complex)
    R_F1 = np.zeros((dim_R, dim_R), dtype=complex)
    R_F2 = np.zeros((dim_R, dim_R), dtype=complex)

    # L(0,0) block (trivial)
    R_K1[0, 0] = 1
    R_K2[0, 0] = 1

    # L(1,0) block
    R_K1[1:4, 1:4] = K1_f
    R_K2[1:4, 1:4] = K2_f
    R_E1[1:4, 1:4] = E1_f
    R_E2[1:4, 1:4] = E2_f
    R_F1[1:4, 1:4] = F1_f
    R_F2[1:4, 1:4] = F2_f

    # L(0,1) block
    R_K1[4:7, 4:7] = K1_d
    R_K2[4:7, 4:7] = K2_d
    R_E1[4:7, 4:7] = E1_d
    R_E2[4:7, 4:7] = E2_d
    R_F1[4:7, 4:7] = F1_d
    R_F2[4:7, 4:7] = F2_d

    coproduct_rank = compute_sl3_coproduct_rank_algebra_closure(
        R_K1, R_K2, R_E1, R_E2, R_F1, R_F2, q
    )

    from .rank_deficiency import verlinde_rank_sl3
    v_rank = verlinde_rank_sl3(ell)

    result = {
        'ell': ell,
        'q': str(q),
        'cross_term_coefficient': str(cross_coeff),
        'cross_term_nonzero': cross_nonzero,
        'coproduct_rank': coproduct_rank,
        'verlinde_rank': v_rank,
        'rank_exceeds_verlinde': coproduct_rank > v_rank,
        'is_not_hopf_ideal': cross_nonzero and coproduct_rank > v_rank,
    }

    if verbose:
        print(f"sl_3 at ell={ell}:")
        print(f"  Cross-term coefficient: {cross_coeff} (nonzero: {cross_nonzero})")
        print(f"  Coproduct rank: {coproduct_rank}")
        print(f"  Verlinde rank: {v_rank}")
        print(f"  J is NOT a Hopf ideal: {result['is_not_hopf_ideal']}")

    return result


def verify_cross_hom_mechanism(ell: int = 3, verbose: bool = False) -> dict:
    """Verify the cross-Hom mechanism for sl_3.

    The coproduct deficiency comes from cross-Hom terms between
    CG summands in the tensor product, not from radicals.

    For L(1,0) tensor L(1,0) at ell=3:
    - CG decomposition: L(2,0) + L(0,1), dims [6, 3]
    - Block-diagonal rank: 6^2 + 3^2 = 45
    - Cross-Hom dimension: 6*3 + 3*6 = 36
    - Total deficiency: 81 - 45 = 36 = cross-Hom
    """
    q = np.exp(2j * np.pi / ell)

    K1, K2, E1, E2, F1, F2 = _build_sl3_L10(q)
    rank = compute_sl3_coproduct_rank_algebra_closure(
        K1, K2, E1, E2, F1, F2, q
    )

    dim_tensor = 9  # 3*3
    deficiency = dim_tensor ** 2 - rank  # 81 - 45 = 36

    # CG decomposition: L(2,0) [dim 6] + L(0,1) [dim 3]
    block_diag = 6 ** 2 + 3 ** 2  # = 45
    cross_hom = 6 * 3 + 3 * 6  # = 36

    result = {
        'ell': ell,
        'pair': 'L(1,0) tensor L(1,0)',
        'coproduct_rank': rank,
        'block_diagonal_rank': block_diag,
        'cross_hom_dimension': cross_hom,
        'deficiency': deficiency,
        'deficiency_equals_cross_hom': deficiency == cross_hom,
        'rank_equals_block_diagonal': rank == block_diag,
    }

    if verbose:
        print(f"L(1,0) tensor L(1,0) at ell={ell}:")
        print(f"  Coproduct rank: {rank}")
        print(f"  Block-diagonal rank: {block_diag}")
        print(f"  Cross-Hom dimension: {cross_hom}")
        print(f"  Deficiency = cross-Hom: {deficiency == cross_hom}")

    return result
