"""Pytest for the direct bar-complex computation of HH^2(u_q(sl_2), C) at ell=3.

This test verifies the A_1 case of the paper's main formula:
    dim HH^2(u_q(sl_2), C) = C(2,2) + 2|Phi^+(A_1)| = 1 + 2 = 3.

The computation builds the full Hochschild bar complex of u_q(sl_2) at q = e^{2*pi*i/3}
(dimension 27) and computes dim ker(d^2) - dim im(d^1) directly via SVD / Gram-matrix
eigenvalues.  The script scripts/verify_sl2_hh2.py contains the full implementation
and is imported here so the test reuses the same code path.

Run:
    pytest tests/test_sl2_hh2_bar_complex.py -v
"""
import importlib.util
import os
import sys

import pytest

# Import the verification script (which lives in scripts/, not in the package).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "verify_sl2_hh2.py")
assert os.path.exists(SCRIPT_PATH), f"Missing {SCRIPT_PATH}"

spec = importlib.util.spec_from_file_location("verify_sl2_hh2", SCRIPT_PATH)
verify_sl2_hh2 = importlib.util.module_from_spec(spec)
sys.modules["verify_sl2_hh2"] = verify_sl2_hh2
spec.loader.exec_module(verify_sl2_hh2)


@pytest.fixture(scope="module")
def hh2_data():
    """Build the bar complex once and reuse across tests in this module."""
    from verify_sl2_hh2 import (
        build_d1, build_d2_sparse, build_multiplication_table, ELL, DIM,
    )
    import numpy as np
    mult = build_multiplication_table()
    epsilon = np.zeros(DIM, dtype=complex)
    # Counit: eps(K^a E^b F^c) = 1 if b = c = 0, else 0.
    for a in range(ELL):
        epsilon[verify_sl2_hh2.idx(a, 0, 0)] = 1.0
    d1 = build_d1(mult, epsilon)
    d2 = build_d2_sparse(mult, epsilon)
    return {"mult": mult, "epsilon": epsilon, "d1": d1, "d2": d2}


def test_algebra_relations(hh2_data):
    """Verify the defining relations of u_q(sl_2) at ell=3."""
    import numpy as np
    from verify_sl2_hh2 import ELL, Q, Q_INV, D, idx, from_idx, DIM
    mult = hh2_data["mult"]
    e0 = idx(0, 0, 0)
    K  = idx(1, 0, 0)
    K2 = idx(2, 0, 0)
    E  = idx(0, 1, 0)
    F  = idx(0, 0, 1)
    # K^3 = 1
    assert abs(mult[e0, K, K2] - 1.0) < 1e-12
    # E^3 = 0
    assert np.max(np.abs(mult[:, idx(0, 2, 0), E])) < 1e-12
    # F^3 = 0
    assert np.max(np.abs(mult[:, idx(0, 0, 2), F])) < 1e-12
    # K E = q^2 E K
    assert abs(mult[idx(1, 1, 0), K, E] - 1.0) < 1e-12
    assert abs(mult[idx(1, 1, 0), E, K] - Q_INV**2) < 1e-12
    # [E, F] = (K - K^-1) / (q - q^-1)
    EF_minus_FE = mult[:, E, F] - mult[:, F, E]
    expected = np.zeros(DIM, dtype=complex)
    expected[K]  = 1.0 / D
    expected[K2] = -1.0 / D
    assert np.max(np.abs(EF_minus_FE - expected)) < 1e-12


def test_associativity(hh2_data):
    """Verify (x*y)*z = x*(y*z) on a sample of basis triples."""
    import numpy as np
    import random
    from verify_sl2_hh2 import DIM
    mult = hh2_data["mult"]
    random.seed(42)
    failures = 0
    for _ in range(500):
        i = random.randrange(DIM); j = random.randrange(DIM); k = random.randrange(DIM)
        v1 = mult[:, i, j]
        left = np.zeros(DIM, dtype=complex)
        for l in range(DIM):
            if abs(v1[l]) > 1e-13:
                left += v1[l] * mult[:, l, k]
        v2 = mult[:, j, k]
        right = np.zeros(DIM, dtype=complex)
        for l in range(DIM):
            if abs(v2[l]) > 1e-13:
                right += v2[l] * mult[:, i, l]
        if np.max(np.abs(left - right)) > 1e-9:
            failures += 1
    assert failures == 0


def test_d2_compose_d1_is_zero(hh2_data):
    """The Hochschild differential relation d^2 o d^1 = 0."""
    import numpy as np
    prod = hh2_data["d2"] @ hh2_data["d1"]
    if hasattr(prod, "toarray"):
        prod = prod.toarray()
    # Use a relative tolerance scaled by the largest entry of d1 and d2.
    scale = max(np.max(np.abs(hh2_data["d1"])),
                np.max(np.abs(hh2_data["d2"].toarray() if hasattr(hh2_data["d2"], "toarray") else hh2_data["d2"])))
    assert np.max(np.abs(prod)) < 1e-9 * scale


def test_counit_is_algebra_map(hh2_data):
    """eps(x*y) = eps(x)*eps(y) for all basis pairs."""
    import numpy as np
    from verify_sl2_hh2 import DIM, idx
    mult = hh2_data["mult"]
    eps = hh2_data["epsilon"]
    e0 = idx(0, 0, 0)
    for i in range(DIM):
        for j in range(DIM):
            exy = sum(eps[k] * mult[k, i, j] for k in range(DIM))
            exy = complex(np.sum(eps[:, None, None] * mult))
            # Actually do it directly:
            exy_direct = complex(np.vdot(eps, mult[:, i, j].conj())) if False else np.sum(eps * mult[:, i, j])
            ex = eps[i]; ey = eps[j]
            assert abs(exy_direct - ex * ey) < 1e-9, \
                f"eps map fails at ({i},{j}): {exy_direct} vs {ex*ey}"


def test_hh2_dimension(hh2_data):
    """dim HH^2(u_q(sl_2), C) at ell = 3 equals 3 (paper's A_1 theorem)."""
    import numpy as np
    from verify_sl2_hh2 import DIM, matrix_rank_complex, sparse_rank_via_gram
    d1 = hh2_data["d1"]
    d2 = hh2_data["d2"]
    rank1, _ = matrix_rank_complex(d1)
    rank2, eigvals = sparse_rank_via_gram(d2)
    dim_ker_d2 = DIM * DIM - rank2
    hh2 = dim_ker_d2 - rank1
    # Sanity: spectral gap should be clean (smallest nonzero eig >> largest "zero" eig).
    sorted_eigs = np.sort(np.abs(eigvals))[::-1]
    assert sorted_eigs[rank2 - 1] > 1.0, \
        f"Smallest nonzero eigenvalue too small: {sorted_eigs[rank2 - 1]}"
    assert sorted_eigs[rank2] < 1e-9, \
        f"Largest zero eigenvalue too big: {sorted_eigs[rank2]}"
    assert hh2 == 3, f"dim HH^2 = {hh2}, expected 3"
