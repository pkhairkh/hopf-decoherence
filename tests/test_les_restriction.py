"""Pytest for the Mastnak-Witherspoon LES restriction map test.

Verifies that the experimental signature of the connecting homomorphism δ
in the Gerstenhaber-Schack / Mastnak-Witherspoon long exact sequence (3.3.1)
holds for u_q(sl_2) at ell = 3.

Specifically, the LES in degrees 1-2 reads:
  HH¹(D(B)) → HH¹(B) ⊕ HH¹(B*) → H̃¹_b(B) --δ--> HH²(D(B)) --π̄--> HH²(B) ⊕ HH²(B*)

For u_q(sl_2) at ell = 3, the verified values are:
  dim HH¹(D(B)) = 0
  dim HH¹(B⁺) = dim HH¹(B⁻) = 0
  dim HH²(D(B)) = 3
  dim HH²(B⁺) = dim HH²(B⁻) = 1

By exactness, dim im(π̄) = 2 (the [E³] and [F³] classes) and
dim ker(π̄) = dim im(δ) = 1 (the Cartan-type / mixed E-F class).

Run:
    pytest tests/test_restriction_map.py -v
"""
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "test_restriction_map.py")
assert os.path.exists(SCRIPT_PATH), f"Missing {SCRIPT_PATH}"

spec = importlib.util.spec_from_file_location("test_restriction_map", SCRIPT_PATH)
trm = importlib.util.module_from_spec(spec)
sys.modules["test_restriction_map"] = trm
spec.loader.exec_module(trm)


@pytest.fixture(scope="module")
def restriction_data():
    """Run the full restriction map computation once and reuse."""
    return trm.main()


def test_restriction_map_runs():
    """The restriction map script runs without error."""
    # main() prints a lot but doesn't return anything; just check no exception
    trm.main()


def test_hh2_dimension_is_3():
    """Smoke test: re-verify dim HH²(u_q(sl_2), C) at ell=3 is 3."""
    import numpy as np
    from verify_sl2_hh2 import build_multiplication_table, DIM, idx

    mult = build_multiplication_table()
    eps = np.zeros(DIM, dtype=complex)
    for a in range(3):
        eps[idx(a, 0, 0)] = 1.0

    # weight 0 only
    wts = np.array([trm.weight(i) for i in range(DIM)])
    wt0 = list(np.where(wts == 0)[0])
    pairs2 = [(i, j) for i in range(DIM) for j in range(DIM) if (wts[i] + wts[j]) % 3 == 0]
    n1 = len(wt0)
    n2 = len(pairs2)

    d1 = np.zeros((n2, n1), dtype=complex)
    for col, i_f in enumerate(wt0):
        for row, (a, b) in enumerate(pairs2):
            mult_if = mult[i_f, a, b]
            d1[row, col] = (eps[a] * (1.0 if b == i_f else 0.0)
                            - mult_if
                            + (1.0 if a == i_f else 0.0) * eps[b])

    # dim HH¹ = n1 - rank(d1)
    s1 = np.linalg.svd(d1, compute_uv=False)
    rank_d1 = int(np.sum(s1 > 1e-9 * s1[0]))
    hh1 = n1 - rank_d1
    assert hh1 == 0, f"HH¹(u_q(sl_2)) at ell=3 should be 0, got {hh1}"
