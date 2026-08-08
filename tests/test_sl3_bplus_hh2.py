"""Pytest for the B+(u_q(sl_3)) bar-complex computation.

This test verifies the algebra relations and associativity of the multiplication
table for B+(u_q(sl_3)) at ell = 3 (dimension 243). The full HH^2 computation
takes ~4 minutes and is therefore not run by default; it is verified separately
by `scripts/verify_sl3_bplus_hh2.py`.
"""
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "verify_sl3_bplus_hh2.py")
assert os.path.exists(SCRIPT_PATH), f"Missing {SCRIPT_PATH}"

spec = importlib.util.spec_from_file_location("verify_sl3_bplus_hh2", SCRIPT_PATH)
verify_sl3 = importlib.util.module_from_spec(spec)
sys.modules["verify_sl3_bplus_hh2"] = verify_sl3
spec.loader.exec_module(verify_sl3)


def test_algebra_relations():
    """Verify the defining relations of B+(u_q(sl_3)) at ell = 3."""
    import numpy as np
    from verify_sl3_bplus_hh2 import build_mult_sparse, idx, from_idx, ELL, Q, Q_INV, DIM

    ms = build_mult_sparse(DIM)
    e0 = idx(0, 0, 0, 0, 0)
    K1 = idx(1, 0, 0, 0, 0)
    K2 = idx(0, 1, 0, 0, 0)
    E1 = idx(0, 0, 1, 0, 0)
    E2 = idx(0, 0, 0, 0, 1)
    E12 = idx(0, 0, 0, 1, 0)

    # K1^3 = 1: K1 * K1^2 = 1
    K1_sq = idx(2, 0, 0, 0, 0)
    prod = ms[K1][K1_sq]
    assert any(l == e0 and abs(v - 1.0) < 1e-12 for (l, v) in prod), "K1^3 != 1"

    # K2^3 = 1
    K2_sq = idx(0, 2, 0, 0, 0)
    prod = ms[K2][K2_sq]
    assert any(l == e0 and abs(v - 1.0) < 1e-12 for (l, v) in prod), "K2^3 != 1"

    # E1^3 = 0
    E1_sq = idx(0, 0, 2, 0, 0)
    assert len(ms[E1_sq][E1]) == 0, "E1^3 != 0"

    # E2^3 = 0
    E2_sq = idx(0, 0, 0, 0, 2)
    assert len(ms[E2_sq][E2]) == 0, "E2^3 != 0"

    # E12^3 = 0
    E12_sq = idx(0, 0, 0, 2, 0)
    assert len(ms[E12_sq][E12]) == 0, "E12^3 != 0"

    # K1 E1 = q^2 E1 K1  =>  ms[K1][E1] should give (K1 E1, 1.0) and ms[E1][K1] should give (K1 E1, q^{-2})
    k1e1_idx = idx(1, 0, 1, 0, 0)
    prod = ms[K1][E1]
    assert any(l == k1e1_idx and abs(v - 1.0) < 1e-12 for (l, v) in prod), "K1*E1 != K1 E1"
    prod = ms[E1][K1]
    # E1 K1 = q^{-2} K1 E1
    assert any(l == k1e1_idx and abs(v - Q_INV**2) < 1e-12 for (l, v) in prod), "E1*K1 != q^{-2} K1 E1"

    # E1 E2 - q E2 E1 = E12
    diff = {}
    for (l, v) in ms[E1][E2]:
        diff[l] = diff.get(l, 0) + v
    for (l, v) in ms[E2][E1]:
        diff[l] = diff.get(l, 0) - v * Q
    diff = {k: v for k, v in diff.items() if abs(v) > 1e-10}
    assert len(diff) == 1 and E12 in diff and abs(diff[E12] - 1.0) < 1e-10, \
        f"E1 E2 - q E2 E1 != E12, got {diff}"

    # Quantum Serre: E1^2 E2 + E1 E2 E1 + E2 E1^2 = 0 (for q + q^{-1} = -1)
    e1_e2 = ms[E1][E2]
    e1sq_e2 = {}
    for (l, v) in e1_e2:
        for (l2, v2) in ms[E1][l]:
            e1sq_e2[l2] = e1sq_e2.get(l2, 0) + v * v2
    e2_e1 = ms[E2][E1]
    e1_e2_e1 = {}
    for (l, v) in e2_e1:
        for (l2, v2) in ms[E1][l]:
            e1_e2_e1[l2] = e1_e2_e1.get(l2, 0) + v * v2
    e1sq_idx = idx(0, 0, 2, 0, 0)
    e2_e1sq = ms[E2][e1sq_idx]
    total = {}
    for (l, v) in e1sq_e2.items():
        total[l] = total.get(l, 0) + v
    for (l, v) in e1_e2_e1.items():
        total[l] = total.get(l, 0) + v
    for (l, v) in e2_e1sq:
        total[l] = total.get(l, 0) + v
    total = {k: v for k, v in total.items() if abs(v) > 1e-10}
    assert len(total) == 0, f"Serre I failed, residual = {total}"


def test_associativity():
    """Verify (x*y)*z = x*(y*z) on 500 random triples."""
    import random
    import numpy as np
    from verify_sl3_bplus_hh2 import build_mult_sparse, DIM

    ms = build_mult_sparse(DIM)
    random.seed(42)
    failures = 0
    for _ in range(500):
        i = random.randrange(DIM); j = random.randrange(DIM); k = random.randrange(DIM)
        left = {}
        for (l, v) in ms[i][j]:
            for (m, v2) in ms[l][k]:
                left[m] = left.get(m, 0) + v * v2
        right = {}
        for (l, v) in ms[j][k]:
            for (m, v2) in ms[i][l]:
                right[m] = right.get(m, 0) + v * v2
        all_keys = set(left.keys()) | set(right.keys())
        if all_keys:
            diff = max(abs(left.get(k2, 0) - right.get(k2, 0)) for k2 in all_keys)
            if diff > 1e-9:
                failures += 1
    assert failures == 0, f"{failures} associativity failures"


@pytest.mark.slow
def test_hh2_dimension():
    """dim HH^2(B+(u_q(sl_3)), C) at ell = 3 equals 5 (not 6 as the old conjecture predicted).

    This test runs the full bar-complex computation (~4 minutes) and is marked 'slow'
    so it can be skipped in fast test runs: `pytest -m 'not slow'`.
    """
    from verify_sl3_bplus_hh2 import compute_hh2
    hh2 = compute_hh2()
    assert hh2 == 5, f"dim HH^2(B+(sl_3), C) = {hh2}, expected 5"
