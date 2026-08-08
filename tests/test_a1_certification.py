"""Tests for the exact cyclotomic certification of HH^2(u_q(sl_2), C) at ell = 3.

These tests verify the three required properties of the certification script
scripts/certify_a1_exact.py:

  1. test_script_runs -- the script runs end-to-end and exits 0,
     producing the certification message.
  2. test_rank_consistent_across_primes -- rank(d^1) and rank(d^2) are the
     same modulo every tested prime p ≡ 1 (mod 3).
  3. test_dim_hh2_is_3 -- the certified dim HH^2 equals 3, matching the
     paper's Theorem 1.2 (C(n+1, 2) + 2|Phi^+| = 1 + 2 = 3 for n = 1).

Run:
    pytest tests/test_a1_certification.py -v
"""
import importlib.util
import os
import sys

import pytest

# Import the certification script (which lives in scripts/, not in the package).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "certify_a1_exact.py")
assert os.path.exists(SCRIPT_PATH), f"Missing {SCRIPT_PATH}"

spec = importlib.util.spec_from_file_location("certify_a1_exact", SCRIPT_PATH)
cert = importlib.util.module_from_spec(spec)
sys.modules["certify_a1_exact"] = cert
spec.loader.exec_module(cert)


# A reduced set of primes for the faster unit tests (the full script uses 11).
TEST_PRIMES = [7, 13, 19, 31, 37]


@pytest.fixture(scope="module")
def bar_complex():
    """Build the bar complex over Z[omega, 1/3] once and reuse across tests."""
    mult = cert.build_multiplication_table()
    epsilon = cert.build_epsilon()
    # Sanity-check the algebra relations.
    cert.sanity_checks(mult)
    d1 = cert.build_d1(mult, epsilon)
    d2 = cert.build_d2(mult, epsilon)
    return {"mult": mult, "epsilon": epsilon, "d1": d1, "d2": d2}


def test_er_arithmetic():
    """Verify basic properties of the Eisenstein-rational arithmetic."""
    # D = 1 + 2*omega, D^2 = -3, 1/D = (-1 - 2*omega)/3.
    assert cert.D == cert.ER(1, 2, 0)
    assert cert.D * cert.D == cert.ER(-3, 0, 0)
    assert cert.D * cert.INV_D == cert.ONE
    # omega^2 = -1 - omega
    assert cert.OMEGA * cert.OMEGA == cert.OMEGA2
    # omega^3 = 1
    assert cert.OMEGA * cert.OMEGA * cert.OMEGA == cert.ONE
    # q^(-1) = q^2, q^(-2) = q
    assert cert.qpow(-1) == cert.OMEGA2
    assert cert.qpow(-2) == cert.OMEGA
    # Reduction mod 7 (q_p = 4 is a primitive cube root of unity mod 7).
    p, q_p = 7, 4
    assert cert.ONE.reduce_mod(p, q_p) == 1
    assert cert.OMEGA.reduce_mod(p, q_p) == 4
    assert cert.OMEGA2.reduce_mod(p, q_p) == (4 * 4) % p  # = 2
    assert cert.INV_D.reduce_mod(p, q_p) == pow(cert.D.reduce_mod(p, q_p), -1, p)


def test_find_cube_root_mod_p():
    """Verify that find_cube_root_mod_p returns a primitive cube root of unity."""
    for p in TEST_PRIMES:
        q_p = cert.find_cube_root_mod_p(p)
        assert q_p is not None, f"No cube root of unity found mod {p}"
        assert q_p != 1, f"q_p = 1 is not primitive mod {p}"
        assert (q_p * q_p * q_p) % p == 1, f"q_p^3 != 1 mod {p}"
        assert (q_p * q_p) % p != 1, f"q_p has order < 3 mod {p}"


def test_algebra_relations(bar_complex):
    """Verify the defining relations of u_q(sl_2) at ell = 3 with exact arithmetic.

    (Already checked inside sanity_checks, but re-asserted here for the test suite.)
    """
    mult = bar_complex["mult"]
    e0 = cert.idx(0, 0, 0)
    K = cert.idx(1, 0, 0)
    K2 = cert.idx(2, 0, 0)
    E = cert.idx(0, 1, 0)
    F = cert.idx(0, 0, 1)
    E2 = cert.idx(0, 2, 0)
    F2 = cert.idx(0, 0, 2)
    KE_idx = cert.idx(1, 1, 0)

    # 1 * x = x * 1 = x.
    for i in range(cert.DIM):
        assert mult.get((i, e0, i), cert.ZERO) == cert.ONE
        assert mult.get((i, i, e0), cert.ZERO) == cert.ONE
    # K^3 = 1.
    assert mult.get((K2, K, K), cert.ZERO) == cert.ONE
    assert mult.get((e0, K, K2), cert.ZERO) == cert.ONE
    # E^3 = F^3 = 0.
    for i in range(cert.DIM):
        assert mult.get((i, E2, E), cert.ZERO).is_zero()
        assert mult.get((i, F2, F), cert.ZERO).is_zero()
    # K E = q^2 E K  (equivalently E K = q^{-2} K E).
    assert mult.get((KE_idx, K, E), cert.ZERO) == cert.ONE
    assert mult.get((KE_idx, E, K), cert.ZERO) == cert.qpow(-2)
    # [E, F] = (K - K^{-1}) / (q - q^{-1}).
    for k in range(cert.DIM):
        diff = mult.get((k, E, F), cert.ZERO) - mult.get((k, F, E), cert.ZERO)
        if k == K:
            assert diff == cert.INV_D
        elif k == K2:
            assert diff == -cert.INV_D
        else:
            assert diff.is_zero()


def test_rank_consistent_across_primes(bar_complex):
    """Check that all primes give the same rank(d^1) and rank(d^2)."""
    d1 = bar_complex["d1"]
    d2 = bar_complex["d2"]

    ranks_d1 = []
    ranks_d2 = []
    for p in TEST_PRIMES:
        q_p = cert.find_cube_root_mod_p(p)
        assert q_p is not None
        A1 = cert.reduce_to_numpy(d1, cert.DIM * cert.DIM, cert.DIM, p, q_p)
        rank1 = cert.rank_mod_p(A1, p)
        ranks_d1.append(rank1)
        A2 = cert.reduce_to_numpy(d2, cert.DIM ** 3, cert.DIM * cert.DIM, p, q_p)
        rank2 = cert.rank_mod_p(A2, p)
        ranks_d2.append(rank2)

    # All primes must give the same rank for semicontinuity to apply.
    assert len(set(ranks_d1)) == 1, \
        f"rank(d^1) inconsistent across primes: {ranks_d1}"
    assert len(set(ranks_d2)) == 1, \
        f"rank(d^2) inconsistent across primes: {ranks_d2}"


def test_dim_hh2_is_3(bar_complex):
    """The certified dim HH^2(u_q(sl_2), C) at ell = 3 equals 3.

    dim HH^2 = dim ker(d^2) - dim im(d^1) = (dim C^2 - rank(d^2)) - rank(d^1)
             = (729 - rank(d^2)) - rank(d^1).
    """
    d1 = bar_complex["d1"]
    d2 = bar_complex["d2"]

    # Use a single prime for speed; consistency across primes is checked
    # in test_rank_consistent_across_primes.
    p = 7
    q_p = cert.find_cube_root_mod_p(p)
    A1 = cert.reduce_to_numpy(d1, cert.DIM * cert.DIM, cert.DIM, p, q_p)
    rank1 = cert.rank_mod_p(A1, p)
    A2 = cert.reduce_to_numpy(d2, cert.DIM ** 3, cert.DIM * cert.DIM, p, q_p)
    rank2 = cert.rank_mod_p(A2, p)

    dim_ker_d2 = cert.DIM * cert.DIM - rank2
    dim_hh2 = dim_ker_d2 - rank1

    assert dim_hh2 == 3, \
        f"dim HH^2 = {dim_hh2} (rank(d^1)={rank1}, rank(d^2)={rank2}), expected 3"


def test_rank_values_match_certified(bar_complex):
    """The certified rank values match the known floating-point computation."""
    d1 = bar_complex["d1"]
    d2 = bar_complex["d2"]
    p = 7
    q_p = cert.find_cube_root_mod_p(p)
    A1 = cert.reduce_to_numpy(d1, cert.DIM * cert.DIM, cert.DIM, p, q_p)
    rank1 = cert.rank_mod_p(A1, p)
    A2 = cert.reduce_to_numpy(d2, cert.DIM ** 3, cert.DIM * cert.DIM, p, q_p)
    rank2 = cert.rank_mod_p(A2, p)
    # From the floating-point SVD computation in verify_sl2_hh2.py.
    assert rank1 == 27, f"rank(d^1) = {rank1}, expected 27"
    assert rank2 == 699, f"rank(d^2) = {rank2}, expected 699"


def test_script_runs():
    """Run the full certification script as a subprocess and check it succeeds.

    This is the end-to-end test: it runs scripts/certify_a1_exact.py, checks
    the exit code is 0, and verifies the certification message appears in
    stdout.  The script also writes its output to
    scripts/certify_a1_output.txt.
    """
    import subprocess
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, \
        f"Script exited {result.returncode}:\n{result.stderr}"
    assert "CERTIFIED" in result.stdout, \
        f"Certification message not found in output:\n{result.stdout}"
    assert "dim HH^2(u_q(sl_2), C) = 3" in result.stdout, \
        f"dim HH^2 = 3 not found in output:\n{result.stdout}"
    assert "RANKS CONSISTENT across all 11 primes" in result.stdout, \
        f"Consistency message not found in output:\n{result.stdout}"
