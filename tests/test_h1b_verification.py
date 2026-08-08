"""Tests for the H̃¹_b(B⁺) computation, verifying the Cholesky-shortcut bug fix.

Background:
    compute_h1b_bplus_sl3.py originally used a Cholesky-based shortcut to test
    whether the Gram matrix G = A^* A is positive definite. The shortcut was
    BUGGY: when G has small-but-positive "zero" eigenvalues (numerical roundoff
    ~1e-13 instead of exactly 0), LAPACK's ?potrf succeeds (it has no tolerance),
    reporting G as numerically positive definite and giving rank = n_cols
    (nullity = 0). For sl_3 at ℓ=3, this masked the 2 real cocycles in shift
    (0,0), reporting dim H̃¹_b = 0 instead of the correct value 2.

    The fix (verified by these tests) replaces the Cholesky shortcut with
    shift-invert eigsh on the sparse Gram matrix, which reliably finds
    small eigenvalues near zero.

These tests confirm:
    1. The sl_2 cross-check still gives dim H̃¹_b = 1 (matching conjecture C(2,2)=1).
    2. The sl_3 computation gives dim H̃¹_b = 2 (NOT 0 from the bug, NOT 3 from
       the conjecture's structural prediction — the structural prediction is
       refuted at A_2).
    3. The Cholesky-shortcut bug is reproducible: a known rank-deficient matrix
       with small-but-positive zero eigenvalues is mishandled by Cholesky but
       correctly identified by eigsh.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest
from scipy.linalg import cholesky, eigvalsh
from scipy.sparse.linalg import eigsh
from scipy import sparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")


def _load_script(name):
    path = os.path.join(SCRIPTS_DIR, name)
    assert os.path.exists(path), f"Missing {path}"
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name.replace(".py", "")] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Cholesky-shortcut bug regression test ----------------------------------

def test_cholesky_shortcut_bug_reproducible():
    """Demonstrate the Cholesky-shortcut bug on a small mock matrix.

    Construct a Hermitian PSD matrix G with 2 zero eigenvalues that are
    small-but-positive (~1e-13) due to added noise. Cholesky should spuriously
    succeed on this matrix, reporting it as positive definite. The eigvalsh-based
    fallback correctly identifies the 2 zero eigenvalues.
    """
    # Construct a matrix with true rank 8 and 2 zero eigenvalues (n=10).
    # G = U D U^*, where D has 8 entries of ~1 and 2 entries of ~1e-13.
    rng = np.random.default_rng(42)
    n = 10
    # Random complex matrix, then QR to get unitary
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    D = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-13, 1e-14])
    G = Q @ D @ Q.conj().T
    G = (G + G.conj().T) * 0.5  # symmetrize

    # True rank = 8 (eight non-zero eigenvalues)
    true_rank = 8

    # Method 1: Cholesky-based shortcut (the buggy approach).
    # If G were exactly PSD with zero eigenvalues, Cholesky would fail.
    # But because the "zero" eigenvalues are small-but-positive, Cholesky
    # succeeds and reports the matrix as positive definite.
    cholesky_succeeded = False
    try:
        L = cholesky(G, lower=True, check_finite=False)
        cholesky_succeeded = True
    except np.linalg.LinAlgError:
        cholesky_succeeded = False
    # Demonstrate the bug: Cholesky spuriously succeeds.
    assert cholesky_succeeded, (
        "Cholesky was expected to spuriously succeed here (the bug). "
        "If it failed, the LAPACK version may differ; the eigvalsh test below "
        "is the more important check."
    )
    # The Cholesky-based shortcut would report rank = n_cols = 10 (WRONG).
    cholesky_rank = n if cholesky_succeeded else None
    assert cholesky_rank == 10, "Cholesky-based shortcut gives wrong rank."

    # Method 2: eigvalsh-based approach (the fixed code's fallback).
    all_eigs = np.sort(np.abs(eigvalsh(G, check_finite=False)))[::-1]
    largest = float(all_eigs[0])
    tol = n * largest * 1e-10
    n_zero = int(np.sum(all_eigs < tol))
    eigvalsh_rank = n - n_zero

    # eigvalsh correctly identifies the 2 zero eigenvalues.
    assert eigvalsh_rank == true_rank, (
        f"eigvalsh-based rank = {eigvalsh_rank}, expected {true_rank}. "
        f"Smallest eigvals: {all_eigs[-5:]}, tol = {tol}"
    )

    # Demonstrate that Cholesky-based shortcut and eigvalsh give different answers.
    assert cholesky_rank != eigvalsh_rank, (
        "Cholesky and eigvalsh give the same rank — bug demonstration failed."
    )


def test_cholesky_shortcut_bug_reproducible_sparse():
    """Same as above but using sparse eigsh (the actual fix's method).

    Uses a 20×20 matrix to ensure k < n-1 (eigsh constraint).
    """
    rng = np.random.default_rng(123)
    n = 20
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    # 2 zero eigenvalues
    diag_vals = [1.0] * 18 + [1e-13, 1e-14]
    D = np.diag(diag_vals)
    G = Q @ D @ Q.conj().T
    G = (G + G.conj().T) * 0.5

    true_rank = 18

    # Cholesky spuriously succeeds
    cholesky_succeeded = False
    try:
        L = cholesky(G, lower=True, check_finite=False)
        cholesky_succeeded = True
    except np.linalg.LinAlgError:
        cholesky_succeeded = False
    assert cholesky_succeeded, "Cholesky was expected to spuriously succeed."

    # eigsh with sigma=0 finds the 2 zero eigenvalues
    G_sparse = sparse.csr_matrix(G)
    k = 5  # need k < n-1 = 19
    eigs_small = eigsh(G_sparse, k=k, sigma=0, which='LM',
                       return_eigenvectors=False, tol=1e-12, maxiter=10000)
    eigs_small = np.sort(np.abs(eigs_small))
    eigs_large = eigsh(G_sparse, k=1, which='LM',
                       return_eigenvectors=False, tol=1e-6, maxiter=10000)
    largest = float(np.abs(eigs_large[0]))
    tol = n * largest * 1e-10
    n_zero = int(np.sum(eigs_small < tol))
    eigsh_rank = n - n_zero

    assert eigsh_rank == true_rank, (
        f"eigsh-based rank = {eigsh_rank}, expected {true_rank}. "
        f"Smallest eigvals: {eigs_small}, tol = {tol}"
    )


# --- sl_2 cross-check (fast, ~1s) -------------------------------------------

def test_sl2_h1b_equals_1():
    """sl_2 cross-check: dim H̃¹_b(B⁺(u_q(sl_2))) = 1 (matches conjecture C(2,2)=1).

    This is the most important sanity check: the sl_2 case is verified
    independently and matches the conjecture. The sl_3 computation uses the
    same algorithmic structure (just with more generators), so if sl_2 works,
    we have confidence that sl_3's logic (after the bug fix) is also correct.
    """
    sl2 = _load_script("compute_h1b_bplus_sl2.py")
    dim_ker = sl2.compute(sl2.DIM, verbose=False)
    # sl_2 should give dim H̃¹_b = 1 (matches conjecture C(2,2) = 1).
    assert dim_ker == 1, (
        f"sl_2 H̃¹_b = {dim_ker}, expected 1 (conjecture C(2,2) = 1). "
        f"This is the sl_2 cross-check; if it fails, the algorithm is broken."
    )


# --- sl_3 main computation (slow, ~90s; marked slow) ------------------------

@pytest.mark.slow
def test_sl3_h1b_equals_2_after_bugfix():
    """sl_3 main computation: dim H̃¹_b(B⁺(u_q(sl_3))) = 2 after the bug fix.

    Before the bug fix (Cholesky shortcut): dim = 0 (WRONG; missed 2 cocycles).
    After the bug fix (eigsh-based):       dim = 2 (CORRECT).

    The conjecture's structural prediction was 3 (= C(3, 2)), so this test
    REFUTES that structural prediction at A_2. The full conjecture
    "dim HH²(u_q(sl_3)) = 9" may still hold (with structural split (2, 7)
    instead of (3, 6)), but is not directly verifiable in this sandbox.
    """
    sl3 = _load_script("compute_h1b_bplus_sl3.py")
    dim_ker, per_shift = sl3.compute(sl3.DIM, verbose=False)
    assert dim_ker == 2, (
        f"sl_3 H̃¹_b = {dim_ker}, expected 2 (after Cholesky-shortcut bug fix). "
        f"If you got 0, the Cholesky-shortcut bug has returned. "
        f"If you got 3, the conjecture's structural prediction was right after all."
    )
    # The 2 cocycles should all be at weight shift (0, 0).
    shift_nullities = {s: nullity for (s, _, _, _, _, nullity, _) in per_shift}
    assert shift_nullities[(0, 0)] == 2, (
        f"Expected nullity = 2 at shift (0,0), got {shift_nullities[(0, 0)]}."
    )
    # All other 8 shifts should have nullity = 0.
    for s, n in shift_nullities.items():
        if s != (0, 0):
            assert n == 0, f"Expected nullity = 0 at shift {s}, got {n}."


# --- Algebra / coalgebra verification (fast) --------------------------------

def test_sl3_algebra_coalgebra_invariants():
    """Verify the sl_3 multiplication and comultiplication tables are correct.

    These are the foundational invariants: Δ(E12) check, coassociativity,
    and counitality. If any of these fail, the coboundary matrix is wrong,
    which would invalidate the dim H̃¹_b computation entirely.
    """
    sl3 = _load_script("compute_h1b_bplus_sl3.py")
    ms = sl3.build_mult(sl3.DIM)
    sl3.ms_global = ms
    Delta = sl3.build_delta(sl3.DIM)

    # Δ(E12) = Δ(E1)Δ(E2) - q Δ(E2)Δ(E1) (Lusztig root vector coproduct)
    e12_err = sl3.verify_delta_E12(Delta, sl3.DIM)
    assert e12_err < 1e-10, f"Δ(E12) check failed: max error = {e12_err}"

    # Coassociativity: (Δ⊗1)Δ = (1⊗Δ)Δ
    coassoc_err, _ = sl3.verify_coassoc(Delta, sl3.DIM, n_samples=10)
    assert coassoc_err < 1e-10, f"Coassociativity failed: max error = {coassoc_err}"

    # Counitality: (ε⊗1)Δ = id = (1⊗ε)Δ
    epsilon = sl3.build_epsilon(sl3.DIM)
    counital_err = sl3.verify_counital(Delta, sl3.DIM, epsilon)
    assert counital_err < 1e-10, f"Counitality failed: max error = {counital_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
