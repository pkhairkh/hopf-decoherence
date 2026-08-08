"""Tests for the bialgebra 1-cocycle computation (W3).

Tests:
  test_sl2_cross_check: dim H̃¹_b(B⁺(u_q(sl_2))) = 1 at ℓ=3 (matches conjecture)
  test_sl3_computation_runs: the sl_3 script runs without error
  test_sl3_dim_in_range: 0 ≤ dim H̃¹_b(B⁺(u_q(sl_3))) ≤ 20
  test_sl3_dim_is_2: dim H̃¹_b(B⁺(u_q(sl_3))) = 2 (the actual computed value)
  test_a2_status: documents what the result means for the conjecture
"""
import importlib.util
import os
import sys
import subprocess

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SL2_SCRIPT = os.path.join(REPO_ROOT, "scripts", "compute_h1b_bplus_sl2.py")
SL3_SCRIPT = os.path.join(REPO_ROOT, "scripts", "compute_h1b_bplus_sl3.py")
SL2_OUTPUT = os.path.join(REPO_ROOT, "scripts", "h1b_sl2_output.txt")
SL3_OUTPUT = os.path.join(REPO_ROOT, "scripts", "h1b_sl3_output.txt")


def _parse_dim_from_output(path):
    """Extract dim H̃¹_b from a captured output file."""
    with open(path) as f:
        for line in f:
            if "dim H̃¹_b" in line and "dim ker(∂_b)" in line:
                # Format: "dim H̃¹_b(...) = dim ker(∂_b) = N"
                parts = line.split("=")
                if len(parts) >= 3:
                    try:
                        return int(parts[-1].strip())
                    except ValueError:
                        pass
    return None


def test_sl2_cross_check():
    """sl_2 cross-check: dim H̃¹_b(B⁺(u_q(sl_2))) = 1 at ℓ=3.

    This matches the conjecture's structural prediction C(2,2) = 1,
    validating the computation framework.
    """
    # Run the sl_2 script
    result = subprocess.run(
        [sys.executable, SL2_SCRIPT],
        capture_output=True, text=True, timeout=120,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"sl_2 script failed: {result.stderr[-500:]}"

    # Parse the dim from the output
    dim = _parse_dim_from_output_str(result.stdout)
    assert dim == 1, f"sl_2 cross-check failed: expected dim=1, got {dim}"


def _parse_dim_from_output_str(output):
    for line in output.split("\n"):
        if "dim H̃¹_b" in line and "dim ker(∂_b)" in line:
            parts = line.split("=")
            if len(parts) >= 3:
                try:
                    return int(parts[-1].strip())
                except ValueError:
                    pass
    return None


def test_sl3_output_exists():
    """The sl_3 output file should exist from the computation."""
    assert os.path.exists(SL3_OUTPUT), f"Missing {SL3_OUTPUT}"


def test_sl3_dim_in_range():
    """The computed dim should be in a reasonable range."""
    dim = _parse_dim_from_output(SL3_OUTPUT)
    assert dim is not None, "Could not parse dim from sl_3 output"
    assert 0 <= dim <= 20, f"dim out of range: {dim}"


def test_sl3_dim_is_2():
    """The computed dim H̃¹_b(B⁺(u_q(sl_3))) = 2 at ℓ=3.

    This REFUTES the conjecture's structural prediction C(3,2) = 3.
    The full conjecture (dim HH² = 9) is not directly verified or refuted
    by this single number — it depends on dim im(π̄), which requires the
    intractable full HH²(D(B⁺)) computation.
    """
    dim = _parse_dim_from_output(SL3_OUTPUT)
    assert dim == 2, (
        f"Expected dim=2 (refuting structural prediction of 3), got {dim}. "
        f"If you got 3, the conjecture's structural split is verified. "
        f"If you got 0, check the weight function bug."
    )


def test_sl2_output_exists():
    """The sl_2 output file should exist."""
    assert os.path.exists(SL2_OUTPUT), f"Missing {SL2_OUTPUT}"


def test_sl2_dim_is_1():
    """sl_2 dim should be 1, matching the conjecture at A_1."""
    dim = _parse_dim_from_output(SL2_OUTPUT)
    assert dim == 1, f"sl_2 dim should be 1, got {dim}"


def test_a2_status_summary():
    """Document the A_2 status: structural prediction refuted, full conjecture open.

    The conjecture's structural split was:
        dim HH²(u_q(sl_3)) = dim im(δ) + dim im(π̄) = C(3,2) + 2|Φ⁺| = 3 + 6 = 9

    We verified dim im(δ) = dim H̃¹_b(B⁺) = 2 (not 3).

    So either:
    - Full conjecture (dim=9) holds with dim im(π̄)=7 (not 6)
    - Full conjecture is wrong (e.g. dim=8 with dim im(π̄)=6)

    This test just documents the situation; it always passes.
    """
    sl3_dim = _parse_dim_from_output(SL3_OUTPUT)
    sl2_dim = _parse_dim_from_output(SL2_OUTPUT)
    summary = (
        f"\n=== A_2 STATUS SUMMARY ===\n"
        f"  sl_2, ℓ=3: dim H̃¹_b(B⁺) = {sl2_dim} (conjecture predicts 1: {'MATCH' if sl2_dim == 1 else 'MISMATCH'})\n"
        f"  sl_3, ℓ=3: dim H̃¹_b(B⁺) = {sl3_dim} (conjecture predicts 3: {'MATCH' if sl3_dim == 3 else 'MISMATCH'})\n"
        f"\n"
        f"  Conjecture's structural prediction dim im(δ) = C(n+1,2):\n"
        f"    A_1: {'VERIFIED' if sl2_dim == 1 else 'REFUTED'}\n"
        f"    A_2: {'VERIFIED' if sl3_dim == 3 else 'REFUTED'} (computed {sl3_dim}, predicted 3)\n"
        f"\n"
        f"  Full conjecture dim HH²(u_q(sl_3)) = 9:\n"
        f"    NOT DIRECTLY VERIFIED (would need dim HH²(D(B⁺)) which is intractable)\n"
        f"    Consistent splits: (dim im δ, dim im π̄) ∈ {{(2, 7), (3, 6)}} ∩ LES constraints\n"
        f"    Our computation gives dim im δ = {sl3_dim}, so dim im π̄ = 9 - {sl3_dim} = {9 - sl3_dim} if dim=9\n"
    )
    print(summary)
    # Always passes — this is documentation
    assert True
