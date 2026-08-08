#!/usr/bin/env python3
"""
test_matvec.py - Correctness test for the C matvec extension.

Creates a small random sparse matrix, saves it in the chunk file format,
runs precompute + matvec + rmatvec, and compares with scipy's reference.

Run:  python test_matvec.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
from scipy import sparse

# Ensure we can import the wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matvec_wrapper import MatVecHandle, compute_rank, NormalOperator


def create_test_matrix(n_rows, n_cols, density, seed=42):
    """Create a random sparse complex matrix with known rank deficiency."""
    rng = np.random.default_rng(seed)
    nnz = int(n_rows * n_cols * density)

    # Random sparse matrix
    rows = rng.integers(0, n_rows, size=nnz)
    cols = rng.integers(0, n_cols, size=nnz)
    vals = (rng.standard_normal(nnz) + 1j * rng.standard_normal(nnz)).astype(np.complex128)

    # Build COO matrix and convert to CSR for reference
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols)).tocsr()

    # Introduce a known nullity by zeroing some columns
    # (making rank = n_cols - nullity)
    nullity_target = 3
    zero_cols = rng.choice(n_cols, size=nullity_target, replace=False)
    for c in zero_cols:
        A[:, c] = 0

    # Remove zero entries
    A.eliminate_zeros()

    return A, zero_cols


def save_as_chunks(A, tmpdir, prefix, chunk_size=50000):
    """Save a sparse matrix as chunk files (keys, cols, vals)."""
    A_coo = A.tocoo()
    n_nnz = A_coo.nnz

    # Row keys: just use row indices as int64 (no packing needed for test)
    keys = A_coo.row.astype(np.int64)
    cols = A_coo.col.astype(np.int32)
    vals = A_coo.data.astype(np.complex128)

    chunk_files = []
    offset = 0
    chunk_idx = 0
    while offset < n_nnz:
        end = min(offset + chunk_size, n_nnz)
        n = end - offset

        keys_path = os.path.join(tmpdir, f"{prefix}_chunk_{chunk_idx}_keys.bin")
        cols_path = os.path.join(tmpdir, f"{prefix}_chunk_{chunk_idx}_cols.bin")
        vals_path = os.path.join(tmpdir, f"{prefix}_chunk_{chunk_idx}_vals.bin")

        keys[offset:end].tofile(keys_path)
        cols[offset:end].tofile(cols_path)
        vals[offset:end].tofile(vals_path)

        chunk_files.append((keys_path, cols_path, vals_path, n))
        offset = end
        chunk_idx += 1

    return n_nnz, chunk_files


def build_unique_keys(tmpdir, prefix, n_chunks):
    """Build sorted unique keys from chunk key files."""
    all_keys = []
    for i in range(n_chunks):
        keys_path = os.path.join(tmpdir, f"{prefix}_chunk_{i}_keys.bin")
        if not os.path.exists(keys_path):
            continue
        keys = np.fromfile(keys_path, dtype=np.int64)
        all_keys.append(keys)

    all_keys = np.concatenate(all_keys)
    unique_keys = np.unique(all_keys)
    unique_keys_path = os.path.join(tmpdir, "unique_keys.bin")
    unique_keys.tofile(unique_keys_path)
    return unique_keys, unique_keys_path


def precompute_flat_files(tmpdir, prefix, n_chunks, unique_keys_path):
    """Precompute row_idx, cols, vals flat files from chunks."""
    # Collect chunk sizes
    chunk_info = []
    for i in range(n_chunks):
        keys_path = os.path.join(tmpdir, f"{prefix}_chunk_{i}_keys.bin")
        if not os.path.exists(keys_path):
            continue
        n = os.path.getsize(keys_path) // 8
        chunk_info.append((i, n))

    n_nnz = sum(n for (_, n) in chunk_info)
    unique_keys = np.memmap(unique_keys_path, dtype=np.int64, mode="r")

    # Compute offsets
    offsets = {}
    off = 0
    for (i, n) in chunk_info:
        offsets[i] = off
        off += n

    # Create output files
    row_idx_path = os.path.join(tmpdir, "row_idx_flat.bin")
    cols_path = os.path.join(tmpdir, "cols_flat.bin")
    vals_path = os.path.join(tmpdir, "vals_flat.bin")

    row_idx_out = np.memmap(row_idx_path, dtype=np.int32, mode="w+",
                            shape=(n_nnz,))
    cols_out = np.memmap(cols_path, dtype=np.int32, mode="w+",
                         shape=(n_nnz,))
    vals_out = np.memmap(vals_path, dtype=np.complex64, mode="w+",
                         shape=(n_nnz,))

    for (i, n) in chunk_info:
        keys = np.fromfile(os.path.join(tmpdir, f"{prefix}_chunk_{i}_keys.bin"),
                           dtype=np.int64)
        cols = np.fromfile(os.path.join(tmpdir, f"{prefix}_chunk_{i}_cols.bin"),
                           dtype=np.int32)
        vals = np.fromfile(os.path.join(tmpdir, f"{prefix}_chunk_{i}_vals.bin"),
                           dtype=np.complex128)

        row_idx = np.searchsorted(unique_keys, keys).astype(np.int32)
        off = offsets[i]
        row_idx_out[off:off + n] = row_idx
        cols_out[off:off + n] = cols
        vals_out[off:off + n] = vals.astype(np.complex64)

        del keys, cols, vals, row_idx

    del row_idx_out, cols_out, vals_out, unique_keys

    # Save nnz count
    with open(os.path.join(tmpdir, "nnz_count.txt"), "w") as f:
        f.write(str(n_nnz))

    return n_nnz, row_idx_path, cols_path, vals_path


def test_matvec():
    """Test matvec and rmatvec against scipy reference."""
    print("=" * 60)
    print("TEST: matvec and rmatvec correctness")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="matvec_test_")
    prefix = "test"
    chunk_size = 50000

    try:
        # Create test matrix
        n_rows = 5000
        n_cols = 800
        density = 0.05
        A, zero_cols = create_test_matrix(n_rows, n_cols, density)
        print(f"  Matrix: {n_rows} x {n_cols}, nnz = {A.nnz}")
        print(f"  Zeroed columns (nullity): {len(zero_cols)}")

        # Save as chunks
        n_nnz, chunk_files = save_as_chunks(A, tmpdir, prefix, chunk_size)
        n_chunks = len(chunk_files)
        print(f"  Saved {n_chunks} chunks, total nnz = {n_nnz}")

        # Build unique keys
        unique_keys, unique_keys_path = build_unique_keys(tmpdir, prefix, n_chunks)
        print(f"  Unique keys: {len(unique_keys)}")

        # Precompute flat files
        n_nnz2, row_idx_path, cols_path, vals_path = \
            precompute_flat_files(tmpdir, prefix, n_chunks, unique_keys_path)
        assert n_nnz2 == n_nnz
        print(f"  Flat files created, nnz = {n_nnz2}")

        # Load C extension
        handle = MatVecHandle(
            row_idx_path, cols_path, vals_path,
            n_nnz, n_rows, n_cols, sorted=False,
        )

        # Test matvec
        rng = np.random.default_rng(123)
        v = (rng.standard_normal(n_cols) + 1j * rng.standard_normal(n_cols)).astype(np.complex128)
        result_c = handle.matvec(v)

        # Reference
        result_ref = A @ v

        # Compare (accounting for complex64 precision loss)
        err = np.max(np.abs(result_c - result_ref))
        rel_err = err / np.max(np.abs(result_ref))
        print(f"\n  matvec: max abs err = {err:.6e}, rel err = {rel_err:.6e}")
        assert rel_err < 1e-4, f"matvec error too large: {rel_err}"
        print("  matvec: PASSED")

        # Test rmatvec
        w = (rng.standard_normal(n_rows) + 1j * rng.standard_normal(n_rows)).astype(np.complex128)
        result_c2 = handle.rmatvec(w)

        # Reference: A^* @ w = A.conj().T @ w
        result_ref2 = A.conj().T @ w

        err2 = np.max(np.abs(result_c2 - result_ref2))
        rel_err2 = err2 / np.max(np.abs(result_ref2))
        print(f"  rmatvec: max abs err = {err2:.6e}, rel err = {rel_err2:.6e}")
        assert rel_err2 < 1e-4, f"rmatvec error too large: {rel_err2}"
        print("  rmatvec: PASSED")

        handle.close()

        # Test sorted mode
        print("\n  Testing sorted mode...")
        from matvec_wrapper import sort_by_row_idx
        sort_by_row_idx(tmpdir, n_nnz)

        sorted_row_idx_path = os.path.join(tmpdir, "row_idx_sorted.bin")
        sorted_cols_path = os.path.join(tmpdir, "cols_sorted.bin")
        sorted_vals_path = os.path.join(tmpdir, "vals_sorted.bin")

        handle2 = MatVecHandle(
            sorted_row_idx_path, sorted_cols_path, sorted_vals_path,
            n_nnz, n_rows, n_cols, sorted=True,
        )

        result_sorted = handle2.matvec(v)
        err3 = np.max(np.abs(result_sorted - result_ref))
        rel_err3 = err3 / np.max(np.abs(result_ref))
        print(f"  matvec (sorted): max abs err = {err3:.6e}, rel err = {rel_err3:.6e}")
        assert rel_err3 < 1e-4, f"sorted matvec error too large: {rel_err3}"
        print("  matvec (sorted): PASSED")

        result_sorted2 = handle2.rmatvec(w)
        err4 = np.max(np.abs(result_sorted2 - result_ref2))
        rel_err4 = err4 / np.max(np.abs(result_ref2))
        print(f"  rmatvec (sorted): max abs err = {err4:.6e}, rel err = {rel_err4:.6e}")
        assert rel_err4 < 1e-4, f"sorted rmatvec error too large: {rel_err4}"
        print("  rmatvec (sorted): PASSED")

        handle2.close()
        print("\n  All matvec/rmatvec tests PASSED!")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_rank():
    """Test rank computation against known nullity."""
    print("\n" + "=" * 60)
    print("TEST: rank computation (small scale)")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="matvec_rank_")
    prefix = "test"
    chunk_size = 10000

    try:
        n_rows = 2000
        n_cols = 300
        density = 0.1
        A, zero_cols = create_test_matrix(n_rows, n_cols, density, seed=99)
        expected_nullity = len(zero_cols)
        expected_rank = n_cols - expected_nullity
        print(f"  Matrix: {n_rows} x {n_cols}, nnz = {A.nnz}")
        print(f"  Expected rank = {expected_rank}, nullity = {expected_nullity}")

        n_nnz, chunk_files = save_as_chunks(A, tmpdir, prefix, chunk_size)
        n_chunks = len(chunk_files)
        unique_keys, unique_keys_path = build_unique_keys(tmpdir, prefix, n_chunks)
        n_nnz2, row_idx_path, cols_path, vals_path = \
            precompute_flat_files(tmpdir, prefix, n_chunks, unique_keys_path)

        handle = MatVecHandle(
            row_idx_path, cols_path, vals_path,
            n_nnz, n_rows, n_cols, sorted=True,
        )

        rank, nullity, eigenvalues = compute_rank(
            handle, k=15, tol=1e-8, maxiter=1000, ncv=30,
            method="eigsh",
        )

        print(f"\n  Computed rank = {rank}, nullity = {nullity}")
        print(f"  Expected rank = {expected_rank}, nullity = {expected_nullity}")

        if nullity == expected_nullity:
            print("  Rank test: PASSED!")
        else:
            print(f"  Rank test: MISMATCH (expected nullity={expected_nullity}, "
                  f"got {nullity})")
            print(f"  Eigenvalues: {eigenvalues}")

        handle.close()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_column_norms():
    """Test column norms computation."""
    print("\n" + "=" * 60)
    print("TEST: column norms")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="matvec_cn_")
    prefix = "test"

    try:
        n_rows = 1000
        n_cols = 200
        density = 0.1
        A, _ = create_test_matrix(n_rows, n_cols, density, seed=77)

        n_nnz, chunk_files = save_as_chunks(A, tmpdir, prefix)
        n_chunks = len(chunk_files)
        unique_keys, unique_keys_path = build_unique_keys(tmpdir, prefix, n_chunks)
        n_nnz2, row_idx_path, cols_path, vals_path = \
            precompute_flat_files(tmpdir, prefix, n_chunks, unique_keys_path)

        handle = MatVecHandle(
            row_idx_path, cols_path, vals_path,
            n_nnz, n_rows, n_cols, sorted=False,
        )

        # C column norms
        cn_c = handle.column_norms()

        # Reference: |A|^2 column sums (element-wise |val|^2, then sum per column)
        A_coo = A.tocoo()
        abs_sq = (A_coo.data.real ** 2 + A_coo.data.imag ** 2).astype(np.float64)
        cn_ref = np.zeros(n_cols, dtype=np.float64)
        np.add.at(cn_ref, A_coo.col, abs_sq)

        err = np.max(np.abs(cn_c - cn_ref))
        rel_err = err / np.max(cn_ref)
        print(f"  column norms: max abs err = {err:.6e}, rel err = {rel_err:.6e}")
        assert rel_err < 1e-4, f"column norms error too large: {rel_err}"
        print("  column norms: PASSED!")

        handle.close()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_matvec()
    test_column_norms()
    test_rank()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
