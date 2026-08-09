#!/usr/bin/env python3
"""
matvec_wrapper.py - Python ctypes wrapper for the matvec C extension.

Provides:
  1. MatVecHandle   - loads mmap'd data, exposes matvec/rmatvec
  2. precompute     - converts chunk files to flat (row_idx, cols, vals) files
  3. sort_by_row_idx- optional sort step for 15x faster matvec
  4. NormalOperator - scipy LinearOperator for A^* A (for eigsh)
  5. compute_rank   - eigensolver strategy to find nullity

Usage on the target machine (16 cores, 125 GB RAM):

    # Step 1: Precompute row_idx from keys + unique_keys
    python matvec_wrapper.py precompute \\
        --tmpdir /tmp/h2b_sl3_box2ztl4 \\
        --prefix shift_0_0_w0 \\
        --n-chunks 702 \\
        --unique-keys /tmp/h2b_sl3_box2ztl4/unique_keys.bin

    # Step 2 (optional but recommended): sort by row_idx for 15x speedup
    python matvec_wrapper.py sort \\
        --tmpdir /tmp/h2b_sl3_box2ztl4

    # Step 3: Compute rank
    python matvec_wrapper.py rank \\
        --tmpdir /tmp/h2b_sl3_box2ztl4 \\
        --n-rows 1568987626 \\
        --n-cols 9448324
"""

import argparse
import ctypes
import gc
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, svds

# =====================================================================
# 1. ctypes bindings
# =====================================================================

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matvec.so")

def _load_lib():
    lib = ctypes.CDLL(_LIB_PATH)

    # void *load_matvec_data(const char*, const char*, const char*,
    #                        int64_t, int64_t, int64_t)
    lib.load_matvec_data.restype = ctypes.c_void_p
    lib.load_matvec_data.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]

    # void free_matvec_data(void*)
    lib.free_matvec_data.restype = None
    lib.free_matvec_data.argtypes = [ctypes.c_void_p]

    # void prefault_data(void*)
    lib.prefault_data.restype = None
    lib.prefault_data.argtypes = [ctypes.c_void_p]

    # void set_sorted(void*, int)
    lib.set_sorted.restype = None
    lib.set_sorted.argtypes = [ctypes.c_void_p, ctypes.c_int]

    # void matvec(void*, const void*, void*)
    lib.matvec.restype = None
    lib.matvec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    # void rmatvec(void*, const void*, void*)
    lib.rmatvec.restype = None
    lib.rmatvec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    # void compute_column_norms(void*, void*)
    lib.compute_column_norms.restype = None
    lib.compute_column_norms.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    # int64_t get_n_nnz / get_n_rows / get_n_cols(void*)
    for name in ("get_n_nnz", "get_n_rows", "get_n_cols"):
        getattr(lib, name).restype = ctypes.c_int64
        getattr(lib, name).argtypes = [ctypes.c_void_p]

    return lib


_LIB = _load_lib()


# =====================================================================
# 2. MatVecHandle
# =====================================================================

class MatVecHandle:
    """
    Wraps the C MVHandle. Holds mmap'd data and preallocated buffers.

    Parameters
    ----------
    row_idx_path : str
        Path to int32[n_nnz] file.
    cols_path : str
        Path to int32[n_nnz] file.
    vals_path : str
        Path to complex64[n_nnz] file.
    n_nnz : int
        Total nonzero count.
    n_rows : int
        Matrix row dimension.
    n_cols : int
        Matrix column dimension.
    sorted : bool
        Whether the data is sorted by row_idx (enables fast path).
    prefault : bool
        Whether to prefault all mmap pages (sequential read, ~64s for 64GB).
    """

    def __init__(self, row_idx_path, cols_path, vals_path,
                 n_nnz, n_rows, n_cols, sorted=False, prefault=False):
        self.n_nnz = n_nnz
        self.n_rows = n_rows
        self.n_cols = n_cols

        self._ptr = _LIB.load_matvec_data(
            row_idx_path.encode("utf-8"),
            cols_path.encode("utf-8"),
            vals_path.encode("utf-8"),
            ctypes.c_int64(n_nnz),
            ctypes.c_int64(n_rows),
            ctypes.c_int64(n_cols),
        )
        if not self._ptr:
            raise RuntimeError("load_matvec_data failed")

        if sorted:
            _LIB.set_sorted(self._ptr, 1)
        if prefault:
            print("  prefaulting mmap pages...", flush=True)
            t0 = time.time()
            _LIB.prefault_data(self._ptr)
            print(f"  prefault done [{time.time()-t0:.1f}s]", flush=True)

        # Preallocate the matvec result buffer (n_rows x complex128).
        # This is ~25 GB for the full problem; allocated once and reused.
        self._matvec_buf = np.zeros(n_rows, dtype=np.complex128)

    @property
    def matvec_buf(self):
        """The preallocated matvec result buffer (n_rows, complex128)."""
        return self._matvec_buf

    def matvec(self, v):
        """
        Compute result = A @ v.
        v: complex128 [n_cols]
        Returns: complex128 [n_rows] (the internal buffer, do not modify).
        """
        v = np.ascontiguousarray(v, dtype=np.complex128)
        _LIB.matvec(
            self._ptr,
            v.ctypes.data_as(ctypes.c_void_p),
            self._matvec_buf.ctypes.data_as(ctypes.c_void_p),
        )
        return self._matvec_buf

    def rmatvec(self, v):
        """
        Compute result = A^* @ v.
        v: complex128 [n_rows]
        Returns: a freshly allocated complex128 [n_cols] array.
        """
        v = np.ascontiguousarray(v, dtype=np.complex128)
        result = np.zeros(self.n_cols, dtype=np.complex128)
        _LIB.rmatvec(
            self._ptr,
            v.ctypes.data_as(ctypes.c_void_p),
            result.ctypes.data_as(ctypes.c_void_p),
        )
        return result

    def column_norms(self):
        """Compute column norms (diagonal of A^* A). Returns double [n_cols]."""
        result = np.zeros(self.n_cols, dtype=np.float64)
        _LIB.compute_column_norms(
            self._ptr,
            result.ctypes.data_as(ctypes.c_void_p),
        )
        return result

    def close(self):
        if self._ptr:
            _LIB.free_matvec_data(self._ptr)
            self._ptr = None
        # Free the matvec buffer
        del self._matvec_buf
        self._matvec_buf = None
        gc.collect()

    def __del__(self):
        self.close()


# =====================================================================
# 3. Precompute: chunk files -> flat (row_idx, cols, vals) files
# =====================================================================

def precompute(tmpdir, prefix, n_chunks, unique_keys_path,
               n_workers=10, dtype_vals=np.complex64):
    """
    Convert chunk files to flat concatenated arrays.

    Reads:
      {tmpdir}/{prefix}_chunk_{i}_keys.bin  (int64)
      {tmpdir}/{prefix}_chunk_{i}_cols.bin  (int32)
      {tmpdir}/{prefix}_chunk_{i}_vals.bin  (complex128)

    Writes:
      {tmpdir}/row_idx_flat.bin   (int32[n_nnz])
      {tmpdir}/cols_flat.bin      (int32[n_nnz])
      {tmpdir}/vals_flat.bin      (complex64[n_nnz])

    Also writes {tmpdir}/nnz_count.txt with the total nonzero count.
    """
    import multiprocessing as mp

    t0 = time.time()
    print(f"=== Precompute: {n_chunks} chunks -> flat files ===", flush=True)

    # Step 1: scan chunk sizes and compute offsets
    print("  scanning chunk sizes...", flush=True)
    chunk_sizes = []
    for i in range(n_chunks):
        keys_path = os.path.join(tmpdir, f"{prefix}_chunk_{i}_keys.bin")
        if not os.path.exists(keys_path):
            print(f"  WARNING: {keys_path} not found, skipping", flush=True)
            continue
        n = os.path.getsize(keys_path) // 8  # int64
        chunk_sizes.append((i, n))

    n_nnz = sum(n for (_, n) in chunk_sizes)
    n_chunks_found = len(chunk_sizes)
    print(f"  found {n_chunks_found} chunks, total nnz = {n_nnz}", flush=True)

    # Step 2: create output files with correct size
    row_idx_path = os.path.join(tmpdir, "row_idx_flat.bin")
    cols_path = os.path.join(tmpdir, "cols_flat.bin")
    vals_path = os.path.join(tmpdir, "vals_flat.bin")

    # Preallocate output files
    for path, dtype, count in [
        (row_idx_path, np.int32, n_nnz),
        (cols_path, np.int32, n_nnz),
        (vals_path, dtype_vals, n_nnz),
    ]:
        with open(path, "wb") as f:
            f.seek(count * np.dtype(dtype).itemsize - 1)
            f.write(b"\x00")
        print(f"  created {path} ({count * np.dtype(dtype).itemsize / 1e9:.1f} GB)",
              flush=True)

    # Save nnz count
    with open(os.path.join(tmpdir, "nnz_count.txt"), "w") as f:
        f.write(str(n_nnz))

    # Step 3: compute offsets
    offsets = {}
    offset = 0
    for (i, n) in chunk_sizes:
        offsets[i] = offset
        offset += n

    # Step 4: process chunks in parallel
    print(f"  processing {n_chunks_found} chunks with {n_workers} workers...",
          flush=True)

    # We can't easily use multiprocessing with memmap writes (race conditions
    # on the same file from different processes). Instead, process chunks
    # sequentially but use numpy's vectorized searchsorted.
    # For true parallelism, each worker can process a subset of chunks
    # and write to non-overlapping regions of the output files.

    unique_keys = np.memmap(unique_keys_path, dtype=np.int64, mode="r")
    print(f"  unique_keys: {len(unique_keys)} entries", flush=True)

    # Open output files as memmaps
    row_idx_out = np.memmap(row_idx_path, dtype=np.int32, mode="r+",
                            shape=(n_nnz,))
    cols_out = np.memmap(cols_path, dtype=np.int32, mode="r+",
                         shape=(n_nnz,))
    vals_out = np.memmap(vals_path, dtype=dtype_vals, mode="r+",
                         shape=(n_nnz,))

    t1 = time.time()
    for idx, (chunk_i, n) in enumerate(chunk_sizes):
        keys_path = os.path.join(tmpdir, f"{prefix}_chunk_{chunk_i}_keys.bin")
        cols_chunk_path = os.path.join(tmpdir, f"{prefix}_chunk_{chunk_i}_cols.bin")
        vals_chunk_path = os.path.join(tmpdir, f"{prefix}_chunk_{chunk_i}_vals.bin")

        # Read chunk data
        keys = np.fromfile(keys_path, dtype=np.int64)
        cols_chunk = np.fromfile(cols_chunk_path, dtype=np.int32)
        vals_chunk = np.fromfile(vals_chunk_path, dtype=np.complex128)

        # Compute row_idx via searchsorted
        row_idx_chunk = np.searchsorted(unique_keys, keys).astype(np.int32)

        # Write to output files
        off = offsets[chunk_i]
        row_idx_out[off:off + n] = row_idx_chunk
        cols_out[off:off + n] = cols_chunk
        vals_out[off:off + n] = vals_chunk.astype(dtype_vals)

        del keys, cols_chunk, vals_chunk, row_idx_chunk

        if (idx + 1) % 50 == 0 or idx == len(chunk_sizes) - 1:
            elapsed = time.time() - t1
            rate = (idx + 1) / elapsed
            eta = (len(chunk_sizes) - idx - 1) / rate
            print(f"  [{idx+1}/{n_chunks_found}] {n} entries  "
                  f"[{elapsed:.0f}s, ETA {eta:.0f}s]", flush=True)

    del row_idx_out, cols_out, vals_out, unique_keys
    gc.collect()

    print(f"\n  Total nnz: {n_nnz}", flush=True)
    print(f"  Output files:", flush=True)
    print(f"    {row_idx_path} ({n_nnz * 4 / 1e9:.1f} GB)", flush=True)
    print(f"    {cols_path} ({n_nnz * 4 / 1e9:.1f} GB)", flush=True)
    print(f"    {vals_path} ({n_nnz * np.dtype(dtype_vals).itemsize / 1e9:.1f} GB)",
          flush=True)
    print(f"  Time: {time.time()-t0:.0f}s", flush=True)

    return n_nnz


# =====================================================================
# 4. Sort by row_idx (optional, 15x matvec speedup)
# =====================================================================

def sort_by_row_idx(tmpdir, n_nnz):
    """
    Sort the flat (row_idx, cols, vals) arrays by row_idx.

    This creates:
      {tmpdir}/row_idx_sorted.bin
      {tmpdir}/cols_sorted.bin
      {tmpdir}/vals_sorted.bin

    After sorting, set sorted=True when creating MatVecHandle.

    Memory: needs ~48 GB for the argsort index (int64[n_nnz] = 32 GB)
    plus the original arrays (mmap'd). Total ~80 GB peak.
    With 125 GB RAM, this fits.

    Time: ~10-30 minutes for 4B entries.
    """
    t0 = time.time()
    print(f"=== Sort by row_idx ({n_nnz} entries) ===", flush=True)

    row_idx_path = os.path.join(tmpdir, "row_idx_flat.bin")
    cols_path = os.path.join(tmpdir, "cols_flat.bin")
    vals_path = os.path.join(tmpdir, "vals_flat.bin")

    # mmap input arrays
    row_idx = np.memmap(row_idx_path, dtype=np.int32, mode="r", shape=(n_nnz,))
    cols = np.memmap(cols_path, dtype=np.int32, mode="r", shape=(n_nnz,))
    vals = np.memmap(vals_path, dtype=np.complex64, mode="r", shape=(n_nnz,))

    # Compute sort permutation
    print("  computing argsort...", flush=True)
    t1 = time.time()
    perm = np.argsort(row_idx, kind="stable")
    print(f"  argsort done [{time.time()-t1:.0f}s]", flush=True)

    # Apply permutation and write sorted files
    print("  writing sorted files...", flush=True)
    t2 = time.time()

    sorted_row_idx_path = os.path.join(tmpdir, "row_idx_sorted.bin")
    sorted_cols_path = os.path.join(tmpdir, "cols_sorted.bin")
    sorted_vals_path = os.path.join(tmpdir, "vals_sorted.bin")

    # Process in blocks to limit peak memory
    block = 100_000_000  # 100M entries per block
    for path, arr, dtype in [
        (sorted_row_idx_path, row_idx, np.int32),
        (sorted_cols_path, cols, np.int32),
        (sorted_vals_path, vals, np.complex64),
    ]:
        with open(path, "wb") as f:
            for start in range(0, n_nnz, block):
                end = min(start + block, n_nnz)
                chunk = arr[perm[start:end]]
                f.write(chunk.tobytes())
                del chunk
        print(f"    wrote {path} [{time.time()-t2:.0f}s]", flush=True)

    del row_idx, cols, vals, perm
    gc.collect()

    print(f"  Total sort time: {time.time()-t0:.0f}s", flush=True)
    print(f"  Use sorted=True in MatVecHandle for 15x faster matvec.", flush=True)


# =====================================================================
# 5. NormalOperator: B = A^* A as a scipy LinearOperator
# =====================================================================

class NormalOperator(LinearOperator):
    """
    B = A^* A, shape (n_cols, n_cols).

    B @ v = A^* @ (A @ v)

    Each matvec requires one C matvec (A @ v) and one C rmatvec (A^* @ w).
    The intermediate result w = A @ v uses the preallocated 25 GB buffer.
    The final result (n_cols, 150 MB) is freshly allocated.

    This is the operator to feed into eigsh(B, k, which='SM') to find
    the smallest eigenvalues of A^* A = singular values squared.
    """

    def __init__(self, handle):
        super().__init__(dtype=np.dtype('complex128'),
                         shape=(handle.n_cols, handle.n_cols))
        self.handle = handle

    def _matvec(self, v):
        # v: complex128 [n_cols]
        # w = A @ v  ->  [n_rows]  (uses preallocated buffer)
        w = self.handle.matvec(v)
        # result = A^* @ w  ->  [n_cols]  (freshly allocated)
        result = self.handle.rmatvec(w)
        return result

    def _rmatvec(self, v):
        # B is Hermitian: B^* = B
        return self._matvec(v)


# =====================================================================
# 6. MatVecOperator: A as a scipy LinearOperator (for svds)
# =====================================================================

class MatVecOperator(LinearOperator):
    """
    A as a scipy LinearOperator, shape (n_rows, n_cols).

    For use with scipy.sparse.linalg.svds(A, k, which='SM').

    NOTE: _matvec returns the internal 25 GB buffer. The caller must
    not hold references across calls. For the normal operator approach
    (eigsh on A^* A), use NormalOperator instead, which avoids this issue.
    """

    def __init__(self, handle):
        super().__init__(dtype=np.dtype('complex128'),
                         shape=(handle.n_rows, handle.n_cols))
        self.handle = handle

    def _matvec(self, v):
        return self.handle.matvec(v)

    def _rmatvec(self, v):
        return self.handle.rmatvec(v)


# =====================================================================
# 7. compute_rank: eigensolver strategy
# =====================================================================

def compute_rank(handle, k=20, tol=1e-6, maxiter=300, ncv=50,
                 method="eigsh", verbose=True):
    """
    Compute the rank of A by finding the nullity (number of zero eigenvalues
    of A^* A).

    Strategy:
    1. Build NormalOperator B = A^* A (LinearOperator, n_cols x n_cols)
    2. Find the k smallest eigenvalues of B using eigsh(which='SM')
    3. Count eigenvalues below a threshold (determined by spectral gap)
    4. nullity = count, rank = n_cols - nullity

    Parameters
    ----------
    handle : MatVecHandle
        Loaded data handle.
    k : int
        Number of smallest eigenvalues to find (default 20, enough to
        distinguish nullity 3 from 4).
    tol : float
        Relative tolerance for eigsh (default 1e-6).
    maxiter : int
        Maximum ARPACK iterations (default 300). Each iteration = 1 matvec
        + 1 rmatvec.
    ncv : int
        Number of Lanczos vectors (default 50, should be > 2k).
    method : str
        "eigsh" for eigenvalues of A^* A, or "svds" for singular values of A.

    Returns
    -------
    rank : int
    nullity : int
    eigenvalues : np.ndarray
        The k smallest eigenvalues of A^* A (or singular values squared).
    """
    n_cols = handle.n_cols
    n_rows = handle.n_rows

    if verbose:
        print(f"\n=== Rank computation ===", flush=True)
        print(f"  n_rows = {n_rows}", flush=True)
        print(f"  n_cols = {n_cols}", flush=True)
        print(f"  n_nnz  = {handle.n_nnz}", flush=True)
        print(f"  method = {method}, k = {k}, tol = {tol}, maxiter = {maxiter}",
              flush=True)

    k_eff = min(k, n_cols - 2)
    ncv_eff = min(ncv, n_cols - 1)

    if method == "eigsh":
        # Find smallest eigenvalues of A^* A
        B = NormalOperator(handle)

        if verbose:
            print(f"\n  Computing {k_eff} smallest eigenvalues of A^* A...",
                  flush=True)
        t0 = time.time()

        eigenvalues = eigsh(
            B, k=k_eff, which="SM", tol=tol, maxiter=maxiter,
            ncv=ncv_eff, return_eigenvectors=False,
        )
        eigenvalues = np.sort(np.real(eigenvalues))  # eigenvalues are real (HPD)

        if verbose:
            print(f"  eigenvalues: {eigenvalues}", flush=True)
            print(f"  [{time.time()-t0:.0f}s]", flush=True)

    elif method == "svds":
        # Find smallest singular values of A directly
        A_op = MatVecOperator(handle)

        if verbose:
            print(f"\n  Computing {k_eff} smallest singular values of A...",
                  flush=True)
        t0 = time.time()

        s_vals = svds(
            A_op, k=k_eff, which="SM", tol=tol, maxiter=maxiter,
            ncv=ncv_eff, return_singular_vectors=False,
        )
        s_vals = np.sort(np.abs(s_vals))
        eigenvalues = s_vals ** 2  # singular values squared

        if verbose:
            print(f"  singular values: {s_vals}", flush=True)
            print(f"  [{time.time()-t0:.0f}s]", flush=True)

    else:
        raise ValueError(f"Unknown method: {method}")

    # ---- Determine nullity from the spectral gap ----
    #
    # The zero eigenvalues (from the null space) will be very small but
    # nonzero due to floating-point arithmetic. We look for a gap in the
    # eigenvalue spectrum: if eigenvalue[i+1] / eigenvalue[i] > 1000,
    # then eigenvalues[0..i] are the "zero" ones.
    #
    # If no clear gap is found, use an absolute threshold relative to
    # the largest eigenvalue.

    nullity = _determine_nullity(eigenvalues, verbose=verbose)
    rank = n_cols - nullity

    if verbose:
        print(f"\n=== Result ===", flush=True)
        print(f"  rank    = {rank}", flush=True)
        print(f"  nullity = {nullity}", flush=True)
        if nullity == 3:
            print(f"  => dim HH^2 = 9", flush=True)
        elif nullity == 4:
            print(f"  => dim HH^2 = 8", flush=True)
        else:
            print(f"  => Unexpected nullity! Check eigenvalues.", flush=True)

    return rank, nullity, eigenvalues


def _determine_nullity(eigenvalues, threshold_ratio=1000.0, verbose=True):
    """
    Determine the nullity from the eigenvalue spectrum.

    Strategy:
    1. Look for a spectral gap (ratio > threshold_ratio between consecutive
       eigenvalues).
    2. If no gap found, use an absolute threshold: eigenvalues below
       max_eigenvalue * 1e-8 are considered zero.
    """
    n = len(eigenvalues)
    if n == 0:
        return 0

    # All eigenvalues should be non-negative (A^* A is PSD)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Strategy 1: look for a spectral gap
    for i in range(n - 1):
        if eigenvalues[i] < 1e-30:
            continue
        ratio = eigenvalues[i + 1] / max(eigenvalues[i], 1e-30)
        if ratio > threshold_ratio:
            if verbose:
                print(f"  Spectral gap found: eigenvalue[{i}]={eigenvalues[i]:.2e} "
                      f"-> eigenvalue[{i+1}]={eigenvalues[i+1]:.2e} "
                      f"(ratio={ratio:.0f})", flush=True)
            return i + 1

    # Strategy 2: absolute threshold
    max_ev = eigenvalues[-1]
    if max_ev < 1e-30:
        # All eigenvalues are essentially zero
        if verbose:
            print(f"  All eigenvalues near zero (max={max_ev:.2e})", flush=True)
        return n

    abs_threshold = max_ev * 1e-8
    nullity = int(np.sum(eigenvalues < abs_threshold))

    if verbose:
        print(f"  No spectral gap found. Using absolute threshold "
              f"{abs_threshold:.2e} (max_ev={max_ev:.2e})", flush=True)
        print(f"  Eigenvalues below threshold: {nullity}", flush=True)

    return nullity


# =====================================================================
# 8. Main: CLI interface
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Sparse matvec C extension")
    sub = parser.add_subparsers(dest="command")

    # precompute
    p_pre = sub.add_parser("precompute", help="Convert chunk files to flat arrays")
    p_pre.add_argument("--tmpdir", required=True)
    p_pre.add_argument("--prefix", required=True)
    p_pre.add_argument("--n-chunks", type=int, required=True)
    p_pre.add_argument("--unique-keys", required=True)
    p_pre.add_argument("--n-workers", type=int, default=10)

    # sort
    p_sort = sub.add_parser("sort", help="Sort flat arrays by row_idx")
    p_sort.add_argument("--tmpdir", required=True)
    p_sort.add_argument("--n-nnz", type=int, required=True)

    # rank
    p_rank = sub.add_parser("rank", help="Compute rank")
    p_rank.add_argument("--tmpdir", required=True)
    p_rank.add_argument("--n-rows", type=int, required=True)
    p_rank.add_argument("--n-cols", type=int, required=True)
    p_rank.add_argument("--sorted", action="store_true",
                        help="Use sorted data files")
    p_rank.add_argument("--prefault", action="store_true",
                        help="Prefault mmap pages")
    p_rank.add_argument("--k", type=int, default=20)
    p_rank.add_argument("--tol", type=float, default=1e-6)
    p_rank.add_argument("--maxiter", type=int, default=300)
    p_rank.add_argument("--ncv", type=int, default=50)
    p_rank.add_argument("--method", choices=["eigsh", "svds"], default="eigsh")

    args = parser.parse_args()

    if args.command == "precompute":
        precompute(args.tmpdir, args.prefix, args.n_chunks,
                   args.unique_keys, args.n_workers)

    elif args.command == "sort":
        sort_by_row_idx(args.tmpdir, args.n_nnz)

    elif args.command == "rank":
        # Read nnz count
        with open(os.path.join(args.tmpdir, "nnz_count.txt")) as f:
            n_nnz = int(f.read().strip())

        if args.sorted:
            row_idx_path = os.path.join(args.tmpdir, "row_idx_sorted.bin")
            cols_path = os.path.join(args.tmpdir, "cols_sorted.bin")
            vals_path = os.path.join(args.tmpdir, "vals_sorted.bin")
        else:
            row_idx_path = os.path.join(args.tmpdir, "row_idx_flat.bin")
            cols_path = os.path.join(args.tmpdir, "cols_flat.bin")
            vals_path = os.path.join(args.tmpdir, "vals_flat.bin")

        handle = MatVecHandle(
            row_idx_path, cols_path, vals_path,
            n_nnz, args.n_rows, args.n_cols,
            sorted=args.sorted, prefault=args.prefault,
        )

        try:
            rank, nullity, eigenvalues = compute_rank(
                handle, k=args.k, tol=args.tol,
                maxiter=args.maxiter, ncv=args.ncv,
                method=args.method,
            )
        finally:
            handle.close()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
