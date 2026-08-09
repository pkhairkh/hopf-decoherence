#!/usr/bin/env python3
"""
Sort ∂_b² entries by row_idx using np.argsort.
Input: row_idx_flat.bin, cols_flat.bin, vals_flat.bin
Output: row_idx_sorted.bin, cols_sorted.bin, vals_sorted.bin

Memory: 32 GB (permutation) + mmap'd I/O
"""
import gc
import os
import resource
import sys
import time
import numpy as np

TMPDIR = "/tmp/h2b_sl3_box2ztl4"
NNZ = 4091264798


def main():
    t0 = time.time()
    print("=== Sort ∂_b² by row_idx ===", flush=True)
    print(f"  NNZ = {NNZ}", flush=True)

    # Load row_idx via mmap (read-only).
    print("\n[1] mmap row_idx_flat.bin...", flush=True)
    row_idx = np.memmap(f"{TMPDIR}/row_idx_flat.bin", dtype=np.int32, mode='r')
    print(f"  shape: {row_idx.shape}, dtype: {row_idx.dtype}", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"  RSS: {rss:.1f} GB", flush=True)

    # Compute argsort.
    print("\n[2] Computing argsort (np.argsort, stable)...", flush=True)
    t1 = time.time()
    perm = np.argsort(row_idx, kind='stable')
    print(f"  done [{time.time()-t1:.0f}s]", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"  RSS: {rss:.1f} GB (permutation: {perm.nbytes/1e9:.1f} GB)", flush=True)

    # Apply permutation and write sorted files.
    print("\n[3] Applying permutation and writing sorted files...", flush=True)
    t2 = time.time()

    # row_idx_sorted
    print("  row_idx_sorted...", flush=True)
    row_idx_sorted = np.memmap(f"{TMPDIR}/row_idx_sorted.bin", dtype=np.int32,
                                mode='w+', shape=row_idx.shape)
    # Process in chunks to avoid memory blowup.
    chunk = 100_000_000  # 100M entries = 400 MB
    for off in range(0, NNZ, chunk):
        n = min(chunk, NNZ - off)
        p = perm[off:off+n]
        row_idx_sorted[off:off+n] = row_idx[p]
        del p
        if (off // chunk) % 5 == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(f"    {off//1e6:.0f}M/{NNZ//1e6:.0f}M  [{time.time()-t2:.0f}s, RSS={rss:.1f} GB]",
                  flush=True)
    del row_idx_sorted
    gc.collect()

    # cols_sorted
    print("  cols_sorted...", flush=True)
    cols = np.memmap(f"{TMPDIR}/cols_flat.bin", dtype=np.int32, mode='r')
    cols_sorted = np.memmap(f"{TMPDIR}/cols_sorted.bin", dtype=np.int32,
                             mode='w+', shape=cols.shape)
    for off in range(0, NNZ, chunk):
        n = min(chunk, NNZ - off)
        p = perm[off:off+n]
        cols_sorted[off:off+n] = cols[p]
        del p
        if (off // chunk) % 5 == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(f"    {off//1e6:.0f}M/{NNZ//1e6:.0f}M  [{time.time()-t2:.0f}s, RSS={rss:.1f} GB]",
                  flush=True)
    del cols, cols_sorted
    gc.collect()

    # vals_sorted
    print("  vals_sorted...", flush=True)
    vals = np.memmap(f"{TMPDIR}/vals_flat.bin", dtype=np.complex64, mode='r')
    vals_sorted = np.memmap(f"{TMPDIR}/vals_sorted.bin", dtype=np.complex64,
                             mode='w+', shape=vals.shape)
    for off in range(0, NNZ, chunk):
        n = min(chunk, NNZ - off)
        p = perm[off:off+n]
        vals_sorted[off:off+n] = vals[p]
        del p
        if (off // chunk) % 5 == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(f"    {off//1e6:.0f}M/{NNZ//1e6:.0f}M  [{time.time()-t2:.0f}s, RSS={rss:.1f} GB]",
                  flush=True)
    del vals, vals_sorted, perm
    gc.collect()

    print(f"  done [{time.time()-t2:.0f}s]", flush=True)

    # Verify.
    print("\n[4] Verifying sort...", flush=True)
    t3 = time.time()
    ridx = np.memmap(f"{TMPDIR}/row_idx_sorted.bin", dtype=np.int32, mode='r')
    # Check first and last entries.
    print(f"  first 10: {ridx[:10]}", flush=True)
    print(f"  last 10:  {ridx[-10:]}", flush=True)
    # Check monotonicity on samples.
    sample = np.linspace(0, NNZ - 2, 10000, dtype=np.int64)
    vals_sample = ridx[sample]
    diffs = np.diff(vals_sample)
    n_violations = np.sum(diffs < 0)
    print(f"  sample violations: {n_violations}/10000", flush=True)
    if n_violations == 0:
        print("  SORT OK", flush=True)
    else:
        print("  WARNING: sort violations detected!", flush=True)
    print(f"  done [{time.time()-t3:.1f}s]", flush=True)

    print(f"\n=== SORT COMPLETE === [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "15"
    os.environ["OPENBLAS_NUM_THREADS"] = "15"
    os.environ["MKL_NUM_THREADS"] = "15"
    main()
