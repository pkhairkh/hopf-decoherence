#!/usr/bin/env python3
"""Export the degree-1 bialgebra coboundary ∂_b¹ for B⁺(u_q(sl₃)) at ℓ=3
as flat binary files.

Outputs (in OUTPUT_DIR = /tmp/h2b_sl3_box2ztl4/):
    d1_row_idx.bin  — int32[nnz]      — row indices into C²
    d1_cols.bin     — int32[nnz]      — column indices into C¹
    d1_vals.bin     — complex64[nnz]  — matrix values (8 bytes each)

The differential is

    ∂_b¹ : C¹ → C²
    ∂_b¹(h) = (∂^h h, -∂^c h)

with
    C¹ = Hom(B̄, B̄),                       dim = 242² = 58,564
    C² = Hom(B̄⊗B̄, B̄) ⊕ Hom(B̄, B̄⊗B̄),   dim = 2 · 242³ = 28,344,976

The Mastnak–Witherspoon formulas (with a, b, c ∈ B̄ = ker ε):

    (∂^h h)(a, b)  = a·h(b) - h(a·b) + h(a)·b
    (∂^c h)(c)     = c₁⊗h(c₂) - Δ(h(c)) + h(c₁)⊗c₂

Layout
------
Both C¹ and C² are block-decomposed by the 2D K-weight shift
s ∈ (Z/3)² into 9 blocks (only 3 of which are non-empty for sl₃ at ℓ=3,
because the Cartan matrix is degenerate mod 3 and the image of the K-weight
map B̄ → (Z/3)² is the diagonal {(0,0), (1,1), (2,2)}).

Shifts are enumerated in lex order
    s = (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)

Per shift s:
  • C¹ columns: (j, k) with wt(j) - wt(k) ≡ s   (j outer, k inner)
  • C² f-rows:  (a, b, t) with wt(t) - wt(a) - wt(b) ≡ s   [Hochschild output]
  • C² g-rows:  (c, α, β) with wt(α) + wt(β) - wt(c) ≡ s   [coHochschild output]

Global flat indexing:
    global_col(s, local_col)      = col_offset_s + local_col
    global_row(s, f_local_row)    = row_offset_s + f_local_row           (f-rows)
    global_row(s, g_local_row)    = row_offset_s + n_f_rows_s + g_local_row   (g-rows)

  where  col_offset_s = Σ_{s' < s}  n_cols_{s'}
         row_offset_s = Σ_{s' < s} (n_f_rows_{s'} + n_g_rows_{s'})

The algebra (PBW basis, multiplication table, comultiplication table, weight
function, and B̄ structure constants) is imported directly from
`compute_h1b_bplus_sl3.py` so the export is bit-for-bit consistent with the
existing H̃¹_b computation.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np

# ------------------------------------------------------------------
# Import the algebra from compute_h1b_bplus_sl3.py.
# The repo lives at /root/hopf-decoherence/scripts/ on the remote sandbox
# and at /home/z/my-project/hopf-decoherence/scripts/ in the local mirror.
# ------------------------------------------------------------------
_REPO_CANDIDATES = (
    "/root/hopf-decoherence/scripts",
    "/home/z/my-project/hopf-decoherence/scripts",
)


def _load_h1b_module():
    for repo_dir in _REPO_CANDIDATES:
        path = os.path.join(repo_dir, "compute_h1b_bplus_sl3.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(
                "compute_h1b_bplus_sl3", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["compute_h1b_bplus_sl3"] = mod
            spec.loader.exec_module(mod)
            return mod, repo_dir
    raise FileNotFoundError(
        "compute_h1b_bplus_sl3.py not found in any of: "
        + ", ".join(_REPO_CANDIDATES))


h1b, REPO_DIR = _load_h1b_module()

ELL = h1b.ELL          # 3
DIM = h1b.DIM          # 243
DIM_B_BAR = DIM - 1    # 242

OUTPUT_DIR = "/tmp/h2b_sl3_box2ztl4"

# Expected dimensions (the actual ones; the problem statement's "9,448,324"
# turns out to be the *per-shift* dim of C² for shift (0,0), not the total).
EXPECTED_DIM_C1 = DIM_B_BAR ** 2          # 58,564
EXPECTED_DIM_C2 = 2 * DIM_B_BAR ** 3      # 28,344,976


# ------------------------------------------------------------------
# Algebra setup: build mult/delta tables and inverse lookups.
# Mirrors compute_h1b_bplus_sl3.compute() up to (but not including) the
# per-shift rank computation.
# ------------------------------------------------------------------

def setup_algebra():
    """Build the multiplication/comultiplication tables and inverse lookups."""
    print(f"Building multiplication table (dim={DIM})...")
    t0 = time.time()
    ms = h1b.build_mult(DIM)
    h1b.ms_global = ms
    print(f"  Mult table: {time.time()-t0:.1f}s")

    print("Building comultiplication table...")
    t0 = time.time()
    Delta = h1b.build_delta(DIM)
    print(f"  Delta table: {time.time()-t0:.1f}s")

    print("Verifying Δ(E12) = Δ(E1)Δ(E2) - q Δ(E2)Δ(E1)...")
    e12_err = h1b.verify_delta_E12(Delta, DIM)
    print(f"  Δ(E12) check max error: {e12_err:.2e}")
    assert e12_err < 1e-10, "Δ(E12) formula inconsistent!"

    epsilon = h1b.build_epsilon(DIM)
    print("Verifying counitality...")
    counital_err = h1b.verify_counital(Delta, DIM, epsilon)
    print(f"  Counital max error: {counital_err:.2e}")
    assert counital_err < 1e-10, "Counitality failed!"

    print("Building B̄ basis structure constants...")
    t0 = time.time()
    mult_bar = h1b.build_mult_bar(ms, epsilon, DIM)
    delta_bar = h1b.build_delta_bar(Delta, DIM)
    print(f"  mult_bar nnz: {len(mult_bar)}")
    print(f"  delta_bar nnz: {len(delta_bar)}")
    print(f"  B̄ basis: {time.time()-t0:.1f}s")

    B_bar = list(range(1, DIM))
    weights = [h1b.weight(i) for i in range(DIM)]

    # ----- Inverse lookups (same naming as compute_h1b_bplus_sl3.py) -----
    # mult_bar[(l, a, b)] = v   means   basis[a] · basis[b] = Σ v · basis[l]   in B̄.
    #
    #   ms_by_b[j]            = (a_arr, t_arr, v_arr)   with mult_bar[t, a, j] = v
    #   ms_by_a[j]            = (b_arr, t_arr, v_arr)   with mult_bar[t, j, b] = v
    #   inv_mult_bar[k]       = (a_arr, b_arr, v_arr)   with mult_bar[k, a, b] = v
    #
    # delta_bar[(c, j, k)] = v  means   Δ(basis[c]) = Σ v · basis[j] ⊗ basis[k]   in B̄⊗B̄.
    # j is the LEFT tensor factor, k is the RIGHT tensor factor.
    #
    #   inv_delta_left_by_k[k]  = (c_arr, al_arr, v_arr)  with delta_bar[c, al, k] = v   (k = RIGHT)
    #   inv_delta_right_by_k[k] = (c_arr, be_arr, v_arr)  with delta_bar[c, k, be] = v   (k = LEFT)
    #   delta_by_c[c]           = (al_arr, be_arr, v_arr) with delta_bar[c, al, be] = v
    print("Precomputing inverse tables...")
    t0 = time.time()
    inv_mult_bar = {l: [] for l in B_bar}
    ms_by_b = {b: [] for b in B_bar}
    ms_by_a = {a: [] for a in B_bar}
    for (l, a, b), v in mult_bar.items():
        inv_mult_bar[l].append((a, b, v))
        ms_by_b[b].append((a, l, v))
        ms_by_a[a].append((b, l, v))

    inv_delta_left_by_k = {k: [] for k in B_bar}
    inv_delta_right_by_k = {k: [] for k in B_bar}
    delta_by_c = {c: [] for c in B_bar}
    for (c, j, k), v in delta_bar.items():
        # delta_bar[c, j, k] = v : j is LEFT, k is RIGHT.
        inv_delta_left_by_k[k].append((c, j, v))   # fix RIGHT=k
        inv_delta_right_by_k[j].append((c, k, v))  # fix LEFT=j  (keyed by LEFT factor)
        delta_by_c[c].append((j, k, v))

    def to_arr3(lst):
        if not lst:
            return (np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                    np.array([], dtype=complex))
        return (np.array([x[0] for x in lst], dtype=np.int32),
                np.array([x[1] for x in lst], dtype=np.int32),
                np.array([x[2] for x in lst], dtype=complex))

    ms_by_b_arr = {b: to_arr3(ms_by_b[b]) for b in B_bar}
    ms_by_a_arr = {a: to_arr3(ms_by_a[a]) for a in B_bar}
    inv_mult_bar_arr = {l: to_arr3(inv_mult_bar[l]) for l in B_bar}
    inv_delta_left_by_k_arr = {k: to_arr3(inv_delta_left_by_k[k]) for k in B_bar}
    inv_delta_right_by_k_arr = {k: to_arr3(inv_delta_right_by_k[k]) for k in B_bar}
    delta_by_c_arr = {c: to_arr3(delta_by_c[c]) for c in B_bar}
    print(f"  Inverse tables: {time.time()-t0:.1f}s")

    return (B_bar, weights,
            ms_by_b_arr, ms_by_a_arr, inv_mult_bar_arr,
            inv_delta_left_by_k_arr, inv_delta_right_by_k_arr, delta_by_c_arr)


# ------------------------------------------------------------------
# Per-shift enumeration
# ------------------------------------------------------------------

def enum_shift(s, B_bar, weights):
    """Enumerate (cols, f_rows, g_rows) for one weight shift s ∈ (Z/3)².

    cols   = [(j, k)   for j, k ∈ B̄  with wt(j) - wt(k) ≡ s]
    f_rows = [(a, b, t) for a, b, t ∈ B̄  with wt(t) - wt(a) - wt(b) ≡ s]
    g_rows = [(c, α, β) for c, α, β ∈ B̄  with wt(α) + wt(β) - wt(c) ≡ s]
    """
    cols_list = [(j, k) for j in B_bar for k in B_bar
                 if ((weights[j][0] - weights[k][0]) % ELL,
                     (weights[j][1] - weights[k][1]) % ELL) == s]
    f_rows_list = [(a, b, t) for a in B_bar for b in B_bar for t in B_bar
                   if ((weights[t][0] - weights[a][0] - weights[b][0]) % ELL,
                       (weights[t][1] - weights[a][1] - weights[b][1]) % ELL) == s]
    g_rows_list = [(c, al, be) for c in B_bar for al in B_bar for be in B_bar
                   if ((weights[al][0] + weights[be][0] - weights[c][0]) % ELL,
                       (weights[al][1] + weights[be][1] - weights[c][1]) % ELL) == s]
    return cols_list, f_rows_list, g_rows_list


# ------------------------------------------------------------------
# Per-shift matrix entries (vectorized, same logic as
# compute_h1b_bplus_sl3.compute_shift_block but WITHOUT the Gram/eigsh step)
# ------------------------------------------------------------------

def build_shift_entries(s, dim,
                        ms_by_b_arr, ms_by_a_arr, inv_mult_bar_arr,
                        inv_delta_left_by_k_arr, inv_delta_right_by_k_arr,
                        delta_by_c_arr,
                        col_offset, row_offset,
                        cols_list, f_rows_list, g_rows_list):
    """Build (row, col, val) entries for one shift with GLOBAL flat indices.

    Returns (arrays_tuple, n_entries, n_f_rows, n_g_rows, elapsed_s)
        arrays_tuple = (rows_arr, cols_arr, vals_arr) as int32, int32, complex64
        or None if no entries were produced.
    """
    t_start = time.time()

    n_cols = len(cols_list)
    n_f_rows = len(f_rows_list)
    n_g_rows = len(g_rows_list)

    if n_cols == 0:
        return None, 0, 0, 0, time.time() - t_start

    # Local row maps (243³ = 14.3M entries each as int32 = 57 MB).
    f_row_map = np.full((dim, dim, dim), -1, dtype=np.int64)
    for i, (a, b, t) in enumerate(f_rows_list):
        f_row_map[a, b, t] = i
    g_row_map = np.full((dim, dim, dim), -1, dtype=np.int64)
    for i, (c, al, be) in enumerate(g_rows_list):
        g_row_map[c, al, be] = i

    t_map = time.time() - t_start

    rows_chunks = []
    cols_chunks = []
    vals_chunks = []
    n_entries = 0

    t_build_start = time.time()
    for col_idx, (j, k) in enumerate(cols_list):
        global_col = col_offset + col_idx

        # ----- ∂^h term 1: row (a, k, t), val = +mult_bar[t, a, j] -----
        # ms_by_b_arr[j] = (a_arr, t_arr, v_arr) with mult_bar[t, a, j] = v.
        a_arr, t_arr, v_arr = ms_by_b_arr[j]
        if a_arr.size > 0:
            row_arr = f_row_map[a_arr, k, t_arr]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append(
                    (row_arr[valid] + row_offset).astype(np.int32))
                cols_chunks.append(
                    np.full(int(np.sum(valid)), global_col, dtype=np.int32))
                vals_chunks.append(v_arr[valid].astype(np.complex64))
                n_entries += int(np.sum(valid))

        # ----- ∂^h term 2: row (a, b, j), val = -mult_bar[k, a, b] -----
        # inv_mult_bar_arr[k] = (a_arr, b_arr, v_arr) with mult_bar[k, a, b] = v.
        a_arr, b_arr, v_arr = inv_mult_bar_arr[k]
        if a_arr.size > 0:
            row_arr = f_row_map[a_arr, b_arr, j]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append(
                    (row_arr[valid] + row_offset).astype(np.int32))
                cols_chunks.append(
                    np.full(int(np.sum(valid)), global_col, dtype=np.int32))
                vals_chunks.append((-v_arr[valid]).astype(np.complex64))
                n_entries += int(np.sum(valid))

        # ----- ∂^h term 3: row (k, b, t), val = +mult_bar[t, j, b] -----
        # ms_by_a_arr[j] = (b_arr, t_arr, v_arr) with mult_bar[t, j, b] = v.
        b_arr, t_arr, v_arr = ms_by_a_arr[j]
        if b_arr.size > 0:
            row_arr = f_row_map[k, b_arr, t_arr]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append(
                    (row_arr[valid] + row_offset).astype(np.int32))
                cols_chunks.append(
                    np.full(int(np.sum(valid)), global_col, dtype=np.int32))
                vals_chunks.append(v_arr[valid].astype(np.complex64))
                n_entries += int(np.sum(valid))

        # ----- -∂^c term 1: row (c, al, j), val = -delta_bar[c, al, k] -----
        # inv_delta_left_by_k_arr[k] = (c_arr, al_arr, v_arr) with delta_bar[c, al, k] = v.
        c_arr, al_arr, v_arr = inv_delta_left_by_k_arr[k]
        if c_arr.size > 0:
            row_arr = g_row_map[c_arr, al_arr, j]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append(
                    (row_arr[valid] + row_offset + n_f_rows).astype(np.int32))
                cols_chunks.append(
                    np.full(int(np.sum(valid)), global_col, dtype=np.int32))
                vals_chunks.append((-v_arr[valid]).astype(np.complex64))
                n_entries += int(np.sum(valid))

        # ----- -∂^c term 2: row (k, al, be), val = +delta_bar[j, al, be] -----
        # delta_by_c_arr[j] = (al_arr, be_arr, v_arr) with delta_bar[j, al, be] = v.
        al_arr, be_arr, v_arr = delta_by_c_arr[j]
        if al_arr.size > 0:
            row_arr = g_row_map[k, al_arr, be_arr]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append(
                    (row_arr[valid] + row_offset + n_f_rows).astype(np.int32))
                cols_chunks.append(
                    np.full(int(np.sum(valid)), global_col, dtype=np.int32))
                vals_chunks.append(v_arr[valid].astype(np.complex64))
                n_entries += int(np.sum(valid))

        # ----- -∂^c term 3: row (c, j, be), val = -delta_bar[c, k, be] -----
        # inv_delta_right_by_k_arr[k] = (c_arr, be_arr, v_arr) with delta_bar[c, k, be] = v.
        c_arr, be_arr, v_arr = inv_delta_right_by_k_arr[k]
        if c_arr.size > 0:
            row_arr = g_row_map[c_arr, j, be_arr]
            valid = row_arr >= 0
            if np.any(valid):
                rows_chunks.append(
                    (row_arr[valid] + row_offset + n_f_rows).astype(np.int32))
                cols_chunks.append(
                    np.full(int(np.sum(valid)), global_col, dtype=np.int32))
                vals_chunks.append((-v_arr[valid]).astype(np.complex64))
                n_entries += int(np.sum(valid))

    t_build = time.time() - t_build_start

    if not rows_chunks:
        return None, 0, n_f_rows, n_g_rows, t_map + t_build

    rows_arr = np.concatenate(rows_chunks)
    cols_arr = np.concatenate(cols_chunks)
    vals_arr = np.concatenate(vals_chunks)
    return (rows_arr, cols_arr, vals_arr), n_entries, n_f_rows, n_g_rows, t_map + t_build


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print(f"=== Export ∂_b¹ for B⁺(u_q(sl₃)) at ℓ={ELL} ===")
    print(f"Repo dir:        {REPO_DIR}")
    print(f"Output dir:      {OUTPUT_DIR}")
    print(f"dim B = {DIM}, dim B̄ = {DIM_B_BAR}")
    print(f"Expected dim C¹ = {EXPECTED_DIM_C1}  (= {DIM_B_BAR}²)")
    print(f"Expected dim C² = {EXPECTED_DIM_C2}  (= 2·{DIM_B_BAR}³)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    setup = setup_algebra()
    (B_bar, weights,
     ms_by_b_arr, ms_by_a_arr, inv_mult_bar_arr,
     inv_delta_left_by_k_arr, inv_delta_right_by_k_arr, delta_by_c_arr) = setup
    dim = DIM

    all_shifts = [(s1, s2) for s1 in range(ELL) for s2 in range(ELL)]

    # ----- Phase 1: enumerate per-shift sizes to determine offsets. -----
    print("\nPhase 1: enumerating shift sizes...")
    t0 = time.time()
    shift_info = []
    col_offset = 0
    row_offset = 0
    for s in all_shifts:
        cols_list, f_rows_list, g_rows_list = enum_shift(s, B_bar, weights)
        n_cols = len(cols_list)
        n_f = len(f_rows_list)
        n_g = len(g_rows_list)
        shift_info.append({
            's': s,
            'n_cols': n_cols, 'n_f_rows': n_f, 'n_g_rows': n_g,
            'col_offset': col_offset, 'row_offset': row_offset,
            'cols_list': cols_list,
            'f_rows_list': f_rows_list, 'g_rows_list': g_rows_list,
        })
        print(f"  shift {s}: cols={n_cols:7d}, f_rows={n_f:8d}, "
              f"g_rows={n_g:8d}, col_offset={col_offset:7d}, "
              f"row_offset={row_offset:9d}")
        col_offset += n_cols
        row_offset += n_f + n_g
    total_cols = col_offset
    total_rows = row_offset
    print(f"\n  Total cols (dim C¹): {total_cols}  "
          f"(expected {EXPECTED_DIM_C1}, match: {total_cols == EXPECTED_DIM_C1})")
    print(f"  Total rows (dim C²): {total_rows}  "
          f"(expected {EXPECTED_DIM_C2}, match: {total_rows == EXPECTED_DIM_C2})")
    print(f"  Enum time: {time.time()-t0:.1f}s")

    assert total_cols == EXPECTED_DIM_C1, \
        f"dim C¹ mismatch: got {total_cols}, expected {EXPECTED_DIM_C1}"
    assert total_rows == EXPECTED_DIM_C2, \
        f"dim C² mismatch: got {total_rows}, expected {EXPECTED_DIM_C2}"

    # ----- Phase 2: build matrix entries per shift. -----
    print("\nPhase 2: building matrix entries per shift...")
    all_rows = []
    all_cols = []
    all_vals = []
    total_nnz = 0
    t_total = time.time()
    for info in shift_info:
        s = info['s']
        if info['n_cols'] == 0:
            print(f"  shift {s}: empty (skipped)")
            continue
        ts = time.time()
        result, n_entries, n_f, n_g, t_build = build_shift_entries(
            s, dim,
            ms_by_b_arr, ms_by_a_arr, inv_mult_bar_arr,
            inv_delta_left_by_k_arr, inv_delta_right_by_k_arr, delta_by_c_arr,
            info['col_offset'], info['row_offset'],
            info['cols_list'], info['f_rows_list'], info['g_rows_list'])
        if result is None:
            print(f"  shift {s}: cols={info['n_cols']:7d}, "
                  f"f_rows={n_f:8d}, g_rows={n_g:8d}, nnz=0  "
                  f"[build={t_build:.1f}s]")
            continue
        rows_arr, cols_arr, vals_arr = result
        all_rows.append(rows_arr)
        all_cols.append(cols_arr)
        all_vals.append(vals_arr)
        total_nnz += len(rows_arr)
        print(f"  shift {s}: cols={info['n_cols']:7d}, "
              f"f_rows={n_f:8d}, g_rows={n_g:8d}, "
              f"nnz={len(rows_arr):8d}  "
              f"[build={time.time()-ts:.1f}s]")

    print(f"\n  Total nnz:        {total_nnz}")
    print(f"  Total build time: {time.time()-t_total:.1f}s")

    # ----- Phase 3: concatenate all entries. -----
    print("\nPhase 3: concatenating entries...")
    rows = np.concatenate(all_rows) if all_rows else \
        np.array([], dtype=np.int32)
    cols = np.concatenate(all_cols) if all_cols else \
        np.array([], dtype=np.int32)
    vals = np.concatenate(all_vals) if all_vals else \
        np.array([], dtype=np.complex64)
    print(f"  rows: dtype={rows.dtype}, shape={rows.shape}, "
          f"min={rows.min()}, max={rows.max()}")
    print(f"  cols: dtype={cols.dtype}, shape={cols.shape}, "
          f"min={cols.min()}, max={cols.max()}")
    print(f"  vals: dtype={vals.dtype}, shape={vals.shape}, "
          f"|max|={np.abs(vals).max() if len(vals) else 0:.4g}")

    # Sanity checks
    assert len(rows) == len(cols) == len(vals), \
        f"length mismatch: {len(rows)} vs {len(cols)} vs {len(vals)}"
    assert len(rows) == total_nnz, \
        f"nnz mismatch: got {len(rows)}, expected {total_nnz}"
    if total_nnz > 0:
        assert rows.min() >= 0, f"row index negative: {rows.min()}"
        assert rows.max() < total_rows, \
            f"row index out of bounds: {rows.max()} >= {total_rows}"
        assert cols.min() >= 0, f"col index negative: {cols.min()}"
        assert cols.max() < total_cols, \
            f"col index out of bounds: {cols.max()} >= {total_cols}"

    # ----- Phase 4: write binary files. -----
    print(f"\nPhase 4: writing binary files to {OUTPUT_DIR}/")
    rows_path = os.path.join(OUTPUT_DIR, "d1_row_idx.bin")
    cols_path = os.path.join(OUTPUT_DIR, "d1_cols.bin")
    vals_path = os.path.join(OUTPUT_DIR, "d1_vals.bin")
    rows.tofile(rows_path)
    cols.tofile(cols_path)
    vals.tofile(vals_path)

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("=== Summary ===")
    print("=" * 60)
    print(f"  dim C¹ (cols):       {total_cols}")
    print(f"  dim C² (rows):       {total_rows}")
    print(f"  nnz:                 {total_nnz}")
    print(f"  Files written:")
    print(f"    d1_row_idx.bin:    {rows.nbytes / 1e6:8.2f} MB  "
          f"({rows.dtype}, shape {rows.shape})")
    print(f"    d1_cols.bin:       {cols.nbytes / 1e6:8.2f} MB  "
          f"({cols.dtype}, shape {cols.shape})")
    print(f"    d1_vals.bin:       {vals.nbytes / 1e6:8.2f} MB  "
          f"({vals.dtype}, shape {vals.shape})")
    print(f"  Total time:          {time.time()-t_total:.1f}s")

    # Per-shift breakdown
    print("\n  Per-shift breakdown:")
    print(f"    {'shift':<8} {'cols':>8} {'f_rows':>9} {'g_rows':>9} "
          f"{'col_off':>8} {'row_off':>10}")
    for info in shift_info:
        s = info['s']
        print(f"    {str(s):<8} {info['n_cols']:>8d} "
              f"{info['n_f_rows']:>9d} {info['n_g_rows']:>9d} "
              f"{info['col_offset']:>8d} {info['row_offset']:>10d}")


if __name__ == "__main__":
    main()
