#!/usr/bin/env python3
"""
Verify D_2(ell) = (ell^3 - ell)/6 by computing rank(Phi on St tensor St)
numerically for ell = 3, 5, 7.

This is the crucial check that the corrected formula is right.
Uses singular value gap analysis for rock-solid rank determination.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hopf_decoherence.q_algebra import build_weyl_module_sl2, verify_algebra_relations
from hopf_decoherence.coproduct import coproduct_matrices_sl2, compute_phi_rank
from hopf_decoherence.rank_deficiency import D_ell, expected_rank


def main():
    print("=" * 80)
    print("  VERIFICATION: D_2(ell) = (ell^3 - ell)/6 for u_q(sl_2)")
    print("=" * 80)

    all_pass = True

    for ell in [3, 5, 7]:
        j = ell - 1  # Highest weight j = ell-1 for Steinberg, dim = ell
        q_root = np.exp(2j * np.pi / ell)
        q_generic = np.exp(1.7j)

        print(f"\n--- ell = {ell}, j = {j}, dim(St) = {j+1} ---")

        # Verify algebra relations
        for q_label, q_val in [('root', q_root), ('generic', q_generic)]:
            K, E, F = build_weyl_module_sl2(j, q_val)
            ok, errs = verify_algebra_relations(K, E, F, q_val)
            status = "PASS" if ok else "FAIL"
            print(f"  Relations at q={q_label}: {status} (errs: {errs[0]:.2e}, {errs[1]:.2e}, {errs[2]:.2e})")
            if not ok:
                all_pass = False

        # Compute rank at root of unity
        rank_root, alg_dim, tens_dim, phi_root = compute_phi_rank(j, q_root, ell)
        print(f"  rank(Phi) at root of unity: {rank_root}")

        # Compute rank at generic q
        rank_generic, _, _, phi_generic = compute_phi_rank(j, q_generic, ell)
        print(f"  rank(Phi) at generic q:      {rank_generic}")

        # Singular value gap analysis
        S_root = np.linalg.svd(phi_root, compute_uv=False)
        S_generic = np.linalg.svd(phi_generic, compute_uv=False)

        # Sort descending
        S_root_sorted = np.sort(S_root)[::-1]
        S_generic_sorted = np.sort(S_generic)[::-1]

        # Find gap in root-of-unity singular values
        expected_r = expected_rank(ell)
        if expected_r < len(S_root_sorted):
            gap_below = S_root_sorted[expected_r - 1]
            gap_above = S_root_sorted[expected_r] if expected_r < len(S_root_sorted) else 0
            gap_ratio = gap_below / max(gap_above, 1e-300)
        else:
            gap_ratio = float('inf')

        print(f"  SV gap at rank boundary: {gap_ratio:.2e}")

        # Compute deficiency
        deficiency = rank_generic - rank_root
        expected_deficiency = D_ell(ell)
        expected_r_val = expected_rank(ell)

        def_match = deficiency == expected_deficiency
        rank_match = rank_root == expected_r_val

        print(f"\n  Computed deficiency:  {deficiency}")
        print(f"  Expected D(ell):      {expected_deficiency}")
        print(f"  Deficiency match:     {'PASS' if def_match else 'FAIL'}")

        print(f"  Computed rank(root):  {rank_root}")
        print(f"  Expected rank(root):  {expected_r_val}")
        print(f"  Rank match:           {'PASS' if rank_match else 'FAIL'}")

        if not def_match or not rank_match:
            all_pass = False

    print(f"\n{'='*80}")
    print(f"OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*80}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
