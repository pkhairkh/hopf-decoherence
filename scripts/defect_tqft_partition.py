#!/usr/bin/env python3
"""
Compute Defect TQFT partition functions on BTZ-relevant 3-manifolds.

The Defect TQFT extends the BCGP non-semisimple TQFT by ker(Delta)-valued
defect lines, producing O(1) entropy corrections from the hidden sector.
"""

import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hopf_decoherence.defect_tqft import (
    DefectTQFTData,
    btz_thermal_partition,
    btz_defect_partition,
    compute_entropy_correction,
    compute_entropy_deficit_limit,
    verify_defect_tqft_consistency,
)
from hopf_decoherence.rank_deficiency import D_ell


def main():
    print("=" * 80)
    print("  DEFECT TQFT: Partition Functions and Entropy Corrections")
    print("=" * 80)

    # Consistency checks for various ell
    print("\n--- Defect TQFT Consistency Checks ---")
    for ell in [3, 5, 7, 11]:
        result = verify_defect_tqft_consistency(ell, verbose=True)
        if not result['all_consistent']:
            print(f"  WARNING: Consistency check failed for ell={ell}")

    # Compute entropy corrections
    print("\n--- Entropy Corrections ---")
    print(f"  Limiting deficit: ln(6/5) = {compute_entropy_deficit_limit():.6f} nats")
    print(f"  Target (full non-invertible): -0.487 nats")

    corrections = {}
    for ell in [3, 5, 7, 11, 51, 101]:
        D = D_ell(ell)
        delta_s = compute_entropy_correction(ell)
        corrections[ell] = {
            'D_2': D,
            'deficiency_fraction': D / ell**3,
            'entropy_correction_nats': delta_s,
        }
        print(f"  ell={ell:>3}: D_2={D:>6}, D/ell^3={D/ell**3:.4f}, "
              f"Delta_S={delta_s:.6f} nats")

    # BTZ partition functions
    print("\n--- BTZ Partition Functions ---")
    for ell in [3, 5, 7]:
        for beta in [0.5, 1.0, 2.0]:
            Z_base = btz_thermal_partition(ell, beta)
            Z_defect = btz_defect_partition(ell, beta)
            ratio = abs(Z_defect / Z_base) if abs(Z_base) > 1e-15 else float('inf')
            print(f"  ell={ell}, beta={beta:.1f}: "
                  f"|Z_base|={abs(Z_base):.6f}, "
                  f"|Z_defect|={abs(Z_defect):.6f}, "
                  f"ratio={ratio:.6f}")

    # Compute kernel basis for ell=3
    print("\n--- Kernel Basis (ell=3) ---")
    data = DefectTQFTData(3)
    kernel = data.compute_kernel_basis()
    print(f"  Kernel dimension: {len(kernel)} (expected: {D_ell(3)})")
    print(f"  Each kernel vector has {kernel.shape[1]} PBW coefficients")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                                 'defect_tqft_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    output = {
        'entropy_corrections': {str(k): v for k, v in corrections.items()},
        'limiting_deficit_nats': compute_entropy_deficit_limit(),
        'target_correction_nats': -0.487,
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
