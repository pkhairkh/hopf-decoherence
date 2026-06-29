#!/usr/bin/env python3
"""
Verify that J is NOT a Hopf ideal for u_q(sl_3) at roots of unity.

This is the critical structural input for the Defect TQFT construction:
the coproduct rank EXCEEDS the Verlinde rank because J is not a Hopf ideal.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hopf_decoherence.non_hopf_ideal import verify_sl3_not_hopf_ideal, verify_cross_hom_mechanism


def main():
    print("=" * 80)
    print("  VERIFICATION: J is NOT a Hopf ideal for u_q(sl_3)")
    print("=" * 80)

    # Part 1: J is not a Hopf ideal
    result1 = verify_sl3_not_hopf_ideal(ell=3, verbose=True)

    print(f"\n  Cross-term coefficient nonzero: {result1['cross_term_nonzero']}")
    print(f"  rank(Phi) = {result1['coproduct_rank']} > V_rank = {result1['verlinde_rank']}")
    print(f"  J is NOT a Hopf ideal: {result1['is_not_hopf_ideal']}")

    # Part 2: Cross-Hom mechanism
    print(f"\n{'='*80}")
    result2 = verify_cross_hom_mechanism(ell=3, verbose=True)

    all_pass = result1['is_not_hopf_ideal'] and result2['deficiency_equals_cross_hom']

    print(f"\n{'='*80}")
    print(f"OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*80}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
