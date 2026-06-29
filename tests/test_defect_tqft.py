"""Test suite for Defect TQFT construction and non-Hopf ideal property."""

import unittest
import numpy as np

from hopf_decoherence.non_hopf_ideal import (
    verify_sl3_not_hopf_ideal,
    verify_cross_hom_mechanism,
)
from hopf_decoherence.defect_tqft import (
    DefectTQFTData,
    compute_entropy_correction,
    compute_entropy_deficit_limit,
    verify_defect_tqft_consistency,
)
from hopf_decoherence.rank_deficiency import D_ell


class TestNonHopfIdeal(unittest.TestCase):
    """Test that J is NOT a Hopf ideal for sl_3."""

    def test_cross_term_nonzero(self):
        """Cross-term coefficient (1+q^2) should be nonzero at ell=3."""
        result = verify_sl3_not_hopf_ideal(ell=3)
        self.assertTrue(result['cross_term_nonzero'])

    def test_coproduct_rank_exceeds_verlinde(self):
        """rank(Phi) should exceed V_rank for sl_3."""
        result = verify_sl3_not_hopf_ideal(ell=3)
        self.assertTrue(result['rank_exceeds_verlinde'])

    def test_not_hopf_ideal(self):
        """J is NOT a Hopf ideal for sl_3."""
        result = verify_sl3_not_hopf_ideal(ell=3)
        self.assertTrue(result['is_not_hopf_ideal'])

    def test_coproduct_rank_value(self):
        """rank(Phi on R tensor R) at ell=3 should be 155."""
        result = verify_sl3_not_hopf_ideal(ell=3)
        self.assertEqual(result['coproduct_rank'], 155)


class TestCrossHomMechanism(unittest.TestCase):
    """Test the cross-Hom mechanism for sl_3."""

    def test_deficiency_equals_cross_hom(self):
        """Deficiency should equal cross-Hom dimension for L(1,0) tensor L(1,0)."""
        result = verify_cross_hom_mechanism(ell=3)
        self.assertTrue(result['deficiency_equals_cross_hom'])

    def test_rank_equals_block_diagonal(self):
        """Coproduct rank should equal block-diagonal rank."""
        result = verify_cross_hom_mechanism(ell=3)
        self.assertTrue(result['rank_equals_block_diagonal'])


class TestDefectTQFT(unittest.TestCase):
    """Test Defect TQFT construction."""

    def test_kernel_basis_dimension(self):
        """Kernel basis dimension should match D_2(ell)."""
        data = DefectTQFTData(3)
        kernel = data.compute_kernel_basis()
        self.assertEqual(len(kernel), D_ell(3))

    def test_entropy_correction_positive(self):
        """Entropy correction from defect sector should be positive."""
        for ell in [3, 5, 7]:
            delta_s = compute_entropy_correction(ell)
            self.assertGreater(delta_s, 0)

    def test_entropy_correction_bounded(self):
        """Entropy correction should be less than ln(6/5)."""
        limit = compute_entropy_deficit_limit()
        for ell in [3, 5, 7, 11]:
            delta_s = compute_entropy_correction(ell)
            self.assertLess(delta_s, limit + 0.01)

    def test_consistency_checks(self):
        """All consistency checks should pass."""
        for ell in [3, 5, 7]:
            result = verify_defect_tqft_consistency(ell)
            self.assertTrue(result['all_consistent'],
                          f"Consistency check failed for ell={ell}: {result}")

    def test_deficit_limit(self):
        """Limiting deficit should be ln(6/5)."""
        import math
        limit = compute_entropy_deficit_limit()
        self.assertAlmostEqual(limit, math.log(6 / 5), places=10)


if __name__ == '__main__':
    unittest.main()
