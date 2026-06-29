"""Test suite for coproduct rank deficiency formulas."""

import unittest
import math
import numpy as np

from hopf_decoherence.q_algebra import build_weyl_module_sl2, RANK_TOL
from hopf_decoherence.coproduct import (
    coproduct_matrices_sl2,
    compute_phi_rank,
    compute_cross_rep_coproduct_rank,
)
from hopf_decoherence.rank_deficiency import (
    D_ell, D_k1k2, expected_rank,
    verlinde_rank_sl3, verlinde_rank_sl3_computed,
    D_min_sl3, deficiency_fraction_sl2,
    deficiency_cascade, excess_sum,
    entropy_deficit_sl2,
)


class TestD2Formula(unittest.TestCase):
    """Test D_2(ell) = (ell^3 - ell)/6 = C(ell+1, 3)."""

    def test_small_values(self):
        """Test D(ell) for small ell values."""
        self.assertEqual(D_ell(3), 4)
        self.assertEqual(D_ell(5), 20)
        self.assertEqual(D_ell(7), 56)
        self.assertEqual(D_ell(9), 120)
        self.assertEqual(D_ell(11), 220)
        self.assertEqual(D_ell(13), 364)

    def test_binomial_form(self):
        """Test that D(ell) = C(ell+1, 3)."""
        for ell in range(3, 30, 2):
            d = D_ell(ell)
            n = ell + 1
            c = n * (n - 1) * (n - 2) // 6
            self.assertEqual(d, c)

    def test_integer_valued(self):
        """D(ell) should always be a positive integer for odd ell >= 3."""
        for ell in range(3, 100, 2):
            d = D_ell(ell)
            self.assertIsInstance(d, int)
            self.assertGreater(d, 0)

    def test_expected_rank(self):
        """rank(Phi) = ell^3 - D(ell) = (5*ell^3 + ell)/6."""
        for ell in [3, 5, 7, 9]:
            self.assertEqual(ell ** 3 - D_ell(ell), expected_rank(ell))


class TestNumericalVerification(unittest.TestCase):
    """Numerically verify D_2(ell) = dim ker(Phi) for small ell."""

    def test_ell_3(self):
        """ell=3: D=4, rank(Phi)=23. Steinberg j=ell-1=2."""
        rank, _, _, _ = compute_phi_rank(2, np.exp(2j * np.pi / 3), 3)
        self.assertEqual(rank, expected_rank(3))

    def test_ell_5(self):
        """ell=5: D=20, rank(Phi)=105. Steinberg j=ell-1=4."""
        rank, _, _, _ = compute_phi_rank(4, np.exp(2j * np.pi / 5), 5)
        self.assertEqual(rank, expected_rank(5))

    def test_ell_7(self):
        """ell=7: D=56, rank(Phi)=287. Steinberg j=ell-1=6."""
        rank, _, _, _ = compute_phi_rank(6, np.exp(2j * np.pi / 7), 7)
        self.assertEqual(rank, expected_rank(7))


class TestCrossRepDeficiency(unittest.TestCase):
    """Test D(ell, k1, k2) = C(k1+k2-ell+3, 3)."""

    def test_no_deficiency_below_threshold(self):
        """D should be 0 when k1+k2 < ell-1."""
        ell = 7
        for k1 in range(ell):
            for k2 in range(ell):
                if k1 + k2 < ell - 1:
                    self.assertEqual(D_k1k2(ell, k1, k2), 0)

    def test_steinberg_special_case(self):
        """For k1=k2=ell-1, D should equal D(ell)."""
        for ell in range(3, 16, 2):
            d_steinberg = D_k1k2(ell, ell - 1, ell - 1)
            d_formula = D_ell(ell)
            self.assertEqual(d_steinberg, d_formula)

    def test_symmetry(self):
        """D(ell, k1, k2) should be symmetric in k1, k2."""
        ell = 7
        for k1 in range(2 * ell):
            for k2 in range(2 * ell):
                self.assertEqual(D_k1k2(ell, k1, k2), D_k1k2(ell, k2, k1))


class TestExcessSum(unittest.TestCase):
    """Test that D = sum of (j-ell+1)^2 over non-simple CG summands."""

    def test_excess_sum_equals_D(self):
        """Verify D = sum (j-ell+1)^2 over non-simple CG summands j."""
        for ell in range(3, 16, 2):
            k = ell - 1
            self.assertEqual(excess_sum(ell, k, k), D_ell(ell),
                            f"excess_sum({ell},{k},{k}) != D({ell})")


class TestSL3Verlinde(unittest.TestCase):
    """Test sl_3 Verlinde rank formula."""

    def test_formula_matches_computation(self):
        """Closed-form formula should match direct summation."""
        for ell in range(3, 20):
            formula = verlinde_rank_sl3(ell)
            computed = verlinde_rank_sl3_computed(ell)
            self.assertEqual(formula, computed)

    def test_deficiency_positive(self):
        """D_min(ell) = ell^8 - V_rank(ell) should be positive for ell >= 3."""
        for ell in range(3, 20):
            self.assertGreater(D_min_sl3(ell), 0)


class TestEntropyBounds(unittest.TestCase):
    """Test entropy-related bounds."""

    def test_bounded_entropy(self):
        """Verify Delta_H <= ln(6/5)."""
        for ell in [3, 5, 7, 11, 101]:
            delta_h = entropy_deficit_sl2(ell)
            self.assertLess(delta_h, math.log(6 / 5) + 1e-10)

    def test_deficiency_fraction_approaches_limit(self):
        """D/ell^3 -> 1/6 as ell -> infinity."""
        for ell in [101, 501, 1001]:
            frac = deficiency_fraction_sl2(ell)
            self.assertAlmostEqual(frac, 1 / 6, places=3)


if __name__ == '__main__':
    unittest.main()
