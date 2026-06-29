"""Test suite for q_algebra module: quantum group representations and relations."""

import unittest
import numpy as np

from hopf_decoherence.q_algebra import (
    q_number, q_factorial, q_binomial,
    build_weyl_module_sl2, build_weyl_module,
    verify_algebra_relations, verify_sl3_relations,
    _build_sl3_L10, _build_sl3_L01,
    compute_rank, RANK_TOL,
)


class TestQNumbers(unittest.TestCase):
    """Test q-number utilities."""

    def test_q_number_basic(self):
        """[0]_q = 0, [1]_q = 1 for generic q."""
        q = np.exp(1.7j)
        self.assertAlmostEqual(q_number(0, q), 0)
        self.assertAlmostEqual(q_number(1, q), 1, places=10)

    def test_q_number_root_of_unity(self):
        """[ell]_q = 0 at q = exp(2*pi*i/ell)."""
        for ell in [3, 5, 7]:
            q = np.exp(2j * np.pi / ell)
            self.assertAlmostEqual(abs(q_number(ell, q)), 0, places=10)

    def test_q_binomial_collapse(self):
        """[ell choose r]_q = 0 for 1 <= r <= ell-1 at root of unity."""
        for ell in [3, 5]:
            q = np.exp(2j * np.pi / ell)
            for r in range(1, ell):
                self.assertAlmostEqual(abs(q_binomial(ell, r, q)), 0, places=8)

    def test_q_binomial_endpoints(self):
        """[n choose 0]_q = [n choose n]_q = 1."""
        q = np.exp(1.7j)
        for n in range(1, 10):
            self.assertAlmostEqual(abs(q_binomial(n, 0, q)), 1, places=10)
            self.assertAlmostEqual(abs(q_binomial(n, n, q)), 1, places=10)


class TestSL2Representations(unittest.TestCase):
    """Test sl_2 Weyl module construction and algebra relations."""

    def test_fundamental_relations(self):
        """Verify u_q(sl_2) fundamental rep relations."""
        for ell in [3, 5, 7, 11]:
            q = np.exp(2j * np.pi / ell)
            K, E, F = build_weyl_module_sl2(1, q)
            ok, errs = verify_algebra_relations(K, E, F, q)
            self.assertTrue(ok, f"Fundamental rep fails at ell={ell}: errs={errs}")

    def test_steinberg_relations(self):
        """Verify Steinberg module L(ell-1) relations at root of unity."""
        for ell in [3, 5, 7]:
            q = np.exp(2j * np.pi / ell)
            j = ell - 1  # Highest weight j = ell-1, dim = ell
            K, E, F = build_weyl_module_sl2(j, q)
            ok, errs = verify_algebra_relations(K, E, F, q)
            self.assertTrue(ok, f"Steinberg rep fails at ell={ell}: errs={errs}")

    def test_generic_q_relations(self):
        """Verify algebra relations at generic q."""
        q = np.exp(1.7j)
        for j in [1, 2, 3, 5]:
            K, E, F = build_weyl_module_sl2(j, q)
            ok, errs = verify_algebra_relations(K, E, F, q)
            self.assertTrue(ok, f"Generic q rep fails at j={j}: errs={errs}")

    def test_K_eigenvalues(self):
        """K diagonal entries are q^{j-2m} in highest-weight-first order."""
        q = np.exp(2j * np.pi / 5)
        j = 2  # dim = 3
        K, E, F = build_weyl_module_sl2(j, q)
        for m in range(j + 1):
            expected = q ** (j - 2 * m)
            self.assertAlmostEqual(K[m, m], expected, places=10)


class TestSL3Representations(unittest.TestCase):
    """Test sl_3 Weyl module construction and algebra relations."""

    def test_fundamental_L10(self):
        """Verify L(1,0) fundamental rep relations."""
        for ell in [3, 5, 7]:
            q = np.exp(2j * np.pi / ell)
            K1, K2, E1, E2, F1, F2 = _build_sl3_L10(q)
            ok, errs = verify_sl3_relations(K1, K2, E1, E2, F1, F2, q)
            self.assertTrue(ok, f"L(1,0) fails at ell={ell}: errs={errs}")

    def test_dual_fundamental_L01(self):
        """Verify L(0,1) dual fundamental rep relations."""
        for ell in [3, 5, 7]:
            q = np.exp(2j * np.pi / ell)
            K1, K2, E1, E2, F1, F2 = _build_sl3_L01(q)
            ok, errs = verify_sl3_relations(K1, K2, E1, E2, F1, F2, q)
            self.assertTrue(ok, f"L(0,1) fails at ell={ell}: errs={errs}")

    def test_nonsimple_roots(self):
        """Verify non-simple root vectors E12, F12 for sl_3."""
        for ell in [3, 5, 7]:
            q = np.exp(2j * np.pi / ell)
            K1, K2, E1, E2, F1, F2 = _build_sl3_L10(q)
            K1inv = np.linalg.inv(K1)
            K2inv = np.linalg.inv(K2)

            E12 = E1 @ E2 - q * E2 @ E1
            F12 = F2 @ F1 - q ** (-1) * F1 @ F2

            # E12 should have K-eigenvalue q for both K1 and K2
            self.assertAlmostEqual(np.max(np.abs(K1 @ E12 @ K1inv - q * E12)), 0, places=10)
            self.assertAlmostEqual(np.max(np.abs(K2 @ E12 @ K2inv - q * E12)), 0, places=10)
            self.assertAlmostEqual(np.max(np.abs(K1 @ F12 @ K1inv - q ** (-1) * F12)), 0, places=10)


class TestBuildWeylModule(unittest.TestCase):
    """Test the unified build_weyl_module entry point."""

    def test_sl2_dispatch(self):
        """build_weyl_module(2, j, q) should dispatch to sl_2 builder."""
        q = np.exp(1.7j)
        K, E, F = build_weyl_module(2, 2, q)
        self.assertEqual(K.shape, (3, 3))

    def test_sl3_dispatch(self):
        """build_weyl_module(3, (1,0), q) should dispatch to sl_3 builder."""
        q = np.exp(2j * np.pi / 3)
        result = build_weyl_module(3, (1, 0), q)
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0].shape, (3, 3))


if __name__ == '__main__':
    unittest.main()
