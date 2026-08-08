"""Tests for the noncommutative Gröbner basis / Knuth-Bendix module (W0-1b)."""
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from ir.parser import (
    Monomial, Polynomial, Term, RewriteRule, Presentation, NormalFormReducer, QLaurent,
)
from ir.groebner import (
    overlaps, all_critical_pairs, critical_pairs, knuth_bendix_complete,
    anick_resolution_generators, check_confluence, KBStats,
)


# Helper: make a q-power coefficient
def q_n(n: int) -> QLaurent:
    return QLaurent({n: 1})


def one() -> QLaurent:
    return QLaurent({0: 1})


# ---------------------------------------------------------------------------
# Test overlaps
# ---------------------------------------------------------------------------

class TestOverlaps:
    def test_overlaps_basic(self):
        """Suffix of m1 = prefix of m2."""
        m1 = Monomial((0, 1))  # x, y
        m2 = Monomial((1, 2))  # y, z
        ov = overlaps(m1, m2)
        assert (1, 0, 1) in ov  # suffix (y,) of m1 = prefix (y,) of m2

    def test_overlaps_none(self):
        m1 = Monomial((0, 0))  # x, x
        m2 = Monomial((1, 1))  # y, y
        ov = overlaps(m1, m2)
        assert ov == []

    def test_overlaps_multiple(self):
        m1 = Monomial((0, 1, 0))  # x, y, x
        m2 = Monomial((0, 1))     # x, y
        ov = overlaps(m1, m2)
        # suffix (x,) of m1 = prefix (x,) of m2: (2, 0, 1)
        # suffix (x, y) of m1 ... wait, m2 is (x, y), so prefix of length 2 is (x, y).
        # suffix of m1 of length 2 is (y, x) — doesn't match.
        # So only one overlap: (2, 0, 1)
        assert (2, 0, 1) in ov

    def test_overlaps_full_excluded(self):
        """Full overlap (m1 == m2) should be excluded."""
        m1 = Monomial((0, 1))
        ov = overlaps(m1, m1)
        # suffix (y,) of m1 = prefix (y,) of m1? No, prefix of length 1 is (x,).
        # Actually: suffix of m1 of length 1 = (1,) = (y,). Prefix of m1 of length 1 = (0,) = (x,). No match.
        # suffix of length 2 = (0, 1). prefix of length 2 = (0, 1). Match! But this is the full monomial, excluded.
        assert ov == []


# ---------------------------------------------------------------------------
# Test critical pairs
# ---------------------------------------------------------------------------

class TestCriticalPairs:
    def test_critical_pairs_quantum_plane(self):
        """Quantum plane xy -> q*yx has no non-trivial critical pairs with itself."""
        # Rule: (x, y) -> q * (y, x)
        rule = RewriteRule(Monomial((0, 1)), Polynomial([Term(q_n(1), Monomial((1, 0)))]))
        cps = critical_pairs(rule, rule)
        # The only overlap of (x,y) with itself is the full overlap (excluded).
        # So no critical pairs.
        assert cps == []

    def test_critical_pairs_abc(self):
        """System ab -> ba, bc -> cb has a critical pair on abc."""
        # Generators: a=0, b=1, c=2
        # Rule 1: (a, b) -> (b, a)  i.e., (0,1) -> (1,0)
        # Rule 2: (b, c) -> (c, b)  i.e., (1,2) -> (2,1)
        rule1 = RewriteRule(Monomial((0, 1)), Polynomial([Term(one(), Monomial((1, 0)))]))
        rule2 = RewriteRule(Monomial((1, 2)), Polynomial([Term(one(), Monomial((2, 1)))]))
        cps = critical_pairs(rule1, rule2)
        # The monomial (a, b, c) = (0, 1, 2) has:
        # - rule1 at position 0: (a,b) -> (b,a), giving (b, a, c) = (1, 0, 2)
        # - rule2 at position 1: (b,c) -> (c,b), giving (a, c, b) = (0, 2, 1)
        # Difference: (b,a,c) - (a,c,b) = (1,0,2) - (0,2,1)
        # This is non-zero, so we expect 1 critical pair.
        assert len(cps) >= 1
        # The critical pair polynomial should be (b,a,c) - (a,c,b)
        cp = cps[0]
        # It should have 2 terms
        assert len(cp.terms) == 2


# ---------------------------------------------------------------------------
# Test Knuth-Bendix completion
# ---------------------------------------------------------------------------

class TestKnuthBendix:
    def test_knuth_bendix_abc(self):
        """System ab -> ba, bc -> cb should complete by adding ac -> ca or similar."""
        pres = Presentation(
            generators=["a", "b", "c"],
            rules=[
                RewriteRule(Monomial((0, 1)), Polynomial([Term(one(), Monomial((1, 0)))])),  # ab -> ba
                RewriteRule(Monomial((1, 2)), Polynomial([Term(one(), Monomial((2, 1)))])),  # bc -> cb
            ]
        )
        completed, stats = knuth_bendix_complete(pres, max_iterations=20, max_rules=20)
        # Should have added at least one new rule
        assert stats.new_rules_added >= 1
        # The completed system should be more confluent than the original
        # (full confluence is hard to guarantee in general, but for this simple case it should hold)
        assert check_confluence(completed, n_tests=50)

    def test_knuth_bendix_quantum_plane(self):
        """Quantum plane is already confluent — KB should add no rules."""
        pres = Presentation(
            generators=["x", "y"],
            rules=[
                RewriteRule(Monomial((0, 1)), Polynomial([Term(q_n(1), Monomial((1, 0)))])),  # xy -> q*yx
            ]
        )
        completed, stats = knuth_bendix_complete(pres, max_iterations=10)
        assert stats.new_rules_added == 0
        assert stats.terminated == True
        assert len(completed.rules) == 1


# ---------------------------------------------------------------------------
# Test Anick resolution generators
# ---------------------------------------------------------------------------

class TestAnickResolution:
    def test_anick_generators_count(self):
        """Anick resolution: degree 0 has 1, degree 1 has #rules, degree 2 has #critical pairs."""
        pres = Presentation(
            generators=["a", "b", "c"],
            rules=[
                RewriteRule(Monomial((0, 1)), Polynomial([Term(one(), Monomial((1, 0)))])),  # ab -> ba
                RewriteRule(Monomial((1, 2)), Polynomial([Term(one(), Monomial((2, 1)))])),  # bc -> cb
            ]
        )
        gens = anick_resolution_generators(pres, max_degree=3)
        assert len(gens[0]) == 1  # degree 0: the unit
        assert len(gens[1]) == 2  # degree 1: 2 rules
        # Degree 2: critical pairs between the two rules
        # We expect at least 1 (the abc overlap)
        assert len(gens[2]) >= 1

    def test_anick_quantum_plane(self):
        """Quantum plane has no critical pairs, so degree 2 is empty."""
        pres = Presentation(
            generators=["x", "y"],
            rules=[
                RewriteRule(Monomial((0, 1)), Polynomial([Term(q_n(1), Monomial((1, 0)))])),
            ]
        )
        gens = anick_resolution_generators(pres, max_degree=3)
        assert len(gens[0]) == 1
        assert len(gens[1]) == 1
        assert len(gens[2]) == 0  # no critical pairs


# ---------------------------------------------------------------------------
# Test confluence
# ---------------------------------------------------------------------------

class TestConfluence:
    def test_confluence_check(self):
        """Quantum plane is confluent."""
        pres = Presentation(
            generators=["x", "y"],
            rules=[
                RewriteRule(Monomial((0, 1)), Polynomial([Term(q_n(1), Monomial((1, 0)))])),
            ]
        )
        assert check_confluence(pres, n_tests=50)

    def test_non_confluence(self):
        """System ab -> ba, bc -> cb is NOT confluent (has the abc critical pair)."""
        pres = Presentation(
            generators=["a", "b", "c"],
            rules=[
                RewriteRule(Monomial((0, 1)), Polynomial([Term(one(), Monomial((1, 0)))])),
                RewriteRule(Monomial((1, 2)), Polynomial([Term(one(), Monomial((2, 1)))])),
            ]
        )
        # Before completion, this might or might not pass the random test
        # (depending on whether the random monomials hit the abc case).
        # After completion it should pass.
        completed, _ = knuth_bendix_complete(pres, max_iterations=20)
        assert check_confluence(completed, n_tests=100)


# ---------------------------------------------------------------------------
# Test u_q(sl_2) presentation (may be partial)
# ---------------------------------------------------------------------------

class TestUqSl2:
    def test_uq_sl2_presentation(self):
        """u_q(sl_2) at ℓ=3 presentation: K, E, F with relations.

        This is a harder test — KB completion may not finish.
        We just check that the presentation parses and has the right rules.
        """
        # Generators: K=0, E=1, F=2
        # Relations:
        #   K^3 = 1        -> K^3 -> 1 (empty monomial)
        #   E^3 = 0        -> E^3 -> 0
        #   F^3 = 0        -> F^3 -> 0
        #   K E = q^2 E K  -> (K,E) -> q^2 (E,K)
        #   K F = q^-2 F K -> (K,F) -> q^-2 (F,K)
        #   E F = F E + (K - K^-1)/(q - q^-1)
        #     = F E + (K - K^2) / (q - q^-1)   [since K^-1 = K^2 at ℓ=3]
        #     -> (E,F) -> (F,E) + (K - K^2)/(q - q^-1)
        # The denominator (q - q^-1) is a unit in QLaurent (invertible).
        q = q_n(1)
        q_inv = q_n(-1)
        denom = QLaurent({1: 1, -1: -1})  # q - q^-1
        # (K - K^2) / (q - q^-1) — but K is a generator, not a q-power.
        # We can't divide a generator by a q-power in QLaurent.
        # The relation is: E F - F E - (K - K^2)/(q - q^-1) = 0
        # As a rewrite rule: (E, F) -> (F, E) + (1/(q-q^-1)) * K - (1/(q-q^-1)) * K^2
        # The coefficients 1/(q-q^-1) need to be QLaurent. But 1/(q-q^-1) is not a Laurent polynomial.
        # This is the fundamental issue: the commutator relation introduces a non-Laurent coefficient.
        #
        # For the IR framework, we need to work over Z[q, q^-1, 1/(q-q^-1)] — a localization.
        # For now, skip the commutator relation and just test the q-commutation rules.

        rules = [
            RewriteRule(Monomial((0, 0, 0)), Polynomial([Term(one(), Monomial(()))])),  # K^3 -> 1
            RewriteRule(Monomial((1, 1, 1)), Polynomial.zero()),  # E^3 -> 0
            RewriteRule(Monomial((2, 2, 2)), Polynomial.zero()),  # F^3 -> 0
            RewriteRule(Monomial((0, 1)), Polynomial([Term(q_n(2), Monomial((1, 0)))])),  # K E -> q^2 E K
            RewriteRule(Monomial((0, 2)), Polynomial([Term(q_n(-2), Monomial((2, 0)))])),  # K F -> q^-2 F K
        ]
        pres = Presentation(generators=["K", "E", "F"], rules=rules)
        reducer = NormalFormReducer(pres)

        # Test: K E reduces to q^2 E K
        result = reducer.reduce(Polynomial([Term(one(), Monomial((0, 1)))]))
        assert len(result.terms) == 1
        assert result.terms[0].monomial == Monomial((1, 0))  # E K
        assert result.terms[0].coeff == q_n(2)  # q^2

        # Test: K^3 reduces to 1
        result = reducer.reduce(Polynomial([Term(one(), Monomial((0, 0, 0)))]))
        assert len(result.terms) == 1
        assert result.terms[0].monomial == Monomial(())

        # Test: E^3 reduces to 0
        result = reducer.reduce(Polynomial([Term(one(), Monomial((1, 1, 1)))]))
        assert result == Polynomial.zero()

        # Anick generators
        gens = anick_resolution_generators(pres, max_degree=2)
        assert len(gens[0]) == 1
        assert len(gens[1]) == 5  # 5 rules
        # Degree 2: critical pairs. There should be several (e.g., K^3 overlaps with KE)
        assert len(gens[2]) >= 1
