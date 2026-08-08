"""Tests for the AST/IR parser module (task W0-1a).

These tests verify the foundational components of the AST/IR framework:

  - QLaurent (Z[q, q^{-1}] coefficient ring)
  - Monomial (tuple of generator indices)
  - Term (coeff * monomial)
  - Polynomial (sum of terms, with normalization)
  - RewriteRule (LHS monomial -> RHS polynomial, with matches/apply)
  - Presentation (generators + rules, with a string parser)
  - NormalFormReducer (leftmost-match reduction)

The integration test (test_quantum_plane_normal_form) verifies the
quantum plane ``x*y = q*y*x``:

  - ``x*y   -> q*y*x``        (one rule application)
  - ``x*y*x -> q*y*x*x``      (one application at position 0)
  - ``x*x*y -> q^2*y*x*x``    (two applications)

Run::

    pytest tests/test_ir_parser.py -v
"""

import os
import sys

# Make the ir/ package importable (it lives at the repo root, not under src/).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ir.parser import (  # noqa: E402
    QLaurent,
    Monomial,
    Term,
    Polynomial,
    RewriteRule,
    Presentation,
    NormalFormReducer,
    ZERO,
    ONE,
    Q,
    Q_INV,
    qpow,
    qint,
)


# ----------------------------------------------------------------------------
# QLaurent (Z[q, q^{-1}] coefficient ring)
# ----------------------------------------------------------------------------


class TestQLaurent:
    """Tests for the QLaurent coefficient ring."""

    def test_qpow_basic(self):
        """q^0 = 1, q^1 = q, q^(-1) = q^{-1}, q^3 has a single term {3: 1}."""
        assert qpow(0) == ONE
        assert qpow(0).is_one()
        assert qpow(1) == Q
        assert qpow(-1) == Q_INV
        assert qpow(3).terms == {3: 1}
        assert qpow(-2).terms == {-2: 1}

    def test_qint_basic(self):
        """qint(n) -> {0: n}; qint(0) is zero."""
        assert qint(0).is_zero()
        assert qint(1).is_one()
        assert qint(5).terms == {0: 5}
        assert qint(-3).terms == {0: -3}

    def test_addition(self):
        """(q + 1) + (q - 1) = 2*q; cancellation works."""
        p = qpow(1) + qint(1)        # q + 1
        q = qpow(1) + qint(-1)       # q - 1
        r = p + q                    # 2*q
        assert r == qint(2) * Q
        assert r.terms == {1: 2}

    def test_multiplication(self):
        """(q + 1) * (q - 1) = q^2 - 1."""
        p = qpow(1) + qint(1)
        q = qpow(1) - qint(1)
        r = p * q
        assert r == qpow(2) - qint(1)
        assert r.terms == {2: 1, 0: -1}

    def test_distributive(self):
        """q^2 * (q - q^{-1}) = q^3 - q."""
        p = qpow(2)
        q = qpow(1) - qpow(-1)
        assert p * q == qpow(3) - qpow(1)

    def test_negation(self):
        """-(2*q + 3) = -2*q - 3."""
        p = qint(2) * Q + qint(3)
        assert -p == qint(-2) * Q + qint(-3)

    def test_zero_and_one(self):
        """ZERO and ONE behave as additive / multiplicative identities."""
        assert (ZERO + Q).is_one() is False
        assert (ZERO + Q) == Q
        assert (ONE * Q) == Q
        assert ZERO.is_zero()
        assert ONE.is_one()
        # Zero times anything is zero.
        assert (ZERO * Q).is_zero()
        assert (Q * ZERO).is_zero()

    def test_hash_equality(self):
        """Equal QLaurents hash equally."""
        a = qpow(2) + qint(1)
        b = qint(1) + qpow(2)
        assert a == b
        assert hash(a) == hash(b)


# ----------------------------------------------------------------------------
# Monomial
# ----------------------------------------------------------------------------


class TestMonomial:
    """Tests for the Monomial class."""

    def test_monomial_basic(self):
        """Length, indexing, slicing, concatenation, equality, hashing."""
        m = Monomial((0, 1, 0))
        assert len(m) == 3
        # Indexing returns a single-element Monomial.
        assert m[0] == Monomial((0,))
        assert m[1] == Monomial((1,))
        # Slicing returns a Monomial.
        assert m[1:] == Monomial((1, 0))
        assert m[:2] == Monomial((0, 1))
        # Concatenation (the free-algebra product).
        assert m + Monomial((2,)) == Monomial((0, 1, 0, 2))
        assert m * Monomial((3,)) == Monomial((0, 1, 0, 3))
        # Equality and hashing.
        assert m == Monomial((0, 1, 0))
        assert hash(m) == hash(Monomial((0, 1, 0)))
        # Empty monomial is the identity.
        e = Monomial.empty()
        assert len(e) == 0
        assert e.is_empty()
        assert e + m == m
        assert m + e == m

    def test_order_key_length_lex(self):
        """order_key() gives length-lex: shorter first, then lex."""
        keys = [
            Monomial((0, 1, 0)).order_key(),
            Monomial((1, 0)).order_key(),
            Monomial((0, 0, 0)).order_key(),
            Monomial((0,)).order_key(),
            Monomial((1, 1)).order_key(),
        ]
        # Sort and verify the expected order.
        sorted_keys = sorted(keys)
        # Expected: (1,(0,)) < (2,(1,0)) < (2,(1,1)) < (3,(0,0,0)) < (3,(0,1,0))
        assert sorted_keys[0] == (1, (0,))
        assert sorted_keys[1] == (2, (1, 0))
        assert sorted_keys[2] == (2, (1, 1))
        assert sorted_keys[3] == (3, (0, 0, 0))
        assert sorted_keys[4] == (3, (0, 1, 0))


# ----------------------------------------------------------------------------
# Polynomial
# ----------------------------------------------------------------------------


class TestPolynomial:
    """Tests for the Polynomial class."""

    def test_polynomial_addition(self):
        """Like monomials combine; cancellation drops zero-coefficient terms."""
        p1 = Polynomial([
            Term(qint(2), Monomial((0, 1))),
            Term(ONE, Monomial((1, 0))),
        ])
        p2 = Polynomial([
            Term(qint(-2), Monomial((0, 1))),
            Term(ONE, Monomial((0, 0))),
        ])
        result = p1 + p2
        # 2*(0,1) cancels with -2*(0,1), leaving (0,0) + (1,0).
        # After normalize, terms are sorted by length-lex: (0,0) then (1,0).
        expected = Polynomial([
            Term(ONE, Monomial((0, 0))),
            Term(ONE, Monomial((1, 0))),
        ])
        assert result == expected
        assert len(result.normalize().terms) == 2

    def test_polynomial_scalar_multiplication(self):
        """Scalar mult distributes and normalizes."""
        p = Polynomial([
            Term(ONE, Monomial((0, 1))),
            Term(qint(2), Monomial((1, 0))),
        ])
        # Multiply by q.
        r = p * Q
        expected = Polynomial([
            Term(Q, Monomial((0, 1))),
            Term(qint(2) * Q, Monomial((1, 0))),
        ])
        assert r == expected
        # Left multiplication by an int.
        r2 = 3 * p
        expected2 = Polynomial([
            Term(qint(3), Monomial((0, 1))),
            Term(qint(6), Monomial((1, 0))),
        ])
        assert r2 == expected2

    def test_polynomial_multiplication(self):
        """Polynomial * Polynomial distributes via the free-algebra product."""
        # (x + y) * (x - y) = x*x - x*y + y*x - y*y, with indices x=0, y=1.
        p = Polynomial([
            Term(ONE, Monomial((0,))),
            Term(ONE, Monomial((1,))),
        ])
        q = Polynomial([
            Term(ONE, Monomial((0,))),
            Term(-ONE, Monomial((1,))),
        ])
        r = p * q
        # Expected: (0,0) - (0,1) + (1,0) - (1,1).  After normalize, sorted by
        # length-lex: (0,0), (0,1), (1,0), (1,1) with signs +, -, +, -.
        expected = Polynomial([
            Term(ONE, Monomial((0, 0))),
            Term(-ONE, Monomial((0, 1))),
            Term(ONE, Monomial((1, 0))),
            Term(-ONE, Monomial((1, 1))),
        ])
        assert r == expected

    def test_polynomial_normalize_cancels(self):
        """normalize() combines like terms and drops zero coefficients."""
        p = Polynomial([
            Term(qint(3), Monomial((0, 1))),
            Term(qint(-3), Monomial((0, 1))),
            Term(ONE, Monomial((1,))),
        ])
        n = p.normalize()
        # The (0,1) terms cancel, leaving only (1,).
        assert n == Polynomial([Term(ONE, Monomial((1,)))])

    def test_polynomial_negation(self):
        """-p has all coefficients negated."""
        p = Polynomial([
            Term(qint(2), Monomial((0,))),
            Term(-Q, Monomial((1,))),
        ])
        assert -p == Polynomial([
            Term(qint(-2), Monomial((0,))),
            Term(Q, Monomial((1,))),
        ])


# ----------------------------------------------------------------------------
# RewriteRule
# ----------------------------------------------------------------------------


class TestRewriteRule:
    """Tests for the RewriteRule class."""

    def test_rewrite_rule_match(self):
        """matches() correctly identifies positions; apply() replaces LHS."""
        # Rule: (0, 1) -> q*(1, 0).
        rule = RewriteRule(
            Monomial((0, 1)),
            Polynomial([Term(qpow(1), Monomial((1, 0)))]),
        )
        # Match at position 0 of (0, 1, 0): gens[0:2] = (0, 1) == lhs.
        assert rule.matches(Monomial((0, 1, 0)), 0)
        # No match at position 1: gens[1:3] = (1, 0) != (0, 1).
        assert not rule.matches(Monomial((0, 1, 0)), 1)
        # Out-of-range positions.
        assert not rule.matches(Monomial((0, 1, 0)), -1)
        assert not rule.matches(Monomial((0, 1, 0)), 2)
        assert not rule.matches(Monomial((0, 1, 0)), 100)
        # Apply at position 0: prefix=(), suffix=(0,), result = q*(1, 0, 0).
        result = rule.apply(Monomial((0, 1, 0)), 0)
        assert result == Polynomial([Term(qpow(1), Monomial((1, 0, 0)))])
        # Apply at position 1 of (0, 0, 1): prefix=(0,), suffix=(),
        # result = q*(0, 1, 0).
        result2 = rule.apply(Monomial((0, 0, 1)), 1)
        assert result2 == Polynomial([Term(qpow(1), Monomial((0, 1, 0)))])

    def test_rewrite_rule_apply_raises_on_mismatch(self):
        """apply() raises ValueError if the rule doesn't match."""
        rule = RewriteRule(
            Monomial((0, 1)),
            Polynomial([Term(Q, Monomial((1, 0)))]),
        )
        try:
            rule.apply(Monomial((1, 0, 1)), 0)
            raise AssertionError("Expected ValueError for non-matching apply")
        except ValueError:
            pass

    def test_rewrite_rule_rejects_empty_lhs(self):
        """A rule with an empty LHS is rejected (it would match everywhere)."""
        try:
            RewriteRule(Monomial(()), Polynomial([Term(ONE, Monomial(()))]))
            raise AssertionError("Expected ValueError for empty LHS")
        except ValueError:
            pass


# ----------------------------------------------------------------------------
# Presentation + parser
# ----------------------------------------------------------------------------


class TestPresentation:
    """Tests for the Presentation class and its parser."""

    def test_presentation_parse(self):
        """parse() turns a string into the correct Polynomial."""
        pres = Presentation(["K", "E"], [])
        # Parse "q^2 * K * E - E * K".
        p = pres.parse("q^2 * K * E - E * K")
        # Expected: q^2*(K, E) - (E, K), with K=0, E=1.
        expected = Polynomial([
            Term(qpow(2), Monomial((0, 1))),
            Term(-ONE, Monomial((1, 0))),
        ])
        assert p == expected

    def test_parse_no_spaces(self):
        """The parser is whitespace-insensitive."""
        pres = Presentation(["x", "y"], [])
        assert pres.parse("q*x*y") == pres.parse("q * x * y")
        assert pres.parse("x*y-y*x") == pres.parse("x*y - y*x")

    def test_parse_q_negative_exponent(self):
        """q^-N is parsed correctly."""
        pres = Presentation(["x", "y"], [])
        p = pres.parse("q^-2 * x")
        assert p == Polynomial([Term(qpow(-2), Monomial((0,)))])

    def test_parse_integer_coefficient(self):
        """Integer coefficients (with no q factor) are parsed correctly."""
        pres = Presentation(["x", "y"], [])
        p = pres.parse("3 * x - 2 * y")
        assert p == Polynomial([
            Term(qint(3), Monomial((0,))),
            Term(qint(-2), Monomial((1,))),
        ])

    def test_parse_parens(self):
        """Parenthesized sub-polynomials can be used as factors."""
        pres = Presentation(["x", "y"], [])
        # (q + 1) * x = q*x + x.
        p = pres.parse("(q + 1) * x")
        assert p == Polynomial([
            Term(Q, Monomial((0,))),
            Term(ONE, Monomial((0,))),
        ])

    def test_parse_leading_minus(self):
        """A leading minus sign is respected."""
        pres = Presentation(["x", "y"], [])
        p = pres.parse("-x * y")
        assert p == Polynomial([Term(-ONE, Monomial((0, 1)))])

    def test_parse_single_generator(self):
        """A bare generator name parses to a length-1 monomial."""
        pres = Presentation(["x", "y"], [])
        assert pres.parse("x") == Polynomial([Term(ONE, Monomial((0,)))])
        assert pres.parse("y") == Polynomial([Term(ONE, Monomial((1,)))])

    def test_parse_rejects_unknown_name(self):
        """An unknown generator name raises ValueError."""
        pres = Presentation(["x", "y"], [])
        try:
            pres.parse("x * z")
            raise AssertionError("Expected ValueError for unknown name 'z'")
        except ValueError:
            pass


# ----------------------------------------------------------------------------
# NormalFormReducer (integration: the quantum plane)
# ----------------------------------------------------------------------------


def _quantum_plane_presentation() -> Presentation:
    """Build the quantum plane presentation: generators x, y; rule (x,y) -> q*(y,x)."""
    # Generators: x = 0, y = 1.
    # Rule: (x, y) -> q * (y, x).
    rule = RewriteRule(
        Monomial((0, 1)),
        Polynomial([Term(qpow(1), Monomial((1, 0)))]),
    )
    return Presentation(["x", "y"], [rule])


class TestQuantumPlane:
    """Integration tests for the quantum plane ``x*y = q*y*x``."""

    def test_quantum_plane_normal_form(self):
        """The quantum plane reduces monomials to y^a * x^b form.

        With rule (x, y) -> q*(y, x), the normal form moves all x's to
        the right (so the leading y's come first).  We verify:

          - x*y   -> q*y*x
          - x*y*x -> q*y*x*x
          - x*x*y -> q^2*y*x*x
          - y*x*x is already normal
          - x*y - q*y*x reduces to 0
          - x*y*x*y -> q^3*y*y*x*x
        """
        pres = _quantum_plane_presentation()
        reducer = NormalFormReducer(pres)

        # x*y -> q*y*x  (one rule application at position 0).
        nf_xy = reducer.normal_form(Monomial((0, 1)))
        assert nf_xy == Polynomial([Term(qpow(1), Monomial((1, 0)))]), \
            f"x*y -> {nf_xy}"

        # x*y*x -> q*y*x*x  (apply at position 0, no further matches).
        nf_xyx = reducer.normal_form(Monomial((0, 1, 0)))
        assert nf_xyx == Polynomial([Term(qpow(1), Monomial((1, 0, 0)))]), \
            f"x*y*x -> {nf_xyx}"

        # x*x*y -> q^2*y*x*x  (apply at position 1: x*y -> q*y*x, giving
        # q*x*y*x; then apply at position 1 again: x*y -> q*y*x, giving
        # q^2*y*x*x).
        nf_xxy = reducer.normal_form(Monomial((0, 0, 1)))
        assert nf_xxy == Polynomial([Term(qpow(2), Monomial((1, 0, 0)))]), \
            f"x*x*y -> {nf_xxy}"

        # y*x*x is already in normal form (no (x,y) subsequence).
        nf_yxx = reducer.normal_form(Monomial((1, 0, 0)))
        assert nf_yxx == Polynomial([Term(ONE, Monomial((1, 0, 0)))]), \
            f"y*x*x -> {nf_yxx}"

        # x*y*x*y -> q^3*y*y*x*x  (4 monomials: x*y at pos 0 -> q*y*x*x*y,
        # then x*y at pos 2 -> q^2*y*x*y*x, then x*y at pos 1 -> q^3*y*y*x*x).
        nf_xyxy = reducer.normal_form(Monomial((0, 1, 0, 1)))
        assert nf_xyxy == Polynomial([Term(qpow(3), Monomial((1, 1, 0, 0)))]), \
            f"x*y*x*y -> {nf_xyxy}"

    def test_quantum_plane_reduce_polynomial(self):
        """The relation x*y - q*y*x reduces to 0 (as it should)."""
        pres = _quantum_plane_presentation()
        reducer = NormalFormReducer(pres)
        p = Polynomial([
            Term(ONE, Monomial((0, 1))),
            Term(-qpow(1), Monomial((1, 0))),
        ])
        assert reducer.reduce(p).is_zero(), \
            f"x*y - q*y*x should reduce to 0, got {reducer.reduce(p)}"

    def test_quantum_plane_via_parser(self):
        """End-to-end: parse a string, reduce it, check the result."""
        pres = _quantum_plane_presentation()
        reducer = NormalFormReducer(pres)
        # Parse "x*x*y" and reduce: should give q^2 * y * x * x.
        p = pres.parse("x * x * y")
        reduced = reducer.reduce(p)
        expected = Polynomial([Term(qpow(2), Monomial((1, 0, 0)))])
        assert reduced == expected, f"x*x*y -> {reduced}, expected {expected}"

    def test_quantum_plane_longer_monomial(self):
        """x^4 * y^3 -> q^12 * y^3 * x^4 (12 inversions of an x past a y)."""
        pres = _quantum_plane_presentation()
        reducer = NormalFormReducer(pres)
        # x*x*x*x*y*y*y has 4*3 = 12 (x,y) pairs to swap.
        mon = Monomial((0, 0, 0, 0, 1, 1, 1))
        nf = reducer.normal_form(mon)
        assert nf == Polynomial([Term(qpow(12), Monomial((1, 1, 1, 0, 0, 0, 0)))]), \
            f"x^4*y^3 -> {nf}"


# ----------------------------------------------------------------------------
# Sanity: identity presentation (no rules)
# ----------------------------------------------------------------------------


class TestNoRules:
    """A presentation with no rules is the free algebra: every monomial is normal."""

    def test_empty_presentation_is_normal(self):
        """With no rules, normal_form returns the input monomial unchanged."""
        pres = Presentation(["x", "y"], [])
        reducer = NormalFormReducer(pres)
        for mon in [Monomial(()), Monomial((0,)), Monomial((0, 1)),
                    Monomial((1, 0, 1)), Monomial((0, 1, 0, 1, 0))]:
            nf = reducer.normal_form(mon)
            assert nf == Polynomial([Term(ONE, mon)]), \
                f"With no rules, {mon} should be its own normal form, got {nf}"


if __name__ == "__main__":
    import unittest
    unittest.main()
