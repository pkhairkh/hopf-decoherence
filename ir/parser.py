"""AST/IR parser and normal form reducer for noncommutative algebras.

This is the foundational module of the AST/IR framework for the
hopf-decoherence project (task W0-1a of the AST/IR research programme).
It provides:

- :class:`QLaurent` -- the coefficient ring ``Z[q, q^{-1}]`` (Laurent
  polynomials in ``q`` with integer coefficients), with addition,
  multiplication, negation, equality, hashing, and a ``qpow`` helper.
- :class:`Monomial` -- a tuple of generator indices (the free algebra's
  basis elements), with length, concatenation, slicing, hashing, equality.
- :class:`Term` -- a coefficient times a monomial.
- :class:`Polynomial` -- a sum of terms, with addition, scalar
  multiplication, negation, polynomial multiplication, and normalization
  (combine like monomials, drop zero coefficients, sort by length-lex).
- :class:`RewriteRule` -- ``LHS monomial -> RHS polynomial``, with
  ``matches(monomial, position)`` and ``apply(monomial, position)``.
- :class:`Presentation` -- generators (with names) and rewrite rules,
  plus ``parse(s)`` for parsing strings like ``"q^2 * K * E - E * K"``
  into a :class:`Polynomial`.
- :class:`NormalFormReducer` -- given a (confluent, terminating) rewrite
  system, reduce any polynomial to normal form.

The next sub-agent (W0-1b) will build Knuth-Bendix completion on top of
this module.

Monomial order
--------------
We use length-lex throughout: monomials are ordered first by length
(shorter first), then lexicographically on the tuple of generator
indices.  This is exposed via :meth:`Monomial.order_key`.  The reducer
itself does not depend on the monomial order; the order is used by
:meth:`Polynomial.normalize` for a canonical term ordering and will be
used by the Knuth-Bendix completion in W0-1b.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Tuple, Union


# ============================================================================
# Coefficient ring: Z[q, q^{-1}] (Laurent polynomials in q with integer coeffs)
# ============================================================================
#
# Each element is stored as a sparse dict {exponent: coefficient} with no
# zero coefficients.  For example, 2*q^3 - q + 5 is stored as
# {3: 2, 1: -1, 0: 5}.
#
# This ring is generic (does not assume q^N = 1 for any N), so it works at
# any root of unity.  When a specific root is needed (e.g., q = omega at
# ell = 3), the caller is responsible for substituting the root -- or for
# building a custom ring like the ER class in scripts/certify_a1_exact.py.
# ============================================================================


class QLaurent:
    """Element of ``Z[q, q^{-1}]``, stored as ``{exponent: coefficient}``.

    The dict has no entries with coefficient 0 (this is the canonical form;
    all operations preserve it).
    """

    __slots__ = ("terms",)

    def __init__(self, terms: Optional[dict] = None) -> None:
        if terms is None:
            self.terms: dict = {}
        elif isinstance(terms, dict):
            # Drop zero coefficients defensively.
            self.terms = {e: c for e, c in terms.items() if c != 0}
        else:
            raise TypeError(
                f"QLaurent: expected dict or None, got {type(terms).__name__}"
            )

    # --- constructors -------------------------------------------------------

    @classmethod
    def from_int(cls, n: int) -> "QLaurent":
        """Return the integer ``n`` as a :class:`QLaurent`."""
        if n == 0:
            return cls({})
        return cls({0: n})

    @classmethod
    def qpow(cls, n: int) -> "QLaurent":
        """Return ``q^n`` as a :class:`QLaurent` (``n`` any integer)."""
        if n == 0:
            return cls({0: 1})
        return cls({n: 1})

    # --- coercion -----------------------------------------------------------

    @staticmethod
    def _coerce(other) -> "QLaurent":
        if isinstance(other, QLaurent):
            return other
        if isinstance(other, int):
            return QLaurent.from_int(other)
        return NotImplemented

    # --- arithmetic ---------------------------------------------------------

    def __add__(self, other) -> "QLaurent":
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        new_terms = dict(self.terms)
        for e, c in other.terms.items():
            new_c = new_terms.get(e, 0) + c
            if new_c == 0:
                new_terms.pop(e, None)
            else:
                new_terms[e] = new_c
        return QLaurent(new_terms)

    def __radd__(self, other) -> "QLaurent":
        return self.__add__(other)

    def __neg__(self) -> "QLaurent":
        return QLaurent({e: -c for e, c in self.terms.items()})

    def __sub__(self, other) -> "QLaurent":
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other) -> "QLaurent":
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other + (-self)

    def __mul__(self, other) -> "QLaurent":
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        new_terms: dict = {}
        for e1, c1 in self.terms.items():
            for e2, c2 in other.terms.items():
                e = e1 + e2
                c = new_terms.get(e, 0) + c1 * c2
                if c == 0:
                    new_terms.pop(e, None)
                else:
                    new_terms[e] = c
        return QLaurent(new_terms)

    def __rmul__(self, other) -> "QLaurent":
        return self.__mul__(other)

    def is_monomial(self) -> bool:
        """True iff this is a single term c * q^n (c integer, n integer)."""
        return len(self.terms) <= 1

    def leading_term(self) -> tuple[int, int]:
        """Return (exponent, coefficient) of the highest-exponent term."""
        if not self.terms:
            return (0, 0)
        e = max(self.terms.keys())
        return (e, self.terms[e])

    def try_divide(self, other: "QLaurent") -> Optional["QLaurent"]:
        """Try to compute self / other exactly in Z[q, q^-1].

        Returns the quotient if exact, None otherwise.
        Exact division is possible when other is a single monomial c * q^n:
          self / (c * q^n) = (1/c) * q^(-n) * self  -- but 1/c must be integer.
        For multi-term other, this is generally not exact.
        """
        if other.is_zero():
            raise ZeroDivisionError("division by zero QLaurent")
        if other.is_monomial():
            if not other.terms:
                # other is 1 (zero terms means 0, but we checked is_zero above... actually is_zero returns True for empty)
                # Wait: is_zero returns True iff not self.terms. So empty terms = zero.
                # But is_monomial returns True for len <= 1, including 0. And we checked is_zero above.
                # So if we're here, other has exactly 1 term.
                return QLaurent(dict(self.terms))  # divide by 1
            e, c = next(iter(other.terms.items()))
            # self / (c * q^e) = (1/c) * q^(-e) * self
            # 1/c must be integer
            if c == 0:
                return None  # shouldn't happen (is_zero checked)
            new_terms = {}
            for e2, c2 in self.terms.items():
                if c2 % c != 0:
                    return None  # not exact
                new_terms[e2 - e] = c2 // c
            return QLaurent(new_terms)
        else:
            # Multi-term divisor: try polynomial long division
            # This is more complex; for now return None (can't divide)
            # A full implementation would use polynomial division
            return None

    def __truediv__(self, other):
        """Division. Returns self / other if exact, else raises ValueError.

        For the IR framework, we mostly divide by single q-powers, which is always exact.
        """
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        result = self.try_divide(other)
        if result is None:
            raise ValueError(f"Cannot divide {self} by {other} exactly in Z[q, q^-1]")
        return result

    def __rtruediv__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other.try_divide(self)

    # --- comparison / hashing ----------------------------------------------

    def __eq__(self, other) -> bool:
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self.terms == other.terms

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.terms.items())))

    # --- predicates ---------------------------------------------------------

    def is_zero(self) -> bool:
        """Return True iff this is the zero element."""
        return not self.terms

    def is_one(self) -> bool:
        """Return True iff this is the multiplicative identity ``1``."""
        return self.terms == {0: 1}

    # --- display ------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.terms:
            return "QLaurent(0)"
        parts = []
        for e in sorted(self.terms):
            c = self.terms[e]
            if e == 0:
                parts.append((c, "1"))
            elif e == 1:
                parts.append((c, "q"))
            else:
                parts.append((c, f"q^{e}"))
        # Render the first term, then alternating + / - for the rest.
        c0, s0 = parts[0]
        out = _format_signed(c0, s0, leading=True)
        for c, s in parts[1:]:
            out += _format_signed(c, s, leading=False)
        return f"QLaurent({out})"

    def __str__(self) -> str:
        return repr(self)


def _format_signed(coeff: int, sym: str, leading: bool) -> str:
    """Format a single ``coeff * sym`` term, with sign handling."""
    if coeff == 1:
        body = sym if sym != "1" else "1"
    elif coeff == -1:
        body = f"-{sym}" if sym != "1" else "-1"
    else:
        body = f"{coeff}*{sym}" if sym != "1" else f"{coeff}"
    if leading:
        return body
    # For non-leading terms, the sign is part of body; replace leading
    # '-' with ' - ' and prepend ' + ' otherwise.
    if body.startswith("-"):
        return " - " + body[1:]
    return " + " + body


# Convenience constants.
ZERO: QLaurent = QLaurent({})
ONE: QLaurent = QLaurent({0: 1})
Q: QLaurent = QLaurent({1: 1})
Q_INV: QLaurent = QLaurent({-1: 1})


def qpow(n: int) -> QLaurent:
    """Return ``q^n`` as a :class:`QLaurent` (``n`` any integer)."""
    return QLaurent.qpow(n)


def qint(n: int) -> QLaurent:
    """Return the integer ``n`` as a :class:`QLaurent`."""
    return QLaurent.from_int(n)


# ============================================================================
# Monomial: tuple of generator indices.
# ============================================================================
#
# A monomial is a tuple of generator indices, e.g. (K, E, K, E) for K*E*K*E.
# The empty tuple () represents the identity element (1).
# Generator indices are non-negative integers assigned by the Presentation.
# ============================================================================


class Monomial:
    """A monomial in the free algebra: a tuple of generator indices.

    The empty tuple is the multiplicative identity (``1``).
    """

    __slots__ = ("gens",)

    def __init__(self, gens: Tuple[int, ...] = ()) -> None:
        if not isinstance(gens, tuple):
            raise TypeError(
                f"Monomial: expected tuple, got {type(gens).__name__}"
            )
        self.gens: Tuple[int, ...] = gens

    # --- constructors -------------------------------------------------------

    @classmethod
    def empty(cls) -> "Monomial":
        """Return the empty monomial (the identity ``1``)."""
        return cls(())

    @classmethod
    def of(cls, *gens: int) -> "Monomial":
        """Construct a monomial from a variadic list of generator indices."""
        return cls(tuple(gens))

    # --- sequence-like operations ------------------------------------------

    def __len__(self) -> int:
        return len(self.gens)

    def __iter__(self) -> Iterator[int]:
        return iter(self.gens)

    def __getitem__(self, key) -> "Monomial":
        if isinstance(key, int):
            return Monomial((self.gens[key],))
        if isinstance(key, slice):
            return Monomial(self.gens[key])
        raise TypeError(f"Monomial: bad index type {type(key).__name__}")

    # --- algebra -----------------------------------------------------------

    def __add__(self, other: "Monomial") -> "Monomial":
        """Concatenation of monomials (the product in the free algebra)."""
        if not isinstance(other, Monomial):
            return NotImplemented
        return Monomial(self.gens + other.gens)

    def __mul__(self, other: "Monomial") -> "Monomial":
        return self.__add__(other)

    # --- comparison / hashing ----------------------------------------------

    def __eq__(self, other) -> bool:
        if not isinstance(other, Monomial):
            return NotImplemented
        return self.gens == other.gens

    def __hash__(self) -> int:
        return hash(self.gens)

    # --- predicates ---------------------------------------------------------

    def is_empty(self) -> bool:
        """Return True iff this is the empty (identity) monomial."""
        return not self.gens

    # --- ordering -----------------------------------------------------------

    def order_key(self) -> Tuple[int, Tuple[int, ...]]:
        """Length-lex ordering key: ``(length, gens)``.

        Shorter monomials come first; ties are broken lexicographically.
        """
        return (len(self.gens), self.gens)

    # --- display ------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Monomial({self.gens})"

    def __str__(self) -> str:
        if not self.gens:
            return "1"
        return "*".join(f"g{i}" for i in self.gens)


# ============================================================================
# Term: coefficient * monomial.
# ============================================================================


@dataclass(frozen=True)
class Term:
    """A single term: a coefficient times a monomial.

    Immutable and hashable (assuming :class:`QLaurent` and
    :class:`Monomial` are hashable, which they are).
    """

    coeff: QLaurent
    monomial: Monomial

    def __neg__(self) -> "Term":
        return Term(-self.coeff, self.monomial)

    def __mul__(self, scalar) -> "Term":
        if isinstance(scalar, QLaurent):
            return Term(self.coeff * scalar, self.monomial)
        if isinstance(scalar, int):
            return Term(self.coeff * qint(scalar), self.monomial)
        return NotImplemented

    def __rmul__(self, scalar) -> "Term":
        return self.__mul__(scalar)

    def is_zero(self) -> bool:
        """Return True iff this term has a zero coefficient."""
        return self.coeff.is_zero()


# ============================================================================
# Polynomial: sum of terms.
# ============================================================================
#
# A polynomial is a list of terms.  The list may contain duplicate
# monomials; call normalize() to combine like terms and drop zero
# coefficients.  All arithmetic operations (+, -, *) call normalize()
# on their result, so two polynomials are equal iff their normalized
# term lists are equal.
# ============================================================================


class Polynomial:
    """A polynomial in the free algebra: a sum of :class:`Term` objects.

    The list of terms is not necessarily normalized; call
    :meth:`normalize` (or rely on the arithmetic operators, which call it
    automatically) to combine like monomials and drop zero coefficients.
    """

    __slots__ = ("terms",)

    def __init__(self, terms: Iterable[Term] = ()) -> None:
        self.terms: list = list(terms)

    # --- constructors -------------------------------------------------------

    @classmethod
    def zero(cls) -> "Polynomial":
        """Return the zero polynomial."""
        return cls([])

    @classmethod
    def from_monomial(
        cls, m: Monomial, coeff: Optional[QLaurent] = None
    ) -> "Polynomial":
        """Return a polynomial with a single term ``coeff * m``."""
        if coeff is None:
            coeff = ONE
        return cls([Term(coeff, m)])

    @classmethod
    def from_term(cls, t: Term) -> "Polynomial":
        """Return a polynomial containing a single term."""
        return cls([t])

    # --- arithmetic ---------------------------------------------------------

    def __add__(self, other) -> "Polynomial":
        if not isinstance(other, Polynomial):
            return NotImplemented
        return Polynomial(self.terms + other.terms).normalize()

    def __neg__(self) -> "Polynomial":
        return Polynomial([-t for t in self.terms])

    def __sub__(self, other) -> "Polynomial":
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other) -> "Polynomial":
        if isinstance(other, (QLaurent, int)):
            scalar = other if isinstance(other, QLaurent) else qint(other)
            return Polynomial(
                [Term(t.coeff * scalar, t.monomial) for t in self.terms]
            ).normalize()
        if isinstance(other, Polynomial):
            new_terms = []
            for t1 in self.terms:
                for t2 in other.terms:
                    new_terms.append(
                        Term(t1.coeff * t2.coeff, t1.monomial + t2.monomial)
                    )
            return Polynomial(new_terms).normalize()
        if isinstance(other, Monomial):
            return Polynomial(
                [Term(t.coeff, t.monomial + other) for t in self.terms]
            ).normalize()
        if isinstance(other, Term):
            return self * Polynomial.from_term(other)
        return NotImplemented

    def __rmul__(self, other) -> "Polynomial":
        # Left multiplication by a scalar or monomial.
        if isinstance(other, (QLaurent, int)):
            return self.__mul__(other)
        if isinstance(other, Monomial):
            return Polynomial(
                [Term(t.coeff, other + t.monomial) for t in self.terms]
            ).normalize()
        if isinstance(other, Term):
            return Polynomial.from_term(other) * self
        return NotImplemented

    # --- comparison / hashing ----------------------------------------------

    def __eq__(self, other) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self.normalize().terms == other.normalize().terms

    def __hash__(self) -> int:
        n = self.normalize()
        return hash(tuple(n.terms))

    # --- predicates ---------------------------------------------------------

    def is_zero(self) -> bool:
        """Return True iff this polynomial is the zero polynomial."""
        return self.normalize().terms == []

    # --- normalization ------------------------------------------------------

    def normalize(self) -> "Polynomial":
        """Combine like monomials and drop zero coefficients.

        Returns a polynomial with:
        - No duplicate monomials.
        - No terms with zero coefficient.
        - Terms sorted by length-lex monomial order.
        """
        seen: dict = {}
        for t in self.terms:
            if t.coeff.is_zero():
                continue
            key = t.monomial
            if key in seen:
                seen[key] = seen[key] + t.coeff
            else:
                seen[key] = t.coeff
        result_terms = []
        for m, c in seen.items():
            if not c.is_zero():
                result_terms.append(Term(c, m))
        result_terms.sort(key=lambda t: t.monomial.order_key())
        return Polynomial(result_terms)

    # --- display ------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.terms:
            return "Polynomial(0)"
        return "Polynomial(" + " + ".join(repr(t) for t in self.terms) + ")"

    def __str__(self) -> str:
        norm = self.normalize()
        if not norm.terms:
            return "0"
        parts = []
        for t in norm.terms:
            c, m = t.coeff, t.monomial
            if c.is_one():
                parts.append(("+", str(m)))
            elif (-c).is_one():
                parts.append(("-", str(m)))
            else:
                parts.append(("+", f"({c})*{m}"))
        sign, body = parts[0]
        out = body if sign == "+" else f"-{body}"
        for sign, body in parts[1:]:
            out += f" {sign} {body}"
        return out


# ============================================================================
# RewriteRule: LHS monomial -> RHS polynomial.
# ============================================================================
#
# The LHS must be a non-empty monomial (the empty monomial would match
# everywhere and lead to infinite loops).  The RHS is a polynomial; if it
# is the zero polynomial, the rule deletes occurrences of the LHS.
# ============================================================================


class RewriteRule:
    """A rewrite rule ``LHS -> RHS`` where LHS is a single monomial."""

    __slots__ = ("lhs", "rhs")

    def __init__(self, lhs: Monomial, rhs: Polynomial) -> None:
        if not isinstance(lhs, Monomial):
            raise TypeError(
                f"RewriteRule: lhs must be a Monomial, got {type(lhs).__name__}"
            )
        if not isinstance(rhs, Polynomial):
            raise TypeError(
                f"RewriteRule: rhs must be a Polynomial, got {type(rhs).__name__}"
            )
        if len(lhs) == 0:
            raise ValueError(
                "RewriteRule: lhs must be a non-empty monomial "
                "(the empty monomial matches everywhere)"
            )
        self.lhs: Monomial = lhs
        self.rhs: Polynomial = rhs.normalize()

    # --- matching -----------------------------------------------------------

    def matches(self, monomial: Monomial, position: int) -> bool:
        """Return True iff the rule matches ``monomial`` at ``position``.

        A match means ``monomial[position : position + len(lhs)] == lhs``,
        considered as tuples of generator indices.
        """
        n = len(self.lhs)
        if position < 0 or position + n > len(monomial):
            return False
        return monomial.gens[position : position + n] == self.lhs.gens

    def apply(self, monomial: Monomial, position: int) -> Polynomial:
        """Apply the rule at ``position`` and return the resulting polynomial.

        Returns ``prefix * RHS * suffix`` where
        ``prefix = monomial[:position]`` and
        ``suffix = monomial[position + len(lhs):]``.

        Raises :class:`ValueError` if the rule does not match at
        ``position``.
        """
        if not self.matches(monomial, position):
            raise ValueError(
                f"RewriteRule.apply: rule does not match at position {position} "
                f"of {monomial}"
            )
        prefix = Monomial(monomial.gens[:position])
        suffix = Monomial(monomial.gens[position + len(self.lhs) :])
        new_terms = []
        for t in self.rhs.terms:
            new_m = prefix + t.monomial + suffix
            new_terms.append(Term(t.coeff, new_m))
        return Polynomial(new_terms).normalize()

    # --- display ------------------------------------------------------------

    def __repr__(self) -> str:
        return f"RewriteRule({self.lhs!r} -> {self.rhs!r})"

    def __str__(self) -> str:
        return f"{self.lhs} -> {self.rhs}"


# ============================================================================
# Presentation: generators + rewrite rules, with a parser.
# ============================================================================
#
# A presentation is a list of generator names (strings) and a list of
# rewrite rules.  Generator names are mapped to integer indices by their
# position in the list, so e.g. generators ['x', 'y'] gives x -> 0, y -> 1.
# The parser uses this mapping to convert a string like "q^2 * x * y - y*x"
# into a Polynomial.
# ============================================================================


class Presentation:
    """A presentation of a noncommutative algebra.

    Parameters
    ----------
    generators : list of str
        The generator names.  Position in the list determines the index
        (e.g. ``generators[0]`` has index 0).
    rules : list of RewriteRule
        The rewrite rules.  Order matters for the :class:`NormalFormReducer`:
        when multiple rules match at the same leftmost position, the
        first matching rule in this list is applied.
    """

    __slots__ = ("generators", "rules", "_name_to_idx")

    def __init__(
        self, generators: list, rules: list
    ) -> None:
        self.generators: list = list(generators)
        self.rules: list = list(rules)
        # Check for duplicate generator names.
        if len(set(self.generators)) != len(self.generators):
            raise ValueError(
                f"Presentation: duplicate generator names in {self.generators}"
            )
        self._name_to_idx = {name: i for i, name in enumerate(self.generators)}

    def parse(self, s: str) -> Polynomial:
        """Parse a string like ``"q^2 * K * E - E * K"`` into a Polynomial.

        Grammar (whitespace-insensitive)::

            polynomial := ['+'|'-'] term (('+'|'-') term)*
            term       := factor ('*' factor)*
            factor     := int
                        | 'q' ['^' ['-'] int]
                        | generator_name
                        | '(' polynomial ')'

        Integer and ``q`` factors multiply to give the term's coefficient;
        generator factors concatenate to give the term's monomial.
        Parenthesized sub-polynomials may appear as factors.

        Raises :class:`ValueError` on a parse error.
        """
        return _parse_polynomial(s, self._name_to_idx)

    def __repr__(self) -> str:
        return (
            f"Presentation(generators={self.generators!r}, "
            f"rules={self.rules!r})"
        )

    def __str__(self) -> str:
        gens = ", ".join(self.generators)
        rules = "; ".join(str(r) for r in self.rules)
        return f"<{gens} | {rules}>"


# ----------------------------------------------------------------------------
# Parser internals.
# ----------------------------------------------------------------------------


def _tokenize(s: str) -> list:
    """Tokenize a polynomial string into a list of tokens.

    Each token is a tuple.  Token kinds:
        ('int', n)        -- integer literal
        ('name', str)     -- identifier (generator name or 'q')
        ('+',) ('-',) ('*',) ('^',) ('(',) (')',)
    """
    tokens = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "+-*^()":
            tokens.append((c,))
            i += 1
        elif c.isdigit():
            j = i
            while j < n and s[j].isdigit():
                j += 1
            tokens.append(("int", int(s[i:j])))
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(("name", s[i:j]))
            i = j
        else:
            raise ValueError(
                f"_tokenize: unexpected character {c!r} at position {i} in {s!r}"
            )
    return tokens


def _parse_polynomial(s: str, name_to_idx: dict) -> Polynomial:
    """Parse a polynomial string with a recursive-descent parser."""
    tokens = _tokenize(s)
    pos = [0]  # mutable container for current position

    def peek():
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def advance():
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def expect(kind):
        t = peek()
        if t is None or t[0] != kind:
            raise ValueError(
                f"parse: expected {kind!r} at token {pos[0]}, got {t}"
            )
        return advance()

    def parse_factor():
        """Parse a factor and return a Polynomial (a single term, or the
        contents of a parenthesized sub-polynomial)."""
        t = peek()
        if t is None:
            raise ValueError("parse: unexpected end of input parsing factor")
        if t[0] == "(":
            advance()
            inner = parse_polynomial_inner()
            expect(")")
            return inner
        if t[0] == "int":
            advance()
            return Polynomial([Term(qint(t[1]), Monomial.empty())])
        if t[0] == "name":
            name = t[1]
            if name == "q":
                advance()
                if peek() is not None and peek()[0] == "^":
                    advance()
                    sign = 1
                    if peek() is not None and peek()[0] == "-":
                        sign = -1
                        advance()
                    nt = peek()
                    if nt is None or nt[0] != "int":
                        raise ValueError(
                            "parse: expected integer after '^' (with optional '-' sign)"
                        )
                    advance()
                    return Polynomial(
                        [Term(qpow(sign * nt[1]), Monomial.empty())]
                    )
                return Polynomial([Term(qpow(1), Monomial.empty())])
            if name in name_to_idx:
                advance()
                return Polynomial(
                    [Term(ONE, Monomial((name_to_idx[name],)))]
                )
            raise ValueError(
                f"parse: unknown name {name!r} -- not 'q' and not a generator"
            )
        raise ValueError(f"parse: unexpected token {t} parsing factor")

    def parse_term():
        """Parse a term: factor ('*' factor)*, returning a Polynomial."""
        poly = parse_factor()
        while peek() is not None and peek()[0] == "*":
            advance()
            poly = poly * parse_factor()
        return poly

    def parse_polynomial_inner():
        """Parse a polynomial: optional sign, then term (('+'|'-') term)*."""
        sign = 1
        if peek() is not None and peek()[0] in ("+", "-"):
            t = advance()
            if t[0] == "-":
                sign = -1
        poly = parse_term()
        if sign == -1:
            poly = -poly
        while peek() is not None and peek()[0] in ("+", "-"):
            t = advance()
            sub = parse_term()
            if t[0] == "-":
                poly = poly - sub
            else:
                poly = poly + sub
        return poly

    result = parse_polynomial_inner()
    if pos[0] != len(tokens):
        raise ValueError(
            f"parse: trailing tokens at position {pos[0]}: {tokens[pos[0]:]}"
        )
    return result.normalize()


# ============================================================================
# NormalFormReducer: reduce polynomials to normal form via a rewrite system.
# ============================================================================
#
# Strategy: at each step, find the leftmost match of any rule in the
# current monomial and apply it.  Recurse on each term of the result.
# The recursion terminates iff the rewrite system is terminating (no
# infinite reduction sequences).  For a confluent + terminating system,
# the result is the unique normal form.
#
# Warning: this reducer does NOT detect infinite loops.  If the rewrite
# system is non-terminating, the reducer may recurse forever.  Knuth-Bendix
# completion (W0-1b) ensures termination by construction.
# ============================================================================


class NormalFormReducer:
    """Reduce polynomials to normal form via a rewrite system.

    Parameters
    ----------
    presentation : Presentation
        A presentation with a (confluent, terminating) rewrite system.

    Notes
    -----
    The reducer uses a leftmost-match strategy: at each step it scans the
    monomial from left to right and applies the first matching rule at the
    first position where any rule matches.  For a confluent rewrite system
    this yields the unique normal form regardless of strategy.

    The reducer recurses on the result of each rule application.  For a
    terminating system this terminates; for a non-terminating system the
    reducer may recurse forever.
    """

    __slots__ = ("presentation",)

    def __init__(self, presentation: Presentation) -> None:
        self.presentation = presentation

    def find_match(
        self, monomial: Monomial
    ) -> Optional[Tuple[int, RewriteRule]]:
        """Find the leftmost match of any rule in ``monomial``.

        Returns ``(position, rule)`` or ``None`` if no rule matches.

        Among multiple rules matching at the same position, the first rule
        in ``presentation.rules`` is returned.
        """
        n = len(monomial)
        for position in range(n):
            for rule in self.presentation.rules:
                if rule.matches(monomial, position):
                    return (position, rule)
        return None

    def normal_form(self, monomial: Monomial) -> Polynomial:
        """Reduce a single monomial to normal form.

        Returns the polynomial obtained by repeatedly applying rules
        until no rule matches.  The result is a polynomial whose terms
        are all in normal form (no rule matches any of their monomials).
        """
        match = self.find_match(monomial)
        if match is None:
            return Polynomial([Term(ONE, monomial)])
        position, rule = match
        result = rule.apply(monomial, position)
        # Recursively reduce each term in the result, scaling by the
        # original term's coefficient.
        reduced_terms = []
        for t in result.terms:
            sub = self.normal_form(t.monomial)
            for st in sub.terms:
                reduced_terms.append(Term(st.coeff * t.coeff, st.monomial))
        return Polynomial(reduced_terms).normalize()

    def reduce(self, poly: Polynomial) -> Polynomial:
        """Reduce a polynomial to normal form.

        Each term's monomial is reduced to normal form (a polynomial),
        scaled by the original coefficient, and the results are summed
        and normalized.
        """
        reduced_terms = []
        for t in poly.terms:
            sub = self.normal_form(t.monomial)
            for st in sub.terms:
                reduced_terms.append(Term(st.coeff * t.coeff, st.monomial))
        return Polynomial(reduced_terms).normalize()


__all__ = [
    # Ring
    "QLaurent",
    "ZERO",
    "ONE",
    "Q",
    "Q_INV",
    "qpow",
    "qint",
    # Free algebra
    "Monomial",
    "Term",
    "Polynomial",
    # Rewriting
    "RewriteRule",
    "Presentation",
    "NormalFormReducer",
]
