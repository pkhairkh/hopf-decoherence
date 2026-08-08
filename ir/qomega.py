"""Specialized coefficient ring Q(omega) for u_q(sl_2) at ell = 3.

This module provides :class:`QOmega3`, an element of the field
``Q(omega)`` where ``omega = e^{2*pi*i/3}`` is a primitive 3rd root of
unity.  Elements are stored exactly as ``a + b*omega`` with
``a, b in fractions.Fraction``, using the relations::

    omega^2 + omega + 1 = 0      (so omega^2 = -1 - omega)
    omega^3 = 1                   (so omega^k = omega^(k mod 3))

Q(omega) is the natural coefficient field for the small quantum group
``u_q(sl_2)`` at ``ell = 3``: it is the smallest field containing both
``Z[q, q^{-1}]`` (the generic coefficient ring of the IR framework) and
the inverse ``1/(q - q^{-1})`` that appears in the commutator relation
``[E, F] = (K - K^{-1})/(q - q^{-1})``.

Design
------
:class:`QOmega3` is a subclass of :class:`ir.parser.QLaurent`.  This lets
it pass the existing ``isinstance(x, QLaurent)`` checks in
``Polynomial.__mul__`` / ``Term.__mul__`` and the
``NormalFormReducer``.  The internal storage is the same dict
``{exponent: coefficient}`` (with ``exponent in {0, 1}`` after
reduction); the only differences from :class:`QLaurent` are:

- Coefficients are :class:`fractions.Fraction` (not just ``int``).
- Every arithmetic operation reduces the result modulo
  ``(q^3 - 1, q^2 + q + 1)`` so the stored form is always the canonical
  ``a + b*omega``.
- :meth:`try_divide` performs exact division in the field Q(omega)
  (always succeeds for nonzero divisors).

Because :class:`QOmega3` is a subclass, mixed-type arithmetic
(``QLaurent + QOmega3`` etc.) works via Python's "subclass reflected
operator first" rule: ``QOmega3.__radd__`` / ``__rmul__`` etc. are tried
before the corresponding ``QLaurent`` operators, so the result is always
a :class:`QOmega3` (correctly reduced).  This means the existing
``NormalFormReducer`` works with QOmega3 coefficients without
modification.
"""
from __future__ import annotations

import cmath
from fractions import Fraction
from typing import Optional

from .parser import QLaurent


# ----------------------------------------------------------------------------
# Module-level Fraction constants
# ----------------------------------------------------------------------------

FRACT_ZERO: Fraction = Fraction(0)
FRACT_ONE: Fraction = Fraction(1)
FRACT_NEG_ONE: Fraction = Fraction(-1)


def _to_fraction(x) -> Fraction:
    """Coerce ``x`` to a :class:`Fraction`.

    Accepts ``int``, ``Fraction``, and ``float`` (the last via
    ``Fraction(x).limit_denominator(10**12)``).
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(x).limit_denominator(10**12)
    raise TypeError(
        f"QOmega3: cannot coerce {type(x).__name__} to Fraction"
    )


# ----------------------------------------------------------------------------
# QOmega3: element of Q(omega)
# ----------------------------------------------------------------------------


class QOmega3(QLaurent):
    """Element of the field ``Q(omega)`` with ``omega`` a primitive 3rd root of 1.

    Stored as a :class:`QLaurent` dict ``{0: a, 1: b}`` representing
    ``a + b*omega`` with ``a, b in Fraction``.  The dict is always kept
    in canonical form: at most two keys (0 and 1), no zero coefficients.
    """

    # Inherit QLaurent's __slots__ = ("terms",); no new slots needed.
    __slots__ = ()

    def __init__(self, terms=None) -> None:
        if terms is None:
            self.terms: dict = {}
        elif isinstance(terms, dict):
            self.terms = {
                e: _to_fraction(c)
                for e, c in terms.items()
                if _to_fraction(c) != FRACT_ZERO
            }
        elif isinstance(terms, QLaurent):
            # Copy from another QLaurent (or QOmega3) -- treat q as omega.
            self.terms = {
                e: _to_fraction(c)
                for e, c in terms.terms.items()
                if _to_fraction(c) != FRACT_ZERO
            }
        else:
            raise TypeError(
                f"QOmega3: expected dict or QLaurent, got {type(terms).__name__}"
            )
        self._reduce_in_place()

    # --- internal reduction ------------------------------------------------

    def _reduce_in_place(self) -> None:
        """Reduce ``self.terms`` modulo ``(q^3 - 1, q^2 + q + 1)``.

        After this call, ``self.terms`` has keys in ``{0, 1}`` and no
        zero coefficients.
        """
        if not self.terms:
            return
        # Step 1: collapse exponents mod 3 (since omega^3 = 1).
        collapsed: dict = {}
        for e, c in self.terms.items():
            e_mod = e % 3
            collapsed[e_mod] = collapsed.get(e_mod, FRACT_ZERO) + c
        # Step 2: replace omega^2 = -1 - omega.
        if 2 in collapsed:
            c2 = collapsed.pop(2)
            if c2 != FRACT_ZERO:
                collapsed[0] = collapsed.get(0, FRACT_ZERO) - c2
                collapsed[1] = collapsed.get(1, FRACT_ZERO) - c2
        # Step 3: drop zero coefficients.
        self.terms = {
            e: c for e, c in collapsed.items() if c != FRACT_ZERO
        }

    # --- constructors ------------------------------------------------------

    @classmethod
    def from_int(cls, n: int) -> "QOmega3":
        """Return the integer ``n`` as a :class:`QOmega3`."""
        if n == 0:
            return cls({})
        return cls({0: Fraction(n)})

    @classmethod
    def from_fraction(cls, a: Fraction) -> "QOmega3":
        """Return the rational ``a`` as a :class:`QOmega3`."""
        if a == 0:
            return cls({})
        return cls({0: a})

    @classmethod
    def from_qpow(cls, e: int) -> "QOmega3":
        """Return ``omega^e`` as a :class:`QOmega3` (``e`` any integer).

        ``omega^0 = 1``, ``omega^1 = omega``, ``omega^2 = -1 - omega``,
        ``omega^3 = 1``, ...
        """
        e_mod = e % 3
        if e_mod == 0:
            return cls({0: FRACT_ONE})
        if e_mod == 1:
            return cls({1: FRACT_ONE})
        # e_mod == 2: omega^2 = -1 - omega
        return cls({0: FRACT_NEG_ONE, 1: FRACT_NEG_ONE})

    @classmethod
    def from_complex(cls, z: complex) -> "QOmega3":
        """Construct a :class:`QOmega3` from a complex number.

        ``z`` must lie in ``Q(omega)``: write ``z = a + b*omega`` with
        ``omega = -1/2 + i*sqrt(3)/2``.  Then
        ``b = 2*Im(z)/sqrt(3)`` and ``a = Re(z) + b/2``.  The floats are
        rounded to nearby Fractions; a :class:`ValueError` is raised if
        the result does not round-trip.
        """
        if abs(z.imag) < 1e-12:
            return cls({0: _to_fraction(z.real)})
        sqrt3_over_2 = (3.0 ** 0.5) / 2.0
        b_float = z.imag / sqrt3_over_2
        a_float = z.real + b_float / 2.0
        a = Fraction(a_float).limit_denominator(10**12)
        b = Fraction(b_float).limit_denominator(10**12)
        result = cls({0: a, 1: b})
        if abs(result.to_complex() - z) > 1e-9:
            raise ValueError(
                f"QOmega3.from_complex: {z} is not in Q(omega) "
                f"(reconstructed {result.to_complex()})"
            )
        return result

    def to_complex(self) -> complex:
        """Convert to a Python ``complex`` number."""
        omega = cmath.exp(2j * cmath.pi / 3)
        a = self.terms.get(0, FRACT_ZERO)
        b = self.terms.get(1, FRACT_ZERO)
        return float(a) + float(b) * omega

    # --- coercion ----------------------------------------------------------

    @staticmethod
    def _coerce(other) -> Optional["QOmega3"]:
        """Coerce ``other`` to a :class:`QOmega3`, or return ``NotImplemented``."""
        if isinstance(other, QOmega3):
            return other
        if isinstance(other, QLaurent):
            # Promote a generic QLaurent by treating q as omega.
            return QOmega3(dict(other.terms))
        if isinstance(other, int):
            return QOmega3.from_int(other)
        if isinstance(other, Fraction):
            return QOmega3.from_fraction(other)
        return NotImplemented

    # --- arithmetic --------------------------------------------------------

    def __add__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        new = dict(self.terms)
        for e, c in other.terms.items():
            new[e] = new.get(e, FRACT_ZERO) + c
        return QOmega3(new)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self) -> "QOmega3":
        return QOmega3({e: -c for e, c in self.terms.items()})

    def __sub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        return other + (-self)

    def __mul__(self, other):
        other = self._coerce(other)
        if other is NotImplemented:
            return NotImplemented
        # Multiply as Laurent polynomials, then reduce.
        new: dict = {}
        for e1, c1 in self.terms.items():
            for e2, c2 in other.terms.items():
                e = e1 + e2
                new[e] = new.get(e, FRACT_ZERO) + c1 * c2
        return QOmega3(new)

    def __rmul__(self, other):
        return self.__mul__(other)

    # --- predicates --------------------------------------------------------

    def is_zero(self) -> bool:
        return not self.terms

    def is_one(self) -> bool:
        return self.terms == {0: FRACT_ONE}

    def is_monomial(self) -> bool:
        """True iff this is a single Fraction times a power of omega."""
        return len(self.terms) <= 1

    # --- division (Q(omega) is a field) ------------------------------------

    def inverse(self) -> "QOmega3":
        """Return the multiplicative inverse of a nonzero element.

        For ``z = a + b*omega`` the inverse is
        ``(a + b*omega^2) / ((a + b*omega)(a + b*omega^2))``.
        Using ``omega + omega^2 = -1`` and ``omega^3 = 1`` the
        denominator simplifies to ``a^2 - a*b + b^2``.
        """
        if self.is_zero():
            raise ZeroDivisionError("QOmega3.inverse: zero has no inverse")
        a = self.terms.get(0, FRACT_ZERO)
        b = self.terms.get(1, FRACT_ZERO)
        denom = a * a - a * b + b * b
        if denom == 0:
            raise ZeroDivisionError(
                f"QOmega3.inverse: denominator vanished for {self}"
            )
        # (a + b*omega)^{-1} = (a + b*omega^2) / denom
        #                   = (a - b - b*omega) / denom   (using omega^2 = -1-omega)
        #                   = ((a-b) + (-b)*omega) / denom
        return QOmega3({
            0: (a - b) / denom,
            1: (-b) / denom,
        })

    def try_divide(self, other):
        """Exact division ``self / other`` in Q(omega).

        Returns the quotient (always exact for nonzero ``other``),
        or raises :class:`ZeroDivisionError` if ``other`` is zero.
        """
        other = self._coerce(other) if not isinstance(other, QOmega3) else other
        if other is NotImplemented:
            return None
        if other.is_zero():
            raise ZeroDivisionError(
                "QOmega3.try_divide: division by zero"
            )
        return self * other.inverse()

    def __truediv__(self, other):
        other = self._coerce(other) if not isinstance(other, QOmega3) else other
        if other is NotImplemented:
            return NotImplemented
        return self.try_divide(other)

    def __rtruediv__(self, other):
        other = self._coerce(other) if not isinstance(other, QOmega3) else other
        if other is NotImplemented:
            return NotImplemented
        return other.try_divide(self)

    # --- comparison / hashing ----------------------------------------------

    def __eq__(self, other) -> bool:
        # Subclass priority: if other is a QLaurent, promote it.
        if isinstance(other, QOmega3):
            return self.terms == other.terms
        if isinstance(other, QLaurent):
            return self.terms == QOmega3(dict(other.terms)).terms
        if isinstance(other, int):
            return self == QOmega3.from_int(other)
        if isinstance(other, Fraction):
            return self == QOmega3.from_fraction(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.terms.items())))

    # --- display ------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.terms:
            return "QOmega3(0)"
        parts = []
        for e in sorted(self.terms):
            c = self.terms[e]
            if e == 0:
                parts.append((c, "1"))
            elif e == 1:
                parts.append((c, "w"))
            else:
                parts.append((c, f"w^{e}"))
        c0, s0 = parts[0]
        out = _format_qw(c0, s0, leading=True)
        for c, s in parts[1:]:
            out += _format_qw(c, s, leading=False)
        return f"QOmega3({out})"

    def __str__(self) -> str:
        return repr(self)


def _format_qw(coeff: Fraction, sym: str, leading: bool) -> str:
    """Format a single ``coeff * sym`` term, with sign handling."""
    if coeff == 1:
        body = sym if sym != "1" else "1"
    elif coeff == -1:
        body = f"-{sym}" if sym != "1" else "-1"
    else:
        body = f"{coeff}*{sym}" if sym != "1" else f"{coeff}"
    if leading:
        return body
    if body.startswith("-"):
        return " - " + body[1:]
    return " + " + body


# ----------------------------------------------------------------------------
# Module-level QOmega3 constants
# ----------------------------------------------------------------------------

OMEGA3_ZERO: QOmega3 = QOmega3({})
OMEGA3_ONE: QOmega3 = QOmega3({0: FRACT_ONE})
OMEGA: QOmega3 = QOmega3({1: FRACT_ONE})  # omega
# omega^2 = -1 - omega
OMEGA2: QOmega3 = QOmega3({0: FRACT_NEG_ONE, 1: FRACT_NEG_ONE})
# q - q^{-1} = omega - omega^2 = omega - (-1-omega) = 1 + 2*omega
Q_MINUS_Q_INV: QOmega3 = QOmega3({0: FRACT_ONE, 1: Fraction(2)})
# 1/(q - q^{-1}) = inverse of (1 + 2*omega).
# Using the formula: (a + b*omega)^{-1} = ((a-b) - b*omega) / (a^2 - ab + b^2)
# with a=1, b=2: denom = 1 - 2 + 4 = 3; inv = (1-2 - 2*omega)/3 = (-1 - 2*omega)/3
Q_MINUS_Q_INV_INV: QOmega3 = QOmega3({
    0: Fraction(-1, 3),
    1: Fraction(-2, 3),
})


__all__ = [
    "QOmega3",
    "FRACT_ZERO",
    "FRACT_ONE",
    "FRACT_NEG_ONE",
    "OMEGA3_ZERO",
    "OMEGA3_ONE",
    "OMEGA",
    "OMEGA2",
    "Q_MINUS_Q_INV",
    "Q_MINUS_Q_INV_INV",
]
