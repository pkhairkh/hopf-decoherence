"""AST/IR framework for noncommutative algebras (hopf-decoherence project).

This package provides the foundation for computing Hochschild cohomology
via the Anick resolution.  Built up across Waves 0 and 1 of the AST/IR
research programme:

- **W0-1a**: ``ir.parser`` -- QLaurent coefficient ring,
  Monomial / Term / Polynomial classes, RewriteRule, Presentation with a
  string parser, and NormalFormReducer.  Targeted at confluent rewrite
  systems over ``Z[q, q^{-1}]``.
- **W0-1b**: ``ir.groebner`` -- Knuth-Bendix completion + Anick
  resolution generator enumeration on top of ``ir.parser``.
- **W1-1a-IR**: ``ir.qomega`` (specialized coefficient field
  ``Q(omega)`` at ``ell = 3``) and ``ir.uq_sl2`` (``u_q(sl_2)``
  presentation at ``ell = 3`` + bar-complex-on-PBW ``dim HH^2``
  computation).  Validates the framework: ``dim HH^2(u_q(sl_2), C) = 3``
  matches the conjecture ``C(n+1, 2) + 2|Phi^+|`` at rank one.
"""

__version__ = "0.2.0-w1-1a-ir"
