"""AST/IR framework for noncommutative algebras (hopf-decoherence project).

This package provides the foundation for computing Hochschild cohomology
via the Anick resolution.  Built up across Wave 0 of the AST/IR research
programme:

- **W0-1a** (this revision): ``ir.parser`` -- QLaurent coefficient ring,
  Monomial / Term / Polynomial classes, RewriteRule, Presentation with a
  string parser, and NormalFormReducer.  Targeted at confluent rewrite
  systems over ``Z[q, q^{-1}]``.
- **W0-1b** (next): Knuth-Bendix completion on top of this module.
"""

__version__ = "0.1.0-w0-1a"
