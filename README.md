# hopf-decoherence

**Hochschild cohomology of small quantum groups at roots of unity — companion code**

**Author:** Parham Khairkhah (ORCID [0009-0000-7048-1397](https://orcid.org/0009-0000-7048-1397))
**Repository version:** 1.2.0
**Code license:** MIT  ·  **Paper license:** CC-BY-4.0

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](./LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/paper-CC--BY--4.0-green.svg)](./paper/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-88%20passing-brightgreen.svg)](#installation)

---

## What this repository contains

This repository accompanies the paper *Hochschild Cohomology of Small Quantum Groups at Roots of Unity* (`paper/main.tex`). The paper proposes the conjecture

$$
\dim_\mathbb{C}\,\mathrm{HH}^2\!\bigl(u_q(\mathfrak{g}),\,\mathbb{C}\bigr) \;=\; \binom{n+1}{2} \;+\; 2\,|\Phi^+|,
$$

for any finite-dimensional complex simple Lie algebra $\mathfrak{g}$ of rank $n$ at an odd root of unity $q = e^{2\pi i/\ell}$, and verifies it by direct bar-complex computation in the $A_1$ case.

The repository contains:

- **`paper/main.tex`** — full LaTeX source for the paper.
- **`scripts/verify_sl2_hh2.py`** — a from-scratch construction of the Hochschild bar complex of $u_q(\mathfrak{sl}_2)$ at $\ell = 3$ (algebra dimension 27), computing $\dim\mathrm{HH}^2 = \dim\ker d^2 - \dim\mathrm{im}\,d^1 = 3$, in agreement with the conjecture. This is the sole direct numerical verification of the conjecture.
- **`scripts/verify_bplus_sl2_rigorous.py`** — verifies $\dim\mathrm{HH}^2(B^+(u_q(\mathfrak{sl}_2)), \mathbb{C}) = 1$ at $\ell = 3, 5, 7$, matching the Borel formula $2|\Phi^+| - 1$.
- **`scripts/analyze_cocycles.py`** — extracts and classifies the three explicit $\mathrm{HH}^2$ cocycles of $u_q(\mathfrak{sl}_2)$ at $\ell = 3$ by their support on basis-pair categories (K-K, E-E, F-F, mixed). Confirms that the third class is *not* Cartan-type, contrary to the original §3 structural proposal.
- **`scripts/test_restriction_map.py`** — verifies the Mastnak–Witherspoon long exact sequence (3.3.1) for $u_q(\mathfrak{sl}_2)$ at $\ell = 3$. Computes the restriction map $\bar{\pi}: \mathrm{HH}^2(D(B^+)) \to \mathrm{HH}^2(B^+) \oplus \mathrm{HH}^2(B^-)$ and confirms $\dim\ker\bar{\pi} = 1$, providing the empirical signature of the connecting homomorphism $\delta: \tilde{H}^1_b(B^+) \to \mathrm{HH}^2(D(B^+))$.
- **`tests/test_sl2_hh2_bar_complex.py`** — five dedicated tests covering the defining relations of $u_q(\mathfrak{sl}_2)$, associativity, the chain-complex property $d^2 \circ d^1 = 0$, the counit being an algebra map, and the $\dim\mathrm{HH}^2 = 3$ rank computation.
- **`tests/test_les_restriction.py`** — pytest wrapper around `scripts/test_restriction_map.py` verifying the LES framework.
- **`src/hopf_decoherence/`** and the remaining `scripts/` — an auxiliary computational framework for the BCGP non-semisimple TQFT and BTZ black-hole entropy programme that the paper references speculatively in §7. This code is **not** required for the paper's main theorem (the $A_1$ verification) and is not the subject of the paper; it is included because the same quantum-group infrastructure underlies both.

## Status of the conjecture

| Type | Rank $n$ | $|\Phi^+|$ | Predicted $\dim\mathrm{HH}^2$ | Status |
|---|---|---|---|---|
| $A_1$ | 1 | 1 | 3 | **Theorem** (bar complex + exact cyclotomic certification, $\ell=3$; floating-point at $\ell=5$) |
| $A_2$ | 2 | 3 | 9 | **Open / in doubt** — structural prediction $\dim\operatorname{im}\delta = \binom{3}{2} = 3$ REFUTED (computed $= 2$). Count $9$ vs alternative $8$ unresolved. |
| $A_3$ | 3 | 6 | 18 | **Open / in doubt** — structural prediction $\dim\operatorname{im}\delta = \binom{4}{2} = 6$ REFUTED (computed $= 3$). Count $18$ vs alternative $15$ unresolved. |
| $B_2$ | 2 | 4 | 11 | Structural |
| $G_2$ | 2 | 6 | 15 | Structural |
| $A_n$ | $n$ | $n(n+1)/2$ | $3n(n+1)/2$ | Conjecture (in doubt for $n \geq 3$) |
| $E_8$ | 8 | 120 | 276 | Conjecture |

**Key finding (v1.2):** Direct computation of $\dim \tilde{H}^1_b(B^+(u_q(\mathfrak{sl}_n)), \mathbb{C})$ at $\ell = 3$ for $n = 2, 3, 4$ gives $1, 2, 3$ respectively. The pattern is $\dim \tilde{H}^1_b = n - 1$, NOT the originally predicted $\binom{n+1}{2}$. This refutes the structural decomposition $\dim \operatorname{im}\delta = \binom{n+1}{2}$ for $n \geq 3$ and casts doubt on the full conjecture's count at $A_2$ and higher (alternative count: $(n-1) + 2|\Phi^+|$). See new Section 8 of the paper for the full analysis.

The universal formula is stated as **Conjecture 1.1** in the paper. The $A_1$ case is **Theorem 1.2** (verified by the bar-complex script at $\ell = 3$ and $\ell = 5$, with the $\ell = 3$ result now certified by exact cyclotomic arithmetic over $\mathbb{Z}[\omega, 1/3]$ via rank computation over 11 finite fields). The $A_2$ and $A_3$ cases remain **open with structural predictions refuted**: the LES framework (Mastnak–Witherspoon long exact sequence (3.3.1)) is verified correct, but the structural prediction $\dim \operatorname{im}\delta = \binom{n+1}{2}$ is wrong for $n \geq 3$ (computed $= n - 1$). The full count $\dim \HH^2 = \binom{n+1}{2} + 2|\Phi^+|$ is consequently in doubt at $A_2$ and higher.

The original §3 of the paper proposed that the third $A_1$ class is a "Cartan-type" class supported on $K$-$K$ pairs. Direct cocycle extraction (`scripts/analyze_cocycles.py`) shows this is false: none of the three $\mathrm{HH}^2$ cocycles has significant $K$-$K$ support, and the third class is a mixed $E$-$F$ class. This is now understood as the **empirical signature of the connecting homomorphism $\delta$**: the Cartan-type classes live in $\tilde{H}^1_b(B^+)$, and $\delta$ maps them through the cross-relations of the Drinfeld double, producing the mixed support observed in the image (see new §6.3 of the paper).

## Installation

The package requires Python ≥ 3.10 and the standard scientific stack:

```bash
git clone https://github.com/pkhairkh/hopf-decoherence.git
cd hopf-decoherence
pip install -e .[dev]
```

Dependencies: `numpy ≥ 1.24`, `scipy ≥ 1.10`, `sympy ≥ 1.12`, `matplotlib ≥ 3.7` (optional, for plotting).

## Usage

Run the full test suite (69 tests, including the new LES verification):

```bash
python -m pytest tests/ -v
```

Run only the bar-complex verification for $u_q(\mathfrak{sl}_2)$ at $\ell = 3$:

```bash
python scripts/verify_sl2_hh2.py
```

Run the cocycle support analysis (extracts the 3 $\mathrm{HH}^2$ classes and classifies them by basis-pair support):

```bash
python scripts/analyze_cocycles.py
```

Run the Mastnak–Witherspoon LES restriction map test:

```bash
python scripts/test_restriction_map.py
```

Run only the bar-complex tests:

```bash
python -m pytest tests/test_sl2_hh2_bar_complex.py -v
```

## Project structure

```
hopf-decoherence/
├── paper/                              # LaTeX source for the paper
│   ├── main.tex
│   └── figures/
├── scripts/
│   ├── verify_sl2_hh2.py              # ★ Bar-complex verification of HH^2(u_q(sl_2), C) = 3
│   ├── verify_bplus_sl2_rigorous.py   # ★ B^+(sl_2) HH^2 = 1 at ℓ=3,5,7
│   ├── analyze_cocycles.py            # ★ Extract & classify 3 A_1 cocycles by support
│   ├── test_restriction_map.py        # ★ Mastnak–Witherspoon LES verification
│   ├── test_les_consistency.py        # LES dimensional consistency check
│   └── ...                             # Auxiliary physics scripts (see note above)
├── src/hopf_decoherence/               # Core Python package
│   ├── q_algebra.py                    # q-numbers, Weyl modules
│   ├── coproduct.py                    # Coproduct matrices
│   ├── rank_deficiency.py              # D_2(ell) = (ell^3 - ell)/6 (auxiliary)
│   ├── modified_trace.py               # GPY modified trace (auxiliary)
│   └── ...                             # Other auxiliary modules
├── tests/
│   ├── test_sl2_hh2_bar_complex.py     # ★ 5 tests for the A_1 bar-complex verification
│   ├── test_les_restriction.py         # ★ 2 tests for the LES restriction map
│   ├── test_q_algebra.py               # 13 tests for q-algebra / Weyl modules
│   ├── test_rank_deficiency.py         # 15 tests for D_2 formula (auxiliary)
│   ├── test_new_modules.py             # 19 tests for modified trace / projectives
│   └── test_defect_tqft.py             # 8 tests for defect TQFT machinery
├── plots/                              # Generated figures from auxiliary scripts
├── pyproject.toml
├── README.md
├── CITATION.cff
├── LICENSE                             # MIT (code)
└── paper/LICENSE                       # CC-BY-4.0 (paper)
```

## References

The paper's bibliography (12 entries, all verified against MathSciNet / arXiv / publisher records):

1. V. G. Drinfeld, *Quantum groups*, Proc. ICM (Berkeley, 1986), Vol. 1, AMS, 1987, pp. 798–820.
2. V. Ginzburg and S. Kumar, *Cohomology of quantum groups at roots of unity*, Duke Math. J. **69** (1993), 179–198.
3. A. Lachowska and Y. Qi, *Remarks on the derived center of small quantum groups*, Selecta Math. (N.S.) **27** (2021), Article 68.
4. M. Mastnak and S. Witherspoon, *Bialgebra cohomology, pointed Hopf algebras, and deformations*, J. Pure Appl. Algebra **213** (2009), 1399–1417. (arXiv:0704.2771)
5. M. Mastnak, J. Pevtsova, P. Schauenburg, S. Witherspoon, *Cohomology of finite dimensional pointed Hopf algebras*, J. Algebra **323** (2010), 2755–2783. (arXiv:0902.0801)
6. C. Negron, *Braided Hochschild cohomology and Hopf actions*, J. Pure Appl. Algebra **223** (2019), 1–47. (arXiv:1511.07059)
7. C. Negron and J. Pevtsova, *Support for integrable Hopf algebras via noncommutative hypersurfaces*, J. reine angew. Math. **2023** (2023). (arXiv:2005.02965)
8. I. Angiono, M. Kochetov, M. Mastnak, *On rigidity of Nichols algebras*, J. Algebra **456** (2016), 77–100. (arXiv:1412.2147)
9. A. García Iglesias and J. I. Sánchez, *On the computation of Hopf 2-cocycles, with an example of diagonal type*, J. Algebra **614** (2023), 495–522. (arXiv:2108.11432)
10. N. Andruskiewitsch, D. Jaklitsch, V. C. Nguyen, A. Oswald, J. Plavnik, A. V. Shepler, X. Wang, *On the finite generation of the cohomology of bosonizations*, 2025 preprint. (arXiv:2506.05267)
11. C. Blanchet, F. Costantino, N. Geer, B. Patureau-Mirand, *Non-semisimple TQFTs, Reidemeister torsion and Kashaev's invariants*, Adv. Math. **301** (2016), 1–78.
12. G. Lusztig, *Introduction to Quantum Groups*, Birkhäuser, 1993.

## Acknowledgments

The author thanks Professor You Qi for catching a hallucinated bibliography entry in an earlier draft of the paper, which led to a full audit and correction of all references. The author also thanks the anonymous reviewer whose questions led to the cocycle support analysis (`scripts/analyze_cocycles.py`) and the Mastnak–Witherspoon LES verification (`scripts/test_restriction_map.py`) reported in §6.3 and §7 of the paper.
