# hopf-decoherence

**Hochschild cohomology of small quantum groups at roots of unity — companion code**

**Author:** Parham Khairkhah (ORCID [0009-0000-7048-1397](https://orcid.org/0009-0000-7048-1397))
**Repository version:** 1.0.0
**Code license:** MIT  ·  **Paper license:** CC-BY-4.0

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](./LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/paper-CC--BY--4.0-green.svg)](./paper/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-67%20passing-brightgreen.svg)](#installation)

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
- **`tests/test_sl2_hh2_bar_complex.py`** — five dedicated tests covering the defining relations of $u_q(\mathfrak{sl}_2)$, associativity, the chain-complex property $d^2 \circ d^1 = 0$, the counit being an algebra map, and the $\dim\mathrm{HH}^2 = 3$ rank computation.
- **`src/hopf_decoherence/`** and the remaining `scripts/` — an auxiliary computational framework for the BCGP non-semisimple TQFT and BTZ black-hole entropy programme that the paper references speculatively in §7. This code is **not** required for the paper's main theorem (the $A_1$ verification) and is not the subject of the paper; it is included because the same quantum-group infrastructure underlies both.

## Status of the conjecture

| Type | Rank $n$ | $|\Phi^+|$ | Predicted $\dim\mathrm{HH}^2$ | Status |
|---|---|---|---|---|
| $A_1$ | 1 | 1 | 3 | **Theorem** (bar complex, $\ell=3$) |
| $A_2$ | 2 | 3 | 9 | Proposition (conditional on Cartan-class existence) |
| $B_2$ | 2 | 4 | 11 | Structural |
| $G_2$ | 2 | 6 | 15 | Structural |
| $A_n$ | $n$ | $n(n+1)/2$ | $3n(n+1)/2$ | Conjecture |
| $E_8$ | 8 | 120 | 276 | Conjecture |

The universal formula is stated as **Conjecture 1.1** in the paper. The $A_1$ case is **Theorem 1.2** (verified by the bar-complex script above). The $A_2$ case is **Proposition 1.3**, conditional on the existence of the Cartan-type classes discussed in §3 (an explicit cocycle construction is currently only available for $A_1$). Higher-rank direct verification is an open problem.

## Installation

The package requires Python ≥ 3.10 and the standard scientific stack:

```bash
git clone https://github.com/pkhairkh/hopf-decoherence.git
cd hopf-decoherence
pip install -e .[dev]
```

Dependencies: `numpy ≥ 1.24`, `scipy ≥ 1.10`, `sympy ≥ 1.12`, `matplotlib ≥ 3.7` (optional, for plotting).

## Usage

Run the full test suite (67 tests):

```bash
python -m pytest tests/ -v
```

Run only the bar-complex verification for $u_q(\mathfrak{sl}_2)$ at $\ell = 3$:

```bash
python scripts/verify_sl2_hh2.py
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
│   └── ...                             # Auxiliary physics scripts (see note above)
├── src/hopf_decoherence/               # Core Python package
│   ├── q_algebra.py                    # q-numbers, Weyl modules
│   ├── coproduct.py                    # Coproduct matrices
│   ├── rank_deficiency.py              # D_2(ell) = (ell^3 - ell)/6 (auxiliary)
│   ├── modified_trace.py               # GPY modified trace (auxiliary)
│   └── ...                             # Other auxiliary modules
├── tests/
│   ├── test_sl2_hh2_bar_complex.py     # ★ 5 tests for the A_1 bar-complex verification
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

The paper's bibliography (8 entries, all verified against MathSciNet / arXiv / publisher records):

1. V. G. Drinfeld, *Quantum groups*, Proc. ICM (Berkeley, 1986), Vol. 1, AMS, 1987, pp. 798–820.
2. V. Ginzburg and S. Kumar, *Cohomology of quantum groups at roots of unity*, Duke Math. J. **69** (1993), 179–198.
3. A. Lachowska and Y. Qi, *Remarks on the derived center of small quantum groups*, Selecta Math. (N.S.) **27** (2021), Article 68.
4. M. Mastnak and S. Witherspoon, *Bialgebra cohomology, pointed Hopf algebras, and deformations*, J. Pure Appl. Algebra **213** (2009), 1399–1417.
5. D. Štefan, *Hochschild cohomology on Hopf Galois extensions*, J. Pure Appl. Algebra **103** (1995), 221–233.
6. C. Negron, *Spectral sequences for the cohomology rings of a smash product*, J. Algebra **433** (2015), 73–106.
7. C. Blanchet, F. Costantino, N. Geer, B. Patureau-Mirand, *Non-semisimple TQFTs, Reidemeister torsion and Kashaev's invariants*, Adv. Math. **301** (2016), 1–78.
8. G. Lusztig, *Introduction to Quantum Groups*, Birkhäuser, 1993.

## Acknowledgments

The author thanks Professor You Qi for catching a hallucinated bibliography entry in an earlier draft of the paper, which led to a full audit and correction of all references.
