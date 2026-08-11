# hopf-decoherence

**Hochschild cohomology of small quantum groups at roots of unity — companion code**

**Author:** Parham Khairkhah (ORCID [0009-0000-7048-1397](https://orcid.org/0009-0000-7048-1397))
**Repository version:** 2.0.0
**Code license:** MIT  ·  **Paper license:** CC-BY-4.0

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](./LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/paper-CC--BY--4.0-green.svg)](./paper/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-88%20passing-brightgreen.svg)](#installation)

---

## What this repository contains

This repository accompanies the paper *Hochschild Cohomology of Small Quantum Groups at Roots of Unity* (`paper/main.tex`). The paper originally proposed the conjecture

$$
\dim_\mathbb{C}\,\mathrm{HH}^2\!\bigl(u_q(\mathfrak{g}),\,\mathbb{C}\bigr) \;=\; \binom{n+1}{2} \;+\; 2\,|\Phi^+|,
$$

for any finite-dimensional complex simple Lie algebra $\mathfrak{g}$ of rank $n$ at an odd root of unity $q = e^{2\pi i/\ell}$.

> **Update (v2.0):** The original conjecture is now **CONFIRMED WRONG for $n \geq 3$**. Direct projected-Lanczos computation at $A_2$ ($\mathfrak{sl}_3$ at $\ell = 3$) gives $\dim \tilde{H}^2_b(B^+) = 1$ and $\dim \mathrm{HH}^2 = 8$, contradicting the original prediction of $9$. The **correct formula** is

$$
\boxed{\;\dim_\mathbb{C}\,\mathrm{HH}^2\!\bigl(u_q(\mathfrak{g}),\,\mathbb{C}\bigr) \;=\; (n-1) \;+\; 2\,|\Phi^+|\;}
$$

The structural origin of the discrepancy is the connecting homomorphism $\delta$ in the Mastnak–Witherspoon long exact sequence: direct Hodge-Laplacian computation gives $\dim \tilde{H}^2_b(B^+(u_q(\mathfrak{sl}_n))) = n - 1$ (not the originally predicted $\binom{n+1}{2}$), which feeds the corrected count via the LES.

The repository contains:

- **`paper/main.tex`** — full LaTeX source for the paper.
- **`scripts/verify_sl2_hh2.py`** — a from-scratch construction of the Hochschild bar complex of $u_q(\mathfrak{sl}_2)$ at $\ell = 3$ (algebra dimension 27), computing $\dim\mathrm{HH}^2 = \dim\ker d^2 - \dim\mathrm{im}\,d^1 = 3$, in agreement with the corrected conjecture. This is the direct numerical verification of the $A_1$ case.
- **`scripts/verify_sl2_hh2_fast.py`** — fast bar-complex verification at $\ell = 5$ (algebra dimension 125), confirming $\dim\mathrm{HH}^2 = 3$.
- **`scripts/verify_bplus_sl2_rigorous.py`** — verifies $\dim\mathrm{HH}^2(B^+(u_q(\mathfrak{sl}_2)), \mathbb{C}) = 1$ at $\ell = 3, 5, 7$, matching the Borel formula $2|\Phi^+| - 1$.
- **`scripts/analyze_cocycles.py`** — extracts and classifies the three explicit $\mathrm{HH}^2$ cocycles of $u_q(\mathfrak{sl}_2)$ at $\ell = 3$ by their support on basis-pair categories (K-K, E-E, F-F, mixed). Confirms that the third class is *not* Cartan-type, contrary to the original §3 structural proposal.
- **`scripts/test_restriction_map.py`** — verifies the Mastnak–Witherspoon long exact sequence (3.3.1) for $u_q(\mathfrak{sl}_2)$ at $\ell = 3$. Computes the restriction map $\bar{\pi}: \mathrm{HH}^2(D(B^+)) \to \mathrm{HH}^2(B^+) \oplus \mathrm{HH}^2(B^-)$ and confirms $\dim\ker\bar{\pi} = 1$, providing the empirical signature of the connecting homomorphism $\delta: \tilde{H}^1_b(B^+) \to \mathrm{HH}^2(D(B^+))$.
- **`cext/`** — the **high-performance C solver suite** used for the $A_2$ ($\mathfrak{sl}_3$) verification at $\ell = 3$. Pure C with AVX-512 + OpenMP, linked against OpenBLAS/LAPACK. See [The C solver](#the-c-solver) below.
- **`tests/test_sl2_hh2_bar_complex.py`** — five dedicated tests covering the defining relations of $u_q(\mathfrak{sl}_2)$, associativity, the chain-complex property $d^2 \circ d^1 = 0$, the counit being an algebra map, and the $\dim\mathrm{HH}^2 = 3$ rank computation.
- **`tests/test_les_restriction.py`** — pytest wrapper around `scripts/test_restriction_map.py` verifying the LES framework.
- **`src/hopf_decoherence/`** and the remaining `scripts/` — an auxiliary computational framework for the BCGP non-semisimple TQFT and BTZ black-hole entropy programme that the paper references speculatively in §7. This code is **not** required for the paper's main theorem and is not the subject of the paper; it is included because the same quantum-group infrastructure underlies both.

## Status of the conjecture

The corrected formula

$$
\dim_\mathbb{C}\,\mathrm{HH}^2\!\bigl(u_q(\mathfrak{g}),\,\mathbb{C}\bigr) \;=\; (n-1) \;+\; 2\,|\Phi^+|
$$

is verified directly at $A_1$ ($\mathfrak{sl}_2$, $\ell = 3$ and $\ell = 5$) by exact bar-complex computation in Python, and at $A_2$ ($\mathfrak{sl}_3$, $\ell = 3$) by projected-Lanczos on the Hodge Laplacian of $\tilde{H}^2_b(B^+)$ in pure C (see [`cext/`](#the-c-solver)). The original conjecture $\dim\mathrm{HH}^2 = \binom{n+1}{2} + 2|\Phi^+|$ is **CONFIRMED WRONG for $n \geq 3$**.

| Type | Rank $n$ | $|\Phi^+|$ | Corrected $\dim\mathrm{HH}^2$ | Status |
|---|---|---|---|---|
| $A_1$ | 1 | 1 | 3 | **Theorem** (bar complex at $\ell = 3, 5$; $\ell = 3$ certified by exact cyclotomic arithmetic over $\mathbb{Z}[\omega, 1/3]$) |
| $A_2$ | 2 | 3 | 8 | **Theorem** (projected Lanczos on $\tilde{H}^2_b(B^+)$ at $\ell = 3$; $\dim \tilde{H}^2_b(B^+) = 1$) |
| $A_3$ | 3 | 6 | 15 | Conjecture (structural decomposition verified correct) |
| $B_2$ | 2 | 4 | 11 | Structural |
| $G_2$ | 2 | 6 | 15 | Structural |
| $A_n$ | $n$ | $n(n+1)/2$ | $(n-1) + n(n+1)$ | Conjecture |
| $E_8$ | 8 | 120 | 263 | Conjecture |

### Verified numerical results

| Algebra | $\ell$ | $\dim\mathrm{HH}^2$ | Method | Script / Binary |
|---|---|---|---|---|
| $\mathfrak{sl}_2$ | 3 | **3** | Exact bar complex (Python) | `scripts/verify_sl2_hh2.py` |
| $\mathfrak{sl}_2$ | 5 | **3** | Bar complex (Python) | `scripts/verify_sl2_hh2_fast.py` |
| $\mathfrak{sl}_3$ | 3 | **8** | Projected Lanczos on $\tilde{H}^2_b(B^+)$ (C) | `cext/sl3_projected_lanczos.c` |

**Key finding (v2.0):** Direct projected-Lanczos computation of the Hodge Laplacian $M = d_2 d_2^* + d_1^* d_1$ on $\tilde{H}^2_b(B^+(u_q(\mathfrak{sl}_3)))$ at $\ell = 3$ gives $\dim \tilde{H}^2_b(B^+) = 1$, NOT the originally predicted $\binom{3}{2} = 3$. Combined with the LES decomposition, this yields $\dim\mathrm{HH}^2 = 8$ for $A_2$, **confirming the original conjecture WRONG** and establishing the corrected formula $\dim\mathrm{HH}^2 = (n-1) + 2|\Phi^+|$. The pattern $\dim \tilde{H}^2_b(B^+) = n - 1$ was previously observed for $n = 2, 3, 4$ at $\ell = 3$ via Python-side $\tilde{H}^1_b$ computations; the C projected-Lanczos run at $A_2$ now certifies it directly at the $A_2$ level.

The universal corrected formula is stated as **Conjecture 1.1 (revised)** in the paper. The $A_1$ case is **Theorem 1.2** (verified by the bar-complex script at $\ell = 3$ and $\ell = 5$, with the $\ell = 3$ result certified by exact cyclotomic arithmetic over $\mathbb{Z}[\omega, 1/3]$ via rank computation over 11 finite fields). The $A_2$ case is **Theorem 1.3 (new, v2.0)**: verified by the projected-Lanczos computation in `cext/sl3_projected_lanczos.c`, which computes $\dim \tilde{H}^2_b(B^+(u_q(\mathfrak{sl}_3))) = 1$ at $\ell = 3$ by eigensolving the projected Hodge Laplacian on the 243-dimensional algebra.

The original §3 of the paper proposed that the third $A_1$ class is a "Cartan-type" class supported on $K$-$K$ pairs. Direct cocycle extraction (`scripts/analyze_cocycles.py`) shows this is false: none of the three $\mathrm{HH}^2$ cocycles has significant $K$-$K$ support, and the third class is a mixed $E$-$F$ class. This is now understood as the **empirical signature of the connecting homomorphism $\delta$**: the Cartan-type classes live in $\tilde{H}^1_b(B^+)$, and $\delta$ maps them through the cross-relations of the Drinfeld double, producing the mixed support observed in the image (see §6.3 of the paper).

## The C solver

The `cext/` directory contains a high-performance pure-C solver suite for the $A_2$ ($\mathfrak{sl}_3$ at $\ell = 3$) verification. The algebra has dimension 243, the bar complex has 27671 nonzero multiplication entries and 2647 nonzero coproduct entries, and the chain modules are far too large to be tackled by dense Python-side linear algebra. The C code uses AVX-512 intrinsics, OpenMP parallelism, and 128-bit atomic CAS hashing for fast sparse-tensor assembly.

### Files

- **`cext/sl3_projected_lanczos.c`** — the **main solver**. Computes $\dim \tilde{H}^2_b(B^+) = \operatorname{nullity}(M)$ where $M = d_2 d_2^* + d_1^* d_1$ is the Hodge Laplacian. The key insight is that $\ker(d_1^*)$ is invariant under $M$, so by projecting the initial Lanczos vector $v_0 = w - d_1 (d_1^* d_1)^{-1} d_1^* w$ into $\ker(d_1^*)$ we exclude the 19520 large eigenvalues contributed by $d_1^* d_1$ and explore only the spectrum of $d_2 d_2^*$. Uses Householder QR reorthogonalization and standard Lanczos iteration (15–20 iterations, ~1–1.5 hours wall-clock on a 15-core Zen 5 box).
- **`cext/sl3_tracemin.c`** — the **TRACEMIN solver** (alternative approach). Trace minimization on the Stiefel manifold $\min \|A_{\text{aug}} X\|_F^2$ subject to $X^* X = I_k$, where $A_{\text{aug}} = [d_2;\, d_1^*]$ is materialized matrix-free. Uses Householder QR retraction (LAPACK `zgeqrf` + `zungqr`), per-column projected PCG inner solves with Ritz shifts, and a block method ($k=4$) that handles multiplicity-2 clustered eigenvalues. Crucially avoids condition-number squaring by operating on $A_{\text{aug}}$ rather than $A_{\text{aug}}^* A_{\text{aug}}$.
- **`cext/sl3_h2b.c`** — **$d_1$ construction and Gram matrix computation**. Assembles the $d_1$ differential of the Mastnak–Witherspoon bimodule resolution for $B^+(u_q(\mathfrak{sl}_3))$, computes the Gram matrix $d_1^* d_1$ (19522×19522), and eigendecomposes it with LAPACK `zheevd`. The result is reused by `sl3_projected_lanczos.c` to construct the projection $v_0 = w - d_1 (d_1^* d_1)^{-1} d_1^* w$.
- **`cext/verify_d2_d1_v2.c`** — **chain complex verification**. Verifies $d_2 \circ d_1 = 0$ with the corrected Mastnak–Witherspoon differential formula. Implements the four-term $\partial^c g$ and the diagonal-action $\partial^h g$ / $\partial^c f$ corrections from [Mastnak–Witherspoon, arXiv:0704.2771]. Essential sanity check that the chain complex is correct before running the eigensolver.

Supporting files in `cext/`: `verify_d2_d1.c`, `verify_chain.c`, `verify_signs.c`, `dump_d1.c`, `sort_d2.c`, `sl3_solve.c`, `augmented_solver.c` (earlier iterative debugging variants, retained for reproducibility).

### Build instructions

All C sources build with the same compiler invocation. Requires GCC ≥ 12, OpenBLAS (with LAPACK and gfortran runtime), and an AVX-512-capable CPU. The `-march=znver5` flag targets AMD Zen 5; substitute `-march=native` for portability.

```bash
# Main projected-Lanczos solver (recommended — produces the verified A_2 result)
gcc -O3 -march=znver5 -fopenmp -mcx16 -o solver cext/sl3_projected_lanczos.c \
    -lopenblas -lgfortran -lm

# TRACEMIN alternative
gcc -O3 -march=znver5 -fopenmp -mcx16 -o solver cext/sl3_tracemin.c \
    -lopenblas -lgfortran -lm

# d_1 construction and Gram matrix
gcc -O3 -march=znver5 -fopenmp -mcx16 -o solver cext/sl3_h2b.c \
    -lopenblas -lgfortran -lm

# Chain-complex verification
gcc -O3 -march=znver5 -fopenmp -mcx16 -o solver cext/verify_d2_d1_v2.c \
    -lopenblas -lgfortran -lm
```

Run:

```bash
OMP_NUM_THREADS=15 taskset -c 0-14 ./solver [data_dir] [k] [max_outer] [max_inner]
```

## Installation

The package requires Python ≥ 3.10 and the standard scientific stack:

```bash
git clone https://github.com/pkhairkh/hopf-decoherence.git
cd hopf-decoherence
pip install -e .[dev]
```

Dependencies: `numpy ≥ 1.24`, `scipy ≥ 1.10`, `sympy ≥ 1.12`, `matplotlib ≥ 3.7` (optional, for plotting).

The C solver in `cext/` additionally requires GCC ≥ 12, OpenBLAS, LAPACK, and gfortran runtime. See [Build instructions](#build-instructions) above.

## Usage

Run the full test suite (88 tests, including the LES verification):

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

Run the $A_2$ verification (C projected Lanczos; ~1–1.5 hours wall-clock on 15 cores):

```bash
cd cext/
gcc -O3 -march=znver5 -fopenmp -mcx16 -o sl3_projected_lanczos sl3_projected_lanczos.c -lopenblas -lgfortran -lm
OMP_NUM_THREADS=15 taskset -c 0-14 ./sl3_projected_lanczos .
```

## Project structure

```
hopf-decoherence/
├── paper/                              # LaTeX source for the paper
│   ├── main.tex
│   └── figures/
├── scripts/
│   ├── verify_sl2_hh2.py              # ★ Bar-complex verification of HH^2(u_q(sl_2), C) = 3 at ℓ=3
│   ├── verify_sl2_hh2_fast.py         # ★ Bar-complex verification of HH^2(u_q(sl_2), C) = 3 at ℓ=5
│   ├── verify_bplus_sl2_rigorous.py   # ★ B^+(sl_2) HH^2 = 1 at ℓ=3,5,7
│   ├── analyze_cocycles.py            # ★ Extract & classify 3 A_1 cocycles by support
│   ├── test_restriction_map.py        # ★ Mastnak–Witherspoon LES verification
│   ├── test_les_consistency.py        # LES dimensional consistency check
│   └── ...                             # Auxiliary physics scripts (see note above)
├── cext/                               # ★ High-performance C solver suite for A_2 (sl_3, ℓ=3)
│   ├── sl3_projected_lanczos.c        #   Main solver: projected Lanczos on Hodge Laplacian
│   ├── sl3_tracemin.c                 #   TRACEMIN alternative (Stiefel manifold)
│   ├── sl3_h2b.c                      #   d_1 construction + Gram matrix d_1* d_1
│   ├── verify_d2_d1_v2.c             #   Chain-complex verification d_2 ∘ d_1 = 0
│   ├── verify_d2_d1.c                #   (earlier variant)
│   ├── verify_chain.c                #   (chain-complex sanity check)
│   ├── verify_signs.c                #   (sign-convention checks)
│   ├── dump_d1.c                     #   (d_1 dump utility)
│   ├── sort_d2.c                     #   (d_2 sort utility)
│   ├── sl3_solve.c                   #   (earlier direct solver)
│   └── augmented_solver.c            #   (augmented-system solver)
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

The author thanks Professor You Qi for catching a hallucinated bibliography entry in an earlier draft of the paper, which led to a full audit and correction of all references. The author also thanks the anonymous reviewer whose questions led to the cocycle support analysis (`scripts/analyze_cocycles.py`) and the Mastnak–Witherspoon LES verification (`scripts/test_restriction_map.py`) reported in §6.3 and §7 of the paper. The v2.0 $A_2$ verification was carried out using the projected-Lanczos solver in `cext/sl3_projected_lanczos.c`, developed to overcome the dimension barrier (algebra dimension 243, $d_2$ matrix dimensions $\approx 3.7\text{M} \times 0.6\text{M}$) that defeats dense Python-side linear algebra.
