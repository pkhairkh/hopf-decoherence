# hopf-decoherence

**Hochschild cohomology of small quantum groups at roots of unity — companion code**

**Author:** Parham Khairkhah (ORCID [0009-0000-7048-1397](https://orcid.org/0009-0000-7048-1397))
**Repository version:** 2.0.0
**Code license:** MIT  ·  **Paper license:** CC-BY-4.0

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](./LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/paper-CC--BY--4.0-green.svg)](./paper/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-191%20collected-brightgreen.svg)](#installation)

---

## What this repository contains

This repository accompanies the paper *Hochschild Cohomology of Small Quantum Groups at Roots of Unity* (`paper/main.tex`). The paper studies the dimension of the second Hochschild cohomology group $\mathrm{HH}^2(u_q(\mathfrak{g}), \mathbb{C})$ of the small quantum group $u_q(\mathfrak{g})$ at an odd root of unity $q = e^{2\pi i/\ell}$.

The **corrected formula** established and verified in this repository is

$$
\boxed{\;\dim_\mathbb{C}\,\mathrm{HH}^2\!\bigl(u_q(\mathfrak{g}),\,\mathbb{C}\bigr) \;=\; \operatorname{rank}(\mathfrak{g}) \;+\; 2\,|\Phi^+|\;}
$$

where $\operatorname{rank}(\mathfrak{g})$ is the rank of the simple Lie algebra and $|\Phi^+|$ is the number of positive roots. Equivalently, for $\mathfrak{sl}_n$ (where $n$ is the matrix size, so $\operatorname{rank}(\mathfrak{sl}_n)=n-1$ and $|\Phi^+| = \tfrac{n(n-1)}{2}$),

$$
\dim_\mathbb{C}\,\mathrm{HH}^2\!\bigl(u_q(\mathfrak{sl}_n),\,\mathbb{C}\bigr) \;=\; (n-1) \;+\; 2\,|\Phi^+|.
$$

> **Example ($\mathfrak{sl}_3$, $A_2$):** $\operatorname{rank}=2$, $|\Phi^+|=3$, so $2|\Phi^+|=6$ and $\dim\mathrm{HH}^2 = 2 + 6 = 8$.

The structural origin of this count is the connecting homomorphism $\delta$ in the Mastnak–Witherspoon long exact sequence: direct Hodge-Laplacian computation gives $\dim \tilde{H}^2_b(B^+(u_q(\mathfrak{sl}_n))) = \operatorname{rank}(\mathfrak{sl}_n) = n-1$, which feeds the count via the LES together with the Borel contribution $2|\Phi^+|$.

> **Note on the original conjecture.** An earlier draft proposed $\dim\mathrm{HH}^2 = \binom{n+1}{2} + 2|\Phi^+|$. This is **refuted**: direct projected-Lanczos computation at $A_2$ ($\mathfrak{sl}_3$, $\ell=3$) gives $\dim \tilde{H}^2_b(B^+) = 1$ and $\dim\mathrm{HH}^2 = 8$, contradicting the old prediction of $9$. The corrected formula $\operatorname{rank}(\mathfrak{g}) + 2|\Phi^+|$ replaces it throughout. All numerical values below use the corrected formula.

The repository contains:

- **`paper/main.tex`** — full LaTeX source for the paper.
- **`scripts/verify_sl2_hh2.py`** — a from-scratch construction of the Hochschild bar complex of $u_q(\mathfrak{sl}_2)$ at $\ell = 3$ (algebra dimension 27), computing $\dim\mathrm{HH}^2 = \dim\ker d^2 - \dim\mathrm{im}\,d^1 = 3$, in agreement with the corrected formula. This is the direct numerical verification of the $A_1$ case.
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
\dim_\mathbb{C}\,\mathrm{HH}^2\!\bigl(u_q(\mathfrak{g}),\,\mathbb{C}\bigr) \;=\; \operatorname{rank}(\mathfrak{g}) \;+\; 2\,|\Phi^+|
$$

is verified directly at $A_1$ ($\mathfrak{sl}_2$, $\ell = 3$ and $\ell = 5$) by exact bar-complex computation in Python, and at $A_2$ ($\mathfrak{sl}_3$, $\ell = 3$) by projected-Lanczos on the Hodge Laplacian of $\tilde{H}^2_b(B^+)$ in pure C (see [`cext/`](#the-c-solver)).

| Type | $\mathfrak{g}$ | $\operatorname{rank}(\mathfrak{g})$ | $\|\Phi^+\|$ | $\dim\mathrm{HH}^2 = \operatorname{rank} + 2\|\Phi^+\|$ | Status |
|---|---|---|---|---|---|
| $A_1$ | $\mathfrak{sl}_2$ | 1 | 1 | 3 | **Theorem** (bar complex at $\ell = 3, 5$; $\ell = 3$ certified by exact cyclotomic arithmetic over $\mathbb{Z}[\omega, 1/3]$) |
| $A_2$ | $\mathfrak{sl}_3$ | 2 | 3 | 8 | **Theorem** (projected Lanczos on $\tilde{H}^2_b(B^+)$ at $\ell = 3$; $\dim \tilde{H}^2_b(B^+) = 1$) |
| $A_3$ | $\mathfrak{sl}_4$ | 3 | 6 | 15 | Conjecture (structural decomposition verified correct) |
| $B_2$ | $\mathfrak{so}_5$ | 2 | 4 | 10 | Structural |
| $G_2$ | $\mathfrak{g}_2$ | 2 | 6 | 14 | Structural |
| $A_n$ | $\mathfrak{sl}_{n+1}$ | $n$ | $\tfrac{n(n+1)}{2}$ | $n^2 + 2n$ | Conjecture |
| $E_8$ | $\mathfrak{e}_8$ | 8 | 120 | 248 | Conjecture |

### Verified numerical results

| Algebra | $\ell$ | $\dim\mathrm{HH}^2$ | Verification method | Script / Binary |
|---|---|---|---|---|
| $\mathfrak{sl}_2$ | 3 | **3** | verified by bar complex (exact, Python) | `scripts/verify_sl2_hh2.py` |
| $\mathfrak{sl}_2$ | 5 | **3** | verified by bar complex (Python) | `scripts/verify_sl2_hh2_fast.py` |
| $\mathfrak{sl}_3$ | 3 | **8** | verified by projected Lanczos on $\tilde{H}^2_b(B^+)$ (C) | `cext/sl3_projected_lanczos.c` |

**Key finding (v2.0):** Direct projected-Lanczos computation of the Hodge Laplacian $M = d_2 d_2^* + d_1^* d_1$ on $\tilde{H}^2_b(B^+(u_q(\mathfrak{sl}_3)))$ at $\ell = 3$ gives $\dim \tilde{H}^2_b(B^+) = 1$, which equals $\operatorname{rank}(\mathfrak{sl}_3) = 2 - 1 = 1$. Combined with the LES decomposition and the Borel contribution $2|\Phi^+| = 6$, this yields $\dim\mathrm{HH}^2 = 1 + 6 + 1 = 8$ for $A_2$ (the $+1$ is the $\tilde{H}^2_b(B^-)$ mirror), establishing the corrected formula $\dim\mathrm{HH}^2 = \operatorname{rank}(\mathfrak{g}) + 2|\Phi^+|$. The pattern $\dim \tilde{H}^2_b(B^+) = \operatorname{rank}(\mathfrak{sl}_n) = n-1$ was previously observed for $n = 2, 3, 4$ at $\ell = 3$ via Python-side $\tilde{H}^1_b$ computations; the C projected-Lanczos run at $A_2$ now certifies it directly at the $A_2$ level.

The universal corrected formula is stated as **Conjecture 1.1 (revised)** in the paper. The $A_1$ case is **Theorem 1.2** (verified by the bar-complex script at $\ell = 3$ and $\ell = 5$, with the $\ell = 3$ result certified by exact cyclotomic arithmetic over $\mathbb{Z}[\omega, 1/3]$ via rank computation over 11 finite fields). The $A_2$ case is **Theorem 1.3 (new, v2.0)**: verified by the projected-Lanczos computation in `cext/sl3_projected_lanczos.c`, which computes $\dim \tilde{H}^2_b(B^+(u_q(\mathfrak{sl}_3))) = 1$ at $\ell = 3$ by eigensolving the projected Hodge Laplacian on the 243-dimensional algebra.

The original §3 of the paper proposed that the third $A_1$ class is a "Cartan-type" class supported on $K$-$K$ pairs. Direct cocycle extraction (`scripts/analyze_cocycles.py`) shows this is false: none of the three $\mathrm{HH}^2$ cocycles has significant $K$-$K$ support, and the third class is a mixed $E$-$F$ class. This is now understood as the **empirical signature of the connecting homomorphism $\delta$**: the Cartan-type classes live in $\tilde{H}^1_b(B^+)$, and $\delta$ maps them through the cross-relations of the Drinfeld double, producing the mixed support observed in the image (see §6.3 of the paper).

## The C solver

The `cext/` directory contains a high-performance pure-C solver suite for the $A_2$ ($\mathfrak{sl}_3$ at $\ell = 3$) verification. The algebra has dimension 243, the bar complex has 27671 nonzero multiplication entries and 2647 nonzero coproduct entries, and the chain modules are far too large to be tackled by dense Python-side linear algebra. The C code uses AVX-512 intrinsics, OpenMP parallelism, and 128-bit atomic CAS hashing for fast sparse-tensor assembly.

### Files

- **`cext/sl3_projected_lanczos.c`** — the **main solver**. Computes $\dim \tilde{H}^2_b(B^+) = \operatorname{nullity}(M)$ where $M = d_2 d_2^* + d_1^* d_1$ is the Hodge Laplacian. The key insight is that $\ker(d_1^*)$ is invariant under $M$, so by projecting the initial Lanczos vector $v_0 = w - d_1 (d_1^* d_1)^{-1} d_1^* w$ into $\ker(d_1^*)$ we exclude the 19520 large eigenvalues contributed by $d_1^* d_1$ and explore only the spectrum of $d_2 d_2^*$. Uses Householder QR reorthogonalization and standard Lanczos iteration (15–20 iterations, ~1–1.5 hours wall-clock on a 15-core Zen 5 box).
- **`cext/sl3_tracemin.c`** — the **TRACEMIN solver** (alternative approach). Trace minimization on the Stiefel manifold $\min \|A_{\text{aug}} X\|_F^2$ subject to $X^* X = I_k$, where $A_{\text{aug}} = [d_2;\, d_1^*]$ is materialized matrix-free. Uses Householder QR retraction (LAPACK `zgeqrf` + `zungqr`), per-column projected PCG inner solves with Ritz shifts, and a block method ($k=4$) that handles multiplicity-2 clustered eigenvalues. Crucially avoids condition-number squaring by operating on $A_{\text{aug}}$ rather than $A_{\text{aug}}^* A_{\text{aug}}$.
- **`cext/sl3_h2b.c`** — **$d_1$ construction and Gram matrix computation**. Assembles the $d_1$ differential of the Mastnak–Witherspoon bimodule resolution for $B^+(u_q(\mathfrak{sl}_3))$, computes the Gram matrix $d_1^* d_1$ (19522×19522), and eigendecomposes it with LAPACK `zheevd`. The result is reused by `sl3_projected_lanczos.c` to construct the projection $v_0 = w - d_1 (d_1^* d_1)^{-1} d_1^* w$.
- **`cext/verify_d2_d1_v2.c`** — **chain complex verification**. Verifies $d_2 \circ d_1 = 0$ with the corrected Mastnak–Witherspoon differential formula. Implements the four-term $\partial^c g$ and the diagonal-action $\partial^h g$ / $\partial^c f$ corrections from [Mastnak–Witherspoon, arXiv:0704.2771]. Essential sanity check that the chain complex is correct before running the eigensolver.

Supporting files in `cext/`: `verify_d2_d1.c`, `verify_chain.c`, `verify_signs.c`, `dump_d1.c`, `sort_d2.c`, `sl3_solve.c`, `augmented_solver.c` (earlier iterative debugging variants, retained for reproducibility). A `Makefile` is provided that builds the four primary targets.

### Build instructions

All C sources build with the same compiler flags. Requires GCC ≥ 12, OpenBLAS (with LAPACK and gfortran runtime), and an AVX-512-capable CPU. The `-march=znver5` flag targets AMD Zen 5; substitute `-march=native` for portability.

The recommended way to build is via the supplied `Makefile`, which builds all four primary targets (`sl3_projected_lanczos`, `sl3_tracemin`, `sl3_h2b`, `verify_d2_d1_v2`):

```bash
cd cext && make
```

Equivalently, build the main projected-Lanczos solver directly with the canonical command:

```bash
gcc -O3 -march=znver5 -fopenmp -mcx16 -o solver cext/sl3_projected_lanczos.c \
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

Run the full test suite (191 tests collected; see note below):

```bash
python -m pytest tests/ -v
```

> **Note:** Several test modules import the `hopf_decoherence` package (e.g. `tests/test_q_algebra.py`, `tests/test_rank_deficiency.py`, `tests/test_defect_tqft.py`). Make sure the editable install (`pip install -e .[dev]`) has been run first, or set `PYTHONPATH=src`, otherwise those modules will fail to import and the collection count will drop accordingly.

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
│   ├── Makefile                        #   Builds the four primary targets
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
├── ir/                                 # Letterplace / Anick resolution package (auxiliary)
├── tests/                              # 191 tests total
│   ├── test_sl2_hh2_bar_complex.py     # ★ 5 tests — A_1 bar-complex verification (HH^2 = 3)
│   ├── test_a1_certification.py        # 7 tests — A_1 exact cyclotomic certification
│   ├── test_les_restriction.py         # ★ 2 tests — LES restriction map
│   ├── test_sl3_bplus_hh2.py           # 3 tests — sl_3 B^+ bar-complex checks
│   ├── test_sl3_les.py                 # 9 tests — sl_3 LES consistency / refutation
│   ├── test_h1b_computation.py         # 7 tests — H̃¹_b(B^+) computations
│   ├── test_h1b_verification.py        # 5 tests — H̃¹_b(B^+) verification
│   ├── test_q_algebra.py               # 13 tests — q-algebra / Weyl modules
│   ├── test_rank_deficiency.py         # 15 tests — D_2 formula (auxiliary)
│   ├── test_new_modules.py             # 23 tests — modified trace / projectives
│   ├── test_defect_tqft.py             # 11 tests — defect TQFT machinery
│   ├── test_letterplace.py             # 36 tests — letterplace resolution
│   ├── test_ir_parser.py               # 31 tests — IR parser
│   ├── test_ir_groebner.py             # 13 tests — IR Gröbner / Knuth–Bendix
│   └── test_ir_uq_sl2.py               # 11 tests — u_q(sl_2) presentation
├── plots/                              # Generated figures from auxiliary scripts
├── literature/                         # Annotated reference texts and working notes
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
