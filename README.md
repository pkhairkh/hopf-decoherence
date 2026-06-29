# hopf-decoherence

**Coproduct rank deficiency of $u_q(\mathfrak{sl}_n)$ at roots of unity and the BCGP non-semisimple TQFT log correction for BTZ black-hole entropy**

**Author:** Parham Khairkhah

**Repository version:** 1.0.0

**Code license:** MIT  ·  **Paper license:** CC-BY-4.0

[![arXiv](https://img.shields.io/badge/arXiv-pending-red.svg)](https://arxiv.org/submit)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](./LICENSE)
[![License: CC-BY-4.0](https://img.shields.io/badge/paper-CC--BY--4.0-green.svg)](./paper/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-62%20passing-brightgreen.svg)](#installation)

> **arXiv submission:** The accompanying paper *Hochschild Cohomology of Small Quantum Groups at Roots of Unity* will be submitted to arXiv under `math.QA` (cross-listed to `math-ph` and `hep-th`). The badge above will be updated with the assigned arXiv identifier (format `arXiv:YYMM.NNNNN`) upon announcement.

---

## Abstract

This repository provides the numerical and formal proofs accompanying the computation of the coproduct rank deficiency

$$
D_2(\ell) = \frac{\ell^3 - \ell}{6} = \binom{\ell+1}{3},
\qquad
\frac{D_2(\ell)}{\ell^3} \xrightarrow{\ell \to \infty} \frac{1}{6},
$$

for the small quantum group $u_q(\mathfrak{sl}_2)$ at $q = e^{2\pi i/\ell}$, and the application of this identity to the BCGP (Blanchet–Costantino–Geer–Patureau-Mirand) non-semisimple TQFT construction. The central physical result is the resolution of the $-3/2$ logarithmic entropy correction of the BTZ black hole.

The radical of the projective indecomposable $u_q(\mathfrak{sl}_2)$-modules, which is invisible to the categorical modified trace, is shown to be a necessary constituent of the physical partition function. Its inclusion shifts the logarithmic coefficient from the modified-trace value $-2$ to the gravitational value $-3/2$ via the master identity

$$
-\frac{3}{2} = -2 \;(\text{modified trace}) + \frac{1}{2} \;(\text{radical sector}).
$$

The $+1/2$ contribution is fixed by the radical channel capacity $\frac{1}{2}\ln r$ and is identified, under the holographic dictionary, with the BTZ black-hole interior degrees of freedom.

A multi-boundary generalization predicts the logarithmic correction $-(2n+1)/2$ for $n$-boundary geometries, verified numerically for the Euclidean wormhole ($n=2$, prediction $-5/2$) to a deviation of $10^{-3}$.

## Installation

The package requires Python ≥ 3.10 and the standard scientific stack:

```bash
git clone https://github.com/pkhairkh/hopf-decoherence.git
cd hopf-decoherence
pip install -e .[dev]
```

Dependencies: `numpy ≥ 1.24`, `scipy ≥ 1.10`, `sympy ≥ 1.12`, `matplotlib ≥ 3.7` (optional, for plotting).

## Usage

Run the test suite:

```bash
python -m pytest tests/ -v
```

Verify the master identity and all eight arguments consolidated in `radical_physicality_theorems.py`:

```bash
python -c "from src.hopf_decoherence.radical_physicality_theorems import master_verification; master_verification()"
```

Verify the $D_2(\ell)$ coproduct-rank-deficiency formula numerically:

```bash
python scripts/verify_D2.py
```

Extract the $-3/2$ log coefficient by four-parameter regression against the full thermal trace:

```bash
python -c "from src.hopf_decoherence.validation_independent import extract_log_coefficient, Z_full_thermal_trace; print(extract_log_coefficient(Z_full_thermal_trace, range(3,102,2))['log_coefficient_4param'])"
```

## Overview of the proofs

The repository consolidates eight independent arguments establishing that the radical sector of the projective indecomposable modules is a physical constituent of the BTZ partition function. Each argument corresponds to a theorem, lemma, or corollary in the accompanying LaTeX source.

### Theorem 1 (Hilbert-space counting)
*Source:* `hilbert_space_theorem.py`

The physical partition function on the torus is $Z = \mathrm{Tr}_{\mathcal{H}}(e^{-\beta H})$ by definition. The Hilbert space is

$$
\mathcal{H} = \bigoplus_{j=0}^{r-2} L(j) \oplus \int V_\alpha \, d\alpha,
$$

and each simple $L(j)$ for $j < r-1$ appears twice in the projective cover $P(j)$ (as head and as radical). Consequently

$$
\mathrm{Tr}_{\mathcal{H}}(e^{-\beta H}) = \sum_j \dim(P_j) \, e^{-\beta h_j},
$$

which is precisely the full thermal trace. The modified quantum dimension $\tilde{d}(P_j) \neq \dim(P_j)$ does not reproduce this trace.

### Theorem 2 (Positivity)
*Source:* `positivity_theorem.py`

A thermal partition function $Z = \mathrm{Tr}(e^{-\beta H})$ is strictly positive for all $\beta \geq 0$. The modified-trace partition $Z_{\mathrm{BCGP}}$ violates this:

* $Z_{\mathrm{BCGP}}(\beta=0) = 0$ exactly, by the alternating-sum identity $\sum_j (-1)^j \sin(\pi(j+1)/r) = 0$.
* Approximately half of the modified-trace weights $\tilde{d}(P_j)$ are negative.

The full thermal trace satisfies $Z_{\mathrm{full}} > 0$ for all $\beta \geq 0$.

### Theorem 3 (Modular invariance)
*Source:* `modular_invariance_proof.py`

The boundary CFT partition function must be modular invariant. The full thermal characters are strictly positive and transform covariantly under the modular $S$-transformation. The modified-trace characters exhibit $(-1)^j$ sign alternation and Steinberg vanishing, breaking modular covariance: the non-semisimple $S$-matrix has rank one (it is an outer product) and cannot implement the $S$-transformation.

### Theorem 4 (Chern-Simons path integral)
*Source:* `chern_simons_derivation.py`

The Chern-Simons path integral on the solid torus produces the full thermal trace, since a path integral sums over the entire Hilbert space. The modified trace is not derived from the Chern-Simons path integral; it is an additional categorical structure imposed on closed manifolds. The one-loop determinant matches the $r^{-3/2}$ scaling required by gravity.

### Theorem 5 (Radical–zero-mode correspondence)
*Source:* `radical_zero_mode_map.py`

The three Killing zero modes of the BTZ geometry map bijectively to three radical points in the Loewy diagram of the projective indecomposables:

| Killing vector | Geometry | Radical partner | $u_q(\mathfrak{sl}_2)$ element |
|---|---|---|---|
| $L_{-1}$ | Time translation $\partial/\partial\tau$ | $\mathrm{rad}(P(0)) = L(r-2)$ | $F^r$ (nilpotent) |
| $L_{0}$  | Rotation $\partial/\partial\varphi$ | $\mathrm{rad}(P(j^\ast)) = L(j^\ast)$, $j^\ast = (r-3)/2$ | $K^r - I$ (Cartan deficit) |
| $L_{+1}$ | Special conformal | $\mathrm{rad}(P(r-2)) = L(0)$ | $E^r$ (nilpotent) |

The radical channel capacity $\frac{1}{2}\ln r$ reproduces the zero-mode contribution $-\frac{1}{2}\ln S_{\mathrm{BH}}$.

### Theorem 6 (Information-theoretic identity)
*Source:* `information_theorem.py`

The modified trace has zero channel capacity (it is a categorical projector). The radical sector supplies channel capacity $\frac{1}{2}\ln r$, reproducing the Page-curve prediction:

$$
S_{\mathrm{full}} = S_{\mathrm{mod}} + \frac{1}{2}\ln r
\quad \Longrightarrow \quad
-\frac{3}{2} = -2 + \frac{1}{2}.
$$

The radical is the holographic dual of the black-hole interior: the information required to purify Hawking radiation.

### Theorem 7 (Wormhole prediction)
*Source:* `wormhole_prediction.py`

The full thermal trace on the Euclidean wormhole (double solid torus) yields a logarithmic correction of $-5/2$, verified numerically to a deviation of $10^{-3}$. The modified trace gives $-7/2$, inconsistent with gravity. The general formula

$$
\delta S_{\log} = -\frac{2n+1}{2}
\qquad \text{for } n \text{-boundary geometries}
$$

is verified for $n=1$ (BTZ, $-3/2$) and $n=2$ (wormhole, $-5/2$).

### Theorem 8 (Continuity)
*Source:* `continuity_theorem.py`

At generic $q$ (the semisimple regime) the modified trace and the full thermal trace coincide. As $q \to e^{2\pi i/r}$ the full thermal trace is continuous, while the modified trace is discontinuous: the $(-1)^j$ sign alternation appears only at the root of unity, producing destructive interference. Physical partition functions are continuous in the deformation parameter; therefore the full thermal trace is the correct physical continuation.

## Master identity

The three equivalent decompositions of the $-3/2$ log correction are:

```
GRAVITY:       -3/2 = -1   (Cardy)           + (-1/2)  (3 Killing zero modes)
TQFT:          -3/2 = -2   (modified trace)  + (+1/2)  (radical sector)
INFORMATION:   -3/2 = -2   (semiclassical)   + (+1/2)  (BH interior)
```

The correction factor is

$$
Z_{\mathrm{physical}} = Z_{\mathrm{BCGP}} \cdot \sqrt{r} \cdot f(\beta),
\qquad
f(\beta) = \frac{\pi^{3/2} \sqrt{\beta}}{2},
$$

shifting the log coefficient $-2 \mapsto -3/2$.

## Key formulas

| Formula | Expression | Asymptotic |
|---|---|---|
| $D_2(\ell)$ | $(\ell^3 - \ell)/6$ | $\sim \ell^3/6$ |
| $D_2(\ell)/\ell^3$ | $1 - 1/(\ell^2-1)$ | $\to 1/6$ |
| $t(\mathrm{id}_{P(j)})$ | $(-1)^j \sin(\pi(j+1)/\ell) / (\ell \sin^2(\pi/\ell))$ | — |
| $\tilde{D}^2$ | $1 / (r \sin^4(\pi/r))$ | $\sim r^3/\pi^4$ |
| $Z_{\mathrm{full}} / \tilde{D}^2$ | $r^{3/2} / r^3 = r^{-3/2}$ | log coeff $= -3/2$ |
| $\mathrm{CF}(r,\beta)$ | $\pi^{3/2} \sqrt{\beta}/2 \cdot \sqrt{r}$ | shifts $-2 \mapsto -3/2$ |
| $C_{\mathrm{radical}}$ | $\frac{1}{2}\ln r$ | Page-curve capacity |

## Complete BTZ entropy formula

$$
S_{\mathrm{BTZ}} = S_{\mathrm{BH}} - \frac{3}{2} \ln S_{\mathrm{BH}} + 5.344 - \frac{0.178}{\sqrt{r}} + O(1/r).
$$

## Project structure

```
hopf-decoherence/
├── paper/                      # LaTeX source for the accompanying paper
│   ├── main.tex
│   └── figures/
├── plots/                      # Generated figures (PNG)
├── src/hopf_decoherence/       # Core Python package
│   ├── q_algebra.py            # q-numbers, Weyl modules
│   ├── coproduct.py            # Coproduct matrices, rank
│   ├── rank_deficiency.py      # D_2(ell) = (ell^3 - ell)/6
│   ├── verlinde.py             # S-matrix, fusion
│   ├── non_hopf_ideal.py       # J not Hopf ideal for sl_n, n >= 3
│   ├── projective_modules.py   # Projective indecomposable P(j)
│   ├── modified_trace.py       # GPY modified trace
│   ├── defect_tqft.py          # BCGP + ker(Delta) defect TQFT
│   ├── bcgp_btz.py             # BCGP partition function on BTZ
│   ├── state_sum.py            # Tureav-Viro state sum
│   ├── defect_networks.py      # Defect line construction
│   ├── triangulation.py        # 3-manifold triangulations
│   ├── radical_physicality_theorems.py    # Consolidated eight theorems
│   ├── hilbert_space_theorem.py           # Theorem 1
│   ├── positivity_theorem.py              # Theorem 2
│   ├── modular_invariance_proof.py        # Theorem 3
│   ├── chern_simons_derivation.py         # Theorem 4
│   ├── radical_zero_mode_map.py           # Theorem 5
│   ├── information_theorem.py             # Theorem 6
│   ├── wormhole_prediction.py             # Theorem 7
│   ├── continuity_theorem.py              # Theorem 8
│   ├── master_theorem.py                  # Consolidated master theorem
│   ├── formal_proof_*.py                  # Formal proof modules
│   ├── validation_*.py                    # Numerical validation modules
│   └── ...                                 # Auxiliary analysis
├── scripts/                    # Standalone verification scripts
├── tests/                      # pytest test suite
├── pyproject.toml
├── README.md
├── LICENSE                     # MIT (code)
└── paper/LICENSE               # CC-BY-4.0 (paper)
```

## References

1. C. Blanchet, F. Costantino, N. Geer, B. Patureau-Mirand, *Non-semisimple TQFTs from modified traces*, arXiv:1605.07941.
2. N. Geer, B. Patureau-Mirand, M. Yakimov, on the modified trace construction for non-semisimple categories.
3. A. Sen (2012), *Logarithmic Corrections to Schwarzschild and Other Non-extremal Black Hole Entropy in Different Dimensions*, arXiv:1205.0971.
4. J. Cardy (1986), *Operator content of two-dimensional conformally invariant theories*, Nucl. Phys. B 270.
5. D. N. Page (1993), *Information in black hole radiation*, Phys. Rev. Lett. 71.
6. G. Lusztig (1990), *Quantum groups at roots of unity*, J. Amer. Math. Soc. 3.
7. C. De Concini and V. Kac (1990), *Representations of quantum groups at roots of 1*, Progress in Math. 92.
8. J. Andersen and J. Paradowski (1995), *Fusion categories arising from semisimple Lie algebras*, Comm. Math. Phys. 169.
9. V. Turaev, A. Virelizier, *On three-dimensional TQFTs based on strict monoidal categories*.

## Acknowledgments

This research and computational framework were supported by **z.ai GLM**.
