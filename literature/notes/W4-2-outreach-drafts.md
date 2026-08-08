# W4-2: Outreach email drafts — Negron / Witherspoon / Qi

- **Task ID**: W4-2
- **Agent**: Sub-agent 4b (Wave 4, general-purpose)
- **Date**: 2025-08-07
- **Status**: drafts only (NOT sent — author to review and send)
- **Author of underlying paper**: Parham Khairkhah (`pkhairkh@icloud.com`)
- **Paper**: *Hochschild Cohomology of Small Quantum Groups at Roots of Unity* — full LaTeX source at <https://github.com/pkhairkh/hopf-decoherence> (`paper/main.tex`); not yet posted to arXiv.

## Trigger for outreach

W3-1c and W4-1 computed `dim H̃¹_b(B⁺(u_q(sl_n)))` at `ℓ = 3` for `n = 2, 3, 4`,
giving `1, 2, 3` respectively — i.e. `dim H̃¹_b(B⁺) = n − 1 = rank(sl_n)`,
not `C(n+1, 2)` as the original structural decomposition predicted. The
structural decomposition `dim im(δ) = C(n+1, 2)` of Conjecture 1.1 in the
paper is therefore **refuted** for `n ≥ 3`. The full `HH²(u_q(sl_3))` bar
computation remains intractable, so the *total count* (8 vs. 9) is still
open. The pattern is clean and the cocycles have an unexpectedly rigid shape
(diagonal in PBW, linear in root-vector exponents, zero Cartan coefficients,
∂ʰ constraints implied by ∂ᶜ via the q-commutators) — exactly the kind of
clean pattern one would expect a structural expert to recognise or rule out.

## Address verification note (for the author)

The three recipient addresses below are taken from the task brief and from
the project's prior correspondence record. Before sending, the author
should re-verify the institutional addresses — especially `negron@usc.edu`
and `you.qi@virginia.edu` — against the recipients' current institutional
webpages. (Witherspoon's `S.Witherspoon@tamu.edu` is long-standing and very
likely current; she recently retired from TAMU, so a check that the
forwarding is still active is also worth a moment.) All three individuals
have publicly listed research pages and are reachable; if any bounce, the
author's webpage is the authoritative source.

The drafts below are written to be sent essentially as-is, with the cover
note prepended as a personal opening paragraph (so each outgoing message =
cover-note paragraph + email body). Total message length per recipient is
~300–350 words, well within a polite cold-outreach budget.

---

## 1. To: Cris Negron (USC) — braided HH cohomology / Hopf actions

- **To**: `negron@usc.edu` *(verify)*
- **Subject**: Empirical pattern in `H̃¹_b(B⁺(u_q(sl_n)))` at `ℓ = 3` — your braided-HH perspective?
- **Relevant paper of theirs**: *Braided Hochschild cohomology and Hopf actions* (arXiv:1511.07059, in corpus as `literature/texts/Negron-braided-HH.txt`); also Negron–Pevtsova, arXiv:2005.02965.

### Cover note (1 paragraph, for Parham to prepend)

> Dear Prof. Negron — I am reaching out as a researcher working on the
> Hochschild cohomology of small quantum groups at odd roots of unity. Your
> work on braided Hochschild cohomology and Hopf actions has been a key
> reference for me, and I have just turned up a computational pattern that
> puzzles me and seems to lie squarely in the regime your framework
> addresses. I would be very grateful for any reaction, however brief.

### Email body

> Dear Prof. Negron,
>
> I am writing to share a finding that may be of interest, and to ask
> whether it matches anything you have seen in your work on braided
> Hochschild cohomology of Hopf algebras.
>
> Briefly: I have been computing `dim H̃¹_b(B⁺(u_q(sl_n)))` at `ℓ = 3` —
> the degree-1 bialgebra cohomology of the positive-Borel bosonization, in
> the sense of Mastnak–Witherspoon — by direct linear algebra on the
> simultaneous derivation/coderivation equations. The results are
>
>   n = 2 (sl₂): dim = 1;   n = 3 (sl₃): dim = 2;   n = 4 (sl₄): dim = 3,
>
> i.e. `dim H̃¹_b(B⁺) = n − 1 = rank(sl_n)`. The cocycles are diagonal in
> the PBW basis, linear in the root-vector exponents, and have zero Cartan
> (`K_i`) coefficients.
>
> This refutes the structural prediction `dim = C(n+1, 2)` made in my draft,
> which was based on a "Cartan-type classes via `K_i^ℓ = 1`" picture. The
> `rank(g)` pattern is more suggestive of the Cartan subalgebra itself than
> of pairwise `K`-relations. Full source is at
> <https://github.com/pkhairkh/hopf-decoherence> (`paper/main.tex`);
> arXiv posting pending.
>
> My question: does this pattern — diagonal, linear, `rank(g)`-dimensional —
> ring any bells from your braided-HH / Hopf-action work? In particular, is
> there a structural reason one would expect `H̃¹_b` of the bosonization to
> be `rank(g)`-dimensional?
>
> Any reaction would be most appreciated.
>
> Best regards,
> Parham Khairkhah
> `pkhairkh@icloud.com`

---

## 2. To: Sarah Witherspoon (TAMU) — co-author of the Mastnak–Witherspoon LES

- **To**: `S.Witherspoon@tamu.edu` *(long-standing address; verify still active — Witherspoon is retired/emerita)*
- **Subject**: Behaviour of your bialgebra-cohomology LES at small odd `ℓ` — a puzzling `rank(g)` pattern
- **Relevant paper of theirs**: Mastnak–Witherspoon, *Bialgebra cohomology, pointed Hopf algebras, and deformations* (arXiv:0704.2771, in corpus as `literature/texts/Mastnak-Witherspoon-LES.txt`).

### Cover note (1 paragraph, for Parham to prepend)

> Dear Prof. Witherspoon — I have been working through the long exact
> sequence of your 2008 paper with Mitja Mastnak for the small quantum
> group at odd roots of unity, and have just found a computational result
> that contradicts a structural prediction I had based on it. I would very
> much value your perspective on whether this is expected behaviour in the
> regime where the hypothesis of your Theorem 6.1.4 fails.

### Email body

> Dear Prof. Witherspoon,
>
> I am writing to share a computational finding that bears on the
> Mastnak–Witherspoon long exact sequence (arXiv:0704.2771) connecting
> bialgebra cohomology of a bosonization to Hochschild cohomology of its
> Drinfeld double.
>
> Your Theorem 6.1.4 requires a hypothesis on prime divisors of `|Γ|`. For
> `u_q(sl_n)` at `ℓ = 3` this fails (`3 | |Γ| = 3ⁿ`), placing us in the gap
> regime where the LES still holds but the structural simplifications do
> not.
>
> Working in that gap, I computed `dim H̃¹_b(B⁺(u_q(sl_n)))` at `ℓ = 3` for
> `n = 2, 3, 4` by direct linear algebra on the derivation/coderivation
> equations. Results: `n = 2 → 1`, `n = 3 → 2`, `n = 4 → 3`, i.e.
> `dim H̃¹_b(B⁺) = n − 1 = rank(sl_n)`, rather than the `C(n+1, 2)` I had
> predicted from a "Cartan-type classes from `K_i^ℓ = 1`" picture. The
> cocycles are diagonal in PBW, linear in root-vector exponents, with zero
> Cartan coefficients.
>
> Two questions:
>
> 1. Have you seen this `rank(g)` pattern in any pointed-Hopf / bosonization
>    setting, especially in the small-`ℓ` gap?
> 2. Does the failure of the Thm 6.1.4 hypothesis have any known structural
>    consequence that might explain a `rank(g)`-dimensional `H̃¹_b`?
>
> Full source: <https://github.com/pkhairkh/hopf-decoherence>
> (`paper/main.tex`). Any reaction would be most appreciated.
>
> Best regards,
> Parham Khairkhah
> `pkhairkh@icloud.com`

---

## 3. To: You Qi (University of Virginia) — Lachowska–Qi, derived center of `u_q(g)`

- **To**: `you.qi@virginia.edu` *(verify; Qi has previously corresponded with the author)*
- **Subject**: `H̃¹_b(B⁺(u_q(sl_n))) = rank(sl_n)` at `ℓ = 3` — the structural decomposition was wrong
- **Relevant paper of theirs**: Lachowska–Qi, *A counterpart of the center of small quantum group at a root of unity* (the LQ21 reference cited in `paper/main.tex`); also Hemelsoet–Voorhaar (arXiv:2104.05113), which extends the LQ BGG approach.

### Cover note (1 paragraph, for Parham to prepend — familiar tone, prior correspondence)

> Hi You — following up on our earlier exchange about the sl₂ Hochschild
> computation. I have now extended the bialgebra-cohomology side of the
> story to sl₃ and sl₄, and the result has surprised me enough that I
> wanted to flag it before posting. The structural decomposition I had
> previously sketched to you is wrong in a clean way, and I would be glad
> of your read, especially in light of the Lachowska–Qi BGG machinery.

### Email body

> Hi You,
>
> Following up on our earlier correspondence: I have now extended the
> Mastnak–Witherspoon LES analysis to `sl_3` and `sl_4`, and the structural
> picture I had previously sketched to you is wrong in a clean way.
>
> Concretely, I computed `dim H̃¹_b(B⁺(u_q(sl_n)))` at `ℓ = 3` by direct
> linear algebra on the MW derivation/coderivation 1-cocycle equations:
>
>   n = 2: 1,   n = 3: 2,   n = 4: 3.
>
> So `dim H̃¹_b(B⁺) = n − 1 = rank(sl_n)`, not `C(n+1, 2)` as I had guessed.
> The cocycles are diagonal in PBW, exactly linear in root-vector exponents
> (residual `~10⁻¹⁶`), with zero Cartan coefficients; the `∂ʰ` constraints
> follow from the `∂ᶜ` ones via the q-commutator relations.
>
> What this means for the headline conjecture
> `dim HH²(u_q(g)) = C(n+1,2) + 2|Φ⁺|`: the structural decomposition
> (Cartan piece `C(n+1,2)`, root piece `2|Φ⁺|`) is wrong; the Cartan piece
> is `rank(g)`. Whether the *total count* is also wrong, or the root piece
> compensates, I cannot yet tell — full `HH²(u_q(sl_3))` is still
> intractable.
>
> Two questions:
>
> 1. Does the `rank(g)` pattern suggest anything from the Lachowska–Qi /
>    BGG perspective — e.g. a principal-block `HH²(sl_3)` computation via
>    Hemelsoet–Voorhaar that would distinguish 8 from 9?
> 2. Is there a structural reason the cocycles should be diagonal and
>    Cartan-coefficient-free?
>
> Full source: <https://github.com/pkhairkh/hopf-decoherence>
> (`paper/main.tex`). Would value your read.
>
> Best,
> Parham

---

## Why these three (and not others)

- **Negron**: His braided-Hochschild framework (arXiv:1511.07059) is the
  natural home for a structural interpretation of *why* the cocycles should
  be `rank(g)`-dimensional and Cartan-coefficient-free. His
  Negron–Pevtsova integrability framework (arXiv:2005.02965) is also
  relevant to the support-theoretic reading of the diagonal/linear
  structure.
- **Witherspoon**: Co-author of the LES that the entire reduction rests on.
  The pattern shows up precisely in the regime where her Thm 6.1.4
  hypothesis fails — she is the most likely person to know whether that
  failure has a known structural consequence. Also the appropriate person
  to ask about pointed-Hopf / bosonization precedents.
- **Qi**: Co-author of the LQ derived-center paper cited in the paper
  abstract; the BGG / sheaf-cohomology machinery of LQ + HV21 is, per
  W1-1a, the only realistic route to an independent `dim HH²(u_q(sl_3))`
  computation that could settle the open 8-vs-9 question. Already in
  correspondence with the author, so a familiar tone is appropriate.

Other natural candidates not drafted here (kept in reserve, in case the
author wishes to widen the outreach after the first three replies): Mitja
Mastnak (Mount Allison, LES co-author with Witherspoon — but Witherspoon is
the more natural first contact for the structural question); Iván Angiono
/ Mikhail Kochetov (rigidity of Nichols algebras — relevant to the
`2|Φ⁺|` root piece, which is *not* what this outreach is about); Nicolas
Hemelsoet (sl₃ BGG computation, principal-block s = 2 — the most direct
route to the 8-vs-9 question, but the author may prefer to go through Qi
first); Thomas Creutzig (log-KL / non-semisimple TQFT context).

## What the drafts deliberately do NOT do

- They do **not** claim the original count is refuted — only the
  structural decomposition. The 8-vs-9 question for `sl_3` is left
  explicitly open. (W4-1 §"Implications" is careful on this point; the
  drafts match that care.)
- They do **not** cite a non-existent arXiv number. The paper is referred
  to by its GitHub URL only, with an explicit "arXiv posting pending"
  note. (If the author prefers, an arXiv stamp can be added before
  sending.)
- They do **not** request coauthorship, a job, or a recommendation. They
  ask only for a reaction / pattern-match check.
- They do **not** attach the full paper as an attachment by default — only
  the GitHub link. (The author may of course attach the PDF if a
  recipient asks.)
