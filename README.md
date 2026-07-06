# SlapstackBet6 — Objects That Know What They Know

> Belief propagation object binding in Gabor packet space: objects are pose
> beliefs, borders are ownership marginals, permanence is inference.

Bets 4/5 (SlapstackBet5) proved generation and editing by construction:
frozen tensors, boolean masks. This repo is the probabilistic upgrade — the
"self-believing object network." An object is a pose belief on the Lie
algebra of Sim(2) plus an intrinsic template; an atom carries a categorical
belief about which object owns it. Binding is loopy BP; uncertainty is the
honest, measurable state of the system.

Method posture: Tindall et al., *Dynamics of disordered quantum systems
with two- and three-dimensional tensor networks* (arXiv:2503.05693), physics
removed — match the graph to the problem's geometry, use cheap local
message passing as the engine, know where the graph makes it lie (double
counting on loops → exact cavity messages: no atom confirms itself).

The Slapstack edge over capsule/GLOM-style routing-by-agreement: the
part→whole vote is **closed-form group algebra**, not a learned transform.

```
identity  = Sim(2)-invariant signature (su·f, su/sv, envelope-relative
            phase, color)         <- pose cannot touch it (unit-tested)
pose vote = the unique xi mapping template atom -> observed atom
            <- inverts the group action exactly (unit-tested, err < 1e-9)
```

## Core results (first drop — all CPU, assertion-locked in `tests.py`)

**6a — GO: static binding by pose-vote BP.** Three templates at unknown
poses + clutter: accuracy 1.000 ± 0.000 over 10 seeds, pose recovered to
~0.01, clutter rejected. Ablation: at 3× noise, identity channel alone
0.77 → full BP 0.98. Pose consistency adds ~20 points at every noise level.

**6b — GO: common fate + calibrated ignorance.** Two spatially interleaved
clouds, motion the only evidence, motions IDENTICAL for the first 5 frames.
Entropy pinned at 0.85 bits exactly while evidence is absent, 0.047 bits
after divergence, accuracy 1.0, ECE 0.082. "Knowing what it knows" is a
number with a calibration score.

**6c — GO: permanence by inference.** Object fully occluded for 5 frames:
pose covariance grows ×3.8 in the dark, emergence predicted to 0.017,
re-binding 1.000. Bet 5 was a tensor that *cannot* move; 6c is a belief
that survives the absence of evidence and reports how much it degraded.

## Open-problem drop (second drop — `tests_open.py`, 6× PASS)

### #4 SOLVED — the pi-ambiguity, and pose beliefs that hold two minds
The physics: `atom(theta+pi, phi) == atom(theta, -phi)`, so orientation
alone fixes rotation only mod pi — and `R(rho+pi) = -R(rho)` flips the
translation vote, so unresolved this breaks binding beyond ±90°. The fix
was in the representation all along: **envelope-relative phase**. Each
correspondence now emits two hypothesis votes (rho vs rho+pi, phase vs
−phase), each scored by phase consistency. Generic phases kill the wrong
hypothesis outright:

```
full SO(2) recovery, exact (err < 1e-9), tested at rho = 2.5, -2.8, 1.9
```

Degenerate phases (phi ≈ 0) leave both alive — and then the pose belief is
a **mixture** (density-peak init + EM + per-atom product-likelihood mode
weights). The satisfying part, `bet6_multimodal.py`:

```
180°-symmetric object, phases 0:   modes 0.50 / 0.50 at rho and rho−pi
same object, ONE generic phase:    modes 1.00 / 0.00 at the true rho
```

Genuine ambiguity is *held*, honestly, at 50/50 — there is no fact of the
matter and the system says so. One phase-breaking atom collapses the
posterior, because evidence multiplies across atoms (a build-time kill:
EM's responsibility-mass weights under-react to a single decisive atom —
mode weights must be Bayes factors, not vote counts).

### #1 SOLVED — real atoms as the template library
`bet6_open.py` trains two actual Slapstack reconstructions (Bet 5 torch
model, hard-concrete gates: a 92-atom tractor, a 91-atom star), extracts
the surviving atoms, and hides both in a cluttered scene at rotations of
**2.0 and −1.2 rad** — far beyond the old ±pi/2 wall, exercising the #4
fix on real atoms with whatever signature collisions optimization left in:

```
accuracy 0.955   pose error 0.023   (15 clutter atoms, noise 0.006)
```

### #2 SOLVED — borders as belief fields
The border is not drawn; it is the decision boundary of the ownership field
P(k|pixel) ∝ Σ_i b_i(k)·energy_i·envelope_i(pixel), computed through the
atoms' actual Gabor envelopes (amplitude-weighted evidence: louder atoms
claim harder). Quantified: mean pixel-ownership entropy **falls as evidence
improves** —

```
ownership entropy: 1.182 (clean scene)  <  1.326 (10x noisier scene)
```

Figures in `runs/bet6_open/`: scene render | ownership field | entropy
ridge (the border, glowing where beliefs are unsure). Honest note: absolute
entropy stays well above zero because both reconstructions carry large
coarse background envelopes that genuinely overlap across the canvas —
mixed ownership there is the *correct* belief, not a failure. The claim is
the contrast, and it is assertion-locked.

### #3 SHIPPED (GPU, untested) — SD as an oracle with belief masks
`bet6d_sds_oracle.py` replaces Bet 5b's hand-declared rectangle with the
output of inference: soft per-atom weights from BP
(`bet6_open.export_belief_weights`, model index space, verified [0,1],
0.95 of believed-object atoms protected). SDS gradients scale by w_i —
protection proportional to the belief that an atom IS the object, with
half-strength edits exactly at the soft border. The loop is Bet 5's
GPU-verified SDS; the only new mechanics are float weights where a hard
mask used to be (`apply_grad_masks` already multiplies by a float tensor).
First run on your box is the smoke test.

## Ledger

**Verified (CPU, this repo):**
- Closed-form pose votes invert the Sim(2) action exactly; intrinsic
  signatures bitwise pose-invariant (the fiber claim as a unit test).
- 6a binding 1.000 ± 0.000 / clutter rejected / BP adds ~20 points over
  the identity channel at all noise levels.
- 6b: entropy 0.85 bits while evidence absent by construction → 0.05
  after; acc 1.0; ECE 0.082.
- 6c: covariance ×3.8 in the dark; emergence predicted to 0.017; perfect
  re-binding.
- #4: exact full-SO(2) pose recovery via phase-scored hypothesis pairs;
  mixture pose beliefs hold genuine 50/50 ambiguity and collapse >0.95 on
  one symmetry-breaking phase, across rotations and seeds.
- #1: real recon atoms (92+91) bound at 0.955 among clutter at rotations
  2.0 / −1.2 rad.
- #2: border softness tracks belief uncertainty (entropy 1.182 vs 1.326,
  clean vs noisy), through actual Gabor envelopes.

**Killed (build-time, kept as warnings):**
- "Uniform correspondence init is fine, BP will sort it out" — wrong
  basin, seed-dependently. The identity channel must seed the pose channel.
- "5× parameter noise is just harder" — it was NaNs from unphysical
  negative sigma/f, silently poisoning fusion. Fields now clamped physical.
- "EM mixture weights are the pose posterior" — they measure explained
  vote mass and under-react to decisive single atoms (0.60/0.40 where the
  truth was ~1/0). Mode weights are now per-atom product likelihoods
  (Bayes factors). Evidence multiplies; votes don't.

**Open:**
1. bet6d on real GPU: the beach edit with *inferred* borders (script
   shipped, unexecuted).
2. Closing the loop on real images: templates here are still placed
   synthetically into scenes; the full pipeline is re-encoding a rendered/
   photographed scene into atoms and binding those (needs the Bet 5 recon
   as the scene encoder — semi-amortized, per Slapstack doctrine).
3. Label-switching and damping schedules at larger K remain untuned
   (partially mitigated: density-peak init + branch-aligned fusion).
4. Multimodal beliefs currently upgrade the FUSION step; the full BP loop
   in `bp_bind` is still unimodal-per-object (fine for phase-rich real
   atoms, wrong for symmetric objects in clutter — merge is mechanical).
5. The Entrain bridge (speculative, flagged): assignment consensus via
   Stuart-Landau phase synchrony instead of categorical messages.

## Files

```
bet6_bp_binding.py    core: Sim(2) algebra, votes, weighted Umeyama,
                      loopy BP with cavity messages, experiments 6a/6b/6c
bet6_multimodal.py    #4: pose_votes_2pi (phase-scored hypothesis pairs),
                      mixture pose beliefs, symmetry stress tests
bet6_open.py          #1 + #2: real-atom template library (trains Bet 5
                      recons on CPU), general bp_bind with 2pi votes,
                      ownership fields + border figures, weight exporter
bet6d_sds_oracle.py   #3: belief-weighted soft-mask SDS (GPU, untested)
bet5_gabor_sds.py     vendored Bet 5 renderer/model (repo merge makes
                      this the canonical copy)
tests.py              first-drop battery  (7x PASS)
tests_open.py         open-problem battery (6x PASS; first run trains the
                      template cache, ~30 s)
runs/                 figures + ledgers: 6a/6b/6c plots, border figures
                      at two noise levels, belief_weights_tractor.npy
```

## Run

```bash
pip install numpy matplotlib pillow torch
python tests.py            # 7x PASS
python tests_open.py       # 6x PASS (first run ~90 s: trains templates)
python bet6_bp_binding.py --exp all --out runs/bet6
python bet6_open.py        # real atoms + borders + weight export

# on the GPU box, after a bet5 scene exists:
python bet6d_sds_oracle.py --init-atoms runs/recon_tractor/atoms.pt \
    --weights runs/belief_weights_tractor.npy \
    --prompt "a tractor on a beach in miami, ocean, sand, blue sky" \
    --iters 1500 --render-size 512 --cfg 50 --no-camera \
    --out runs/bet6d_beach_soft
```

## Honesty notes

- Gaussian pose beliefs use Euclidean coordinates on the Lie algebra —
  small-covariance approximation; mixture modes are branch-aligned on the
  circle before fusion, which keeps the algebra honest near modes only.
- 6b/6c assume persistent atom tracks (the representation's own claim,
  assumed here, not demonstrated on real video).
- The weight-export demo binds a model's own atoms back to themselves as a
  shape check; the real deployment (scene model ⊃ template) is documented
  in `bet6d_sds_oracle.py` and untested until the GPU run.
- `bp_bind` remains unimodal per object (see Open #4-residual above).

---
*Slapstack lineage: hard-concrete gates, envelope-relative phase, honest
ledgers. Do not hype. Do not lie. Just show.*
