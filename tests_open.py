#!/usr/bin/env python3
"""
Open-problem drop verification battery. CPU. First run trains the template
cache (~30 s); subsequent runs ~1 min total.

    python tests_open.py
"""
import math

import numpy as np

from bet6_bp_binding import make_template, transform_atoms
from bet6_multimodal import (pose_votes_2pi, wrap_pi, mixture_bind,
                             run_symmetry_demo)
from bet6_open import (build_templates, make_scene, bp_bind, run_real_atoms,
                       export_belief_weights)


def test_vote_2pi_inverts_large_rotation():
    """The pi-ambiguity fix: with a generic phase, the phase-consistent
    hypothesis recovers rotations far beyond +/- pi/2, exactly."""
    rng = np.random.default_rng(0)
    tmpl = make_template(6, rng)                    # generic phases
    for rho in (2.5, -2.8, 1.9):
        xi = np.array([0.2, -0.1, rho, 0.15])
        obs = transform_atoms(tmpl, xi)
        for i in range(6):
            hyps = pose_votes_2pi(obs[i], tmpl[i])
            best = max(hyps, key=lambda vp: vp[1])[0]
            err = max(abs(best[0] - xi[0]), abs(best[1] - xi[1]),
                      abs(wrap_pi(best[2] - xi[2])), abs(best[3] - xi[3]))
            assert err < 1e-9, (rho, i, err)
    print("PASS vote_2pi: full SO(2) recovery, exact (err < 1e-9)")


def test_symmetric_holds_two_minds():
    """A 180-deg-symmetric object with degenerate phases is GENUINELY
    ambiguous: the pose belief must hold two ~equal modes at rho and
    rho - pi. Calibrated ambiguity in pose space."""
    for rho in (2.5, -2.0):
        for s in range(2):
            r, _ = run_symmetry_demo(rho_true=rho, seed=s, verbose=False)
            sym = r["symmetric"]
            assert min(sym[0][0], sym[1][0]) > 0.35, (rho, s, sym)
            rhos = sorted(wrap_pi(m[1][2]) for m in sym)
            gap = abs(wrap_pi(rhos[1] - rhos[0]))
            assert abs(gap - math.pi) < 0.05, (rho, s, rhos)
    print("PASS symmetric: two modes ~50/50, exactly pi apart")


def test_one_phase_breaks_symmetry():
    """One generic phase must COLLAPSE the mixture to the true pose —
    evidence multiplies across atoms (Bayes factor, not vote counting)."""
    for rho in (2.5, -2.0, 3.0):
        for s in range(2):
            r, _ = run_symmetry_demo(rho_true=rho, seed=s, verbose=False)
            brk = r["broken"]
            assert brk[0][0] > 0.95, (rho, s, brk)
            assert abs(wrap_pi(brk[0][1][2] - rho)) < 0.03, (rho, s, brk)
    print("PASS broken symmetry: single generic phase collapses the "
          "posterior > 0.95 to the true rotation")


def test_real_atoms_binding():
    """Templates are REAL Slapstack recon atoms (hard-concrete-gated torch
    model), hidden at rotations of 2.0 and -1.2 rad among clutter."""
    res, _ = run_real_atoms(noise=0.006, verbose=False)
    assert res["accuracy"] >= 0.90, res
    assert res["pose_err_max_abs"] < 0.06, res
    print(f"PASS real atoms: accuracy {res['accuracy']:.3f}, pose err "
          f"{res['pose_err_max_abs']:.3f} at rotations 2.0 / -1.2 rad "
          f"({res['n_template_atoms']} template atoms)")


def test_borders_sharpen_with_evidence():
    """The border is a belief field: mean pixel-ownership entropy must FALL
    as observation noise falls. Softness tracks uncertainty."""
    lo, _ = run_real_atoms(noise=0.006, verbose=False)
    hi, _ = run_real_atoms(noise=0.06, verbose=False)
    assert hi["mean_ownership_entropy"] > lo["mean_ownership_entropy"] + 0.05, \
        (lo["mean_ownership_entropy"], hi["mean_ownership_entropy"])
    print(f"PASS borders: ownership entropy {lo['mean_ownership_entropy']:.3f}"
          f" (clean) < {hi['mean_ownership_entropy']:.3f} (noisy)")


def test_weight_export_shape_and_semantics():
    res, pack = run_real_atoms(noise=0.006, verbose=False)
    obs, marg, P, ent, support, meta = pack
    w = export_belief_weights(meta, marg, k_obj=0,
                              path="runs/belief_weights_tractor.npy")
    assert len(w) == int(meta["n_model_tractor"])
    assert 0.0 <= w.min() and w.max() <= 1.0
    keep = meta["keep_tractor"]
    frac_protected = float((w[keep] < 0.5).mean())
    assert frac_protected > 0.85, frac_protected
    print(f"PASS weight export: model index space, [0,1], "
          f"{frac_protected:.2f} of believed-object atoms protected")


if __name__ == "__main__":
    test_vote_2pi_inverts_large_rotation()
    test_symmetric_holds_two_minds()
    test_one_phase_breaks_symmetry()
    test_real_atoms_binding()
    test_borders_sharpen_with_evidence()
    test_weight_export_shape_and_semantics()
    print("\nall open-problem tests pass")
