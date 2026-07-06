#!/usr/bin/env python3
"""
Bet 6 verification battery. CPU, ~1 min, no GPU or diffusion model needed.

    python tests.py

Covers: exact Sim(2) algebra (vote inversion, signature invariance,
weighted fit), 6a binding + the BP-beats-signatures ablation, 6b
calibration, 6c permanence through occlusion.
"""
import math

import numpy as np

from bet6_bp_binding import (make_template, transform_atoms, signature,
                             pose_vote, fit_sim2, apply_pose,
                             run_6a, run_6b, run_6c)


def test_pose_vote_inverts_action():
    """pose_vote must exactly recover the pose that produced an observation."""
    rng = np.random.default_rng(0)
    tmpl = make_template(8, rng)
    xi = np.array([0.3, -0.2, 0.35, 0.25])
    obs = transform_atoms(tmpl, xi)
    for i in range(8):
        err = np.abs(pose_vote(obs[i], tmpl[i]) - xi).max()
        assert err < 1e-9, f"vote error {err}"
    print("PASS pose_vote: closed-form vote inverts the group action exactly")


def test_signature_invariance():
    """The fiber claim as a unit test: intrinsic signatures are untouched by
    any Sim(2) pose. Identity cannot be moved by moving."""
    rng = np.random.default_rng(1)
    tmpl = make_template(10, rng)
    for xi in ([0.5, -0.4, 0.4, 0.3], [-0.2, 0.1, -0.5, -0.25]):
        d = np.abs(signature(transform_atoms(tmpl, np.array(xi)))
                   - signature(tmpl)).max()
        assert d < 1e-9, f"signature moved by pose: {d}"
    print("PASS signature_invariance: pose cannot touch identity (err < 1e-9)")


def test_fit_sim2():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 0.4, (30, 2))
    xi = np.array([0.15, -0.3, 0.5, 0.2])
    Y = apply_pose(xi, X)
    w = rng.uniform(0.2, 1.0, 30)
    err = np.abs(fit_sim2(X, Y, w) - xi).max()
    assert err < 1e-8, f"fit error {err}"
    print(f"PASS fit_sim2: weighted Umeyama exact (err {err:.1e})")


def test_6a_binding():
    for s in range(3):
        r = run_6a(seed=s, verbose=False)
        assert r["accuracy"] == 1.0, f"seed {s}: acc {r['accuracy']}"
        assert r["pose_err_max_abs"] < 0.05
    print("PASS 6a: perfect binding + pose recovery over 3 seeds")


def test_6a_bp_beats_signatures():
    """BP pose consistency must add accuracy beyond the identity channel."""
    so = np.mean([run_6a(seed=s, noise=0.03, iters=0, verbose=False)["accuracy"]
                  for s in range(5)])
    bp = np.mean([run_6a(seed=s, noise=0.03, iters=40, verbose=False)["accuracy"]
                  for s in range(5)])
    assert bp > so + 0.10, f"BP {bp:.3f} vs sig-only {so:.3f}"
    print(f"PASS 6a ablation: BP {bp:.3f} > signature-only {so:.3f} at 3x noise")


def test_6b_calibration():
    r = run_6b(seed=0, verbose=False)
    assert r["entropy_ambiguous_mean"] > 0.80, r
    assert r["entropy_final"] < 0.15, r
    assert r["acc_final"] == 1.0, r
    assert r["ece"] < 0.15, r
    print(f"PASS 6b: entropy {r['entropy_ambiguous_mean']:.2f} bits while "
          f"evidence absent -> {r['entropy_final']:.2f} after; acc 1.0; "
          f"ECE {r['ece']:.3f}")


def test_6c_permanence():
    r = run_6c(seed=0, verbose=False)
    assert r["cov_growth_during_occlusion"] > 2.0, r
    assert r["rebind_accuracy"] == 1.0, r
    assert r["centroid_pred_err_at_emergence"] < 0.05, r
    print(f"PASS 6c: covariance grew x{r['cov_growth_during_occlusion']:.1f} "
          f"in the dark; predicted emergence within "
          f"{r['centroid_pred_err_at_emergence']:.3f}; re-binding 1.0")


if __name__ == "__main__":
    test_pose_vote_inverts_action()
    test_signature_invariance()
    test_fit_sim2()
    test_6a_binding()
    test_6a_bp_beats_signatures()
    test_6b_calibration()
    test_6c_permanence()
    print("\nall tests pass")
