#!/usr/bin/env python3
"""
Open problem 4 — the pi-ambiguity, and pose beliefs that can hold two minds.
=============================================================================

The physics: a Gabor atom obeys  atom(theta + pi, phi) == atom(theta, -phi).
Orientation alone is therefore only defined mod pi, which is why Bet 6's
first cut wrapped rotation differences to [-pi/2, pi/2) and restricted test
poses. But rotating a pose vote by pi FLIPS the translation estimate
(R(rho+pi) = -R(rho)), so the ambiguity is not cosmetic — unresolved, it
breaks binding for any rotation beyond +/- 90 degrees.

The fix is already in the representation: ENVELOPE-RELATIVE PHASE. The two
hypotheses are distinguishable because the flip negates phase:

    H0:  rho = d_theta,        requires  phi_obs ==  phi_tmpl
    H1:  rho = d_theta + pi,   requires  phi_obs == -phi_tmpl

Each correspondence therefore emits TWO pose votes, each scored by its
phase consistency. Generic phases (away from 0 and pi, where phi == -phi)
kill the wrong hypothesis outright — full SO(2) rotation recovery, exactly.
Degenerate phases leave both hypotheses alive, and then the pose belief
MUST be allowed to hold two modes. This module upgrades the object's pose
belief from a single Gaussian to a mixture, initialized by density peaks in
vote space and refined by EM.

The satisfying part: a 180-degree-symmetric object with degenerate phases
is GENUINELY pose-ambiguous — there is no fact of the matter mod pi — and
the mixture should report two modes at ~equal weight. Calibrated ambiguity,
the 6b story transported from assignment space to pose space. Breaking the
symmetry with a single generic phase should collapse the mixture.
"""

import math

import numpy as np

from bet6_bp_binding import make_template, transform_atoms, signature, rot


def wrap_pi(d):
    return (d + math.pi) % (2 * math.pi) - math.pi


def pose_votes_2pi(obs, tmpl, sig_phase=0.35):
    """Two pose-vote hypotheses per correspondence, each with its phase-
    consistency log-likelihood. Resolves rotation mod 2*pi, not mod pi."""
    s = (obs[3] / tmpl[3] * obs[4] / tmpl[4] * tmpl[5] / obs[5]) ** (1 / 3)
    d_theta = obs[2] - tmpl[2]
    out = []
    for H in (0, 1):
        rho = wrap_pi(d_theta + H * math.pi)
        phi_expected = tmpl[6] if H == 0 else -tmpl[6]
        dphi = wrap_pi(obs[6] - phi_expected)
        pc = -0.5 * dphi ** 2 / sig_phase ** 2
        t = obs[0:2] - s * rot(rho) @ tmpl[0:2]
        out.append((np.array([t[0], t[1], rho, math.log(s)]), pc))
    return out


def _wrap_to(ref_rho, xi):
    """Re-express a vote's rho on the branch nearest a reference angle so
    Gaussian algebra on the circle stays honest near a mode."""
    out = xi.copy()
    out[2] = ref_rho + wrap_pi(xi[2] - ref_rho)
    return out


def _density_peaks(votes, weights, M, radius=0.45):
    """Greedy mode-seeking init: pick the highest-weighted-density vote,
    suppress its neighborhood (angle-aware distance), repeat."""
    scale = np.array([0.15, 0.15, 0.30, 0.20])
    V = np.array(votes)
    W = np.array(weights)
    dens = np.zeros(len(V))
    for i in range(len(V)):
        d = (V - V[i]) / scale
        d[:, 2] = wrap_pi(V[:, 2] - V[i, 2]) / scale[2]
        dens[i] = (W * np.exp(-0.5 * (d ** 2).sum(1))).sum()
    peaks, alive = [], np.ones(len(V), bool)
    for _ in range(M):
        if not alive.any():
            break
        i = int(np.argmax(np.where(alive, dens, -np.inf)))
        peaks.append(V[i].copy())
        d = (V - V[i]) / scale
        d[:, 2] = wrap_pi(V[:, 2] - V[i, 2]) / scale[2]
        alive &= (d ** 2).sum(1) > radius ** 2 / scale.mean() ** 2 * 0  # keep simple:
        alive &= np.sqrt(((V[:, :2] - V[i, :2]) ** 2).sum(1)) + \
            np.abs(wrap_pi(V[:, 2] - V[i, 2])) > radius
    return peaks


def mixture_bind(template, obs, M=2, iters=30, sig_var=0.05, out_ll=-14.0):
    """Bind observed atoms to ONE template whose pose belief is a mixture of
    M Gaussians on sim(2). Returns modes (weight, mu) sorted by weight, and
    per-candidate responsibilities. EM over {correspondence, hypothesis,
    mode}; the pose-vote noise V is fixed (same register as Bet 6 BP)."""
    sig_t = signature(template)
    sig_o = signature(obs)
    V = np.diag([0.03, 0.03, 0.05, 0.05]) ** 2
    Vinv = np.linalg.inv(V)

    cands, votes, base_ll = [], [], []
    for i in range(len(obs)):
        d2 = ((sig_t - sig_o[i]) ** 2).sum(1)
        for j in np.argsort(d2)[:3]:
            for (xi, pc) in pose_votes_2pi(obs[i], template[int(j)]):
                cands.append((i, int(j)))
                votes.append(xi)
                base_ll.append(-0.5 * d2[int(j)] / sig_var + pc)
    votes = np.array(votes)
    base_ll = np.array(base_ll)
    w0 = np.exp(base_ll - base_ll.max())

    mus = _density_peaks(list(votes), w0, M)
    M = len(mus)
    pis = np.full(M, 1.0 / M)

    for _ in range(iters):
        # E-step: responsibilities over modes x candidates (+outlier mass)
        logr = np.empty((len(votes), M))
        for m in range(M):
            dv = votes - mus[m]
            dv[:, 2] = wrap_pi(votes[:, 2] - mus[m][2])
            Cov = V * 2
            logr[:, m] = (math.log(pis[m] + 1e-12) + base_ll
                          - 0.5 * np.einsum("ni,ij,nj->n", dv,
                                            np.linalg.inv(Cov), dv))
        lse = np.logaddexp.reduce(np.column_stack([logr, 
                                                   np.full((len(votes), 1),
                                                           out_ll)]), axis=1)
        R = np.exp(logr - lse[:, None])          # (n_votes, M)
        # one-vote-per-atom normalization: an atom cannot vote twice
        atom_ids = np.array([c[0] for c in cands])
        for i in np.unique(atom_ids):
            m_i = atom_ids == i
            tot = R[m_i].sum()
            if tot > 1.0:
                R[m_i] /= tot
        # M-step
        for m in range(M):
            w = R[:, m]
            if w.sum() < 1e-8:
                continue
            v_adj = np.array([_wrap_to(mus[m][2], v) for v in votes])
            mus[m] = (w[:, None] * v_adj).sum(0) / w.sum()
            mus[m][2] = wrap_pi(mus[m][2])
        pis = R.sum(0) / max(R.sum(), 1e-12)
        pis = pis / pis.sum()

    # --- final mode weights: proper posterior, not responsibility mass ---
    # EM's pi measures how many votes each mode explains, which under-reacts
    # to a single decisive atom (evidence should MULTIPLY across atoms).
    # Score each located mode by the product over atoms of that atom's best
    # explanation under the mode (logsumexp over its candidates + outlier).
    atom_ids = np.array([c[0] for c in cands])
    post = np.zeros(M)
    for m in range(M):
        dv = votes - mus[m]
        dv[:, 2] = wrap_pi(votes[:, 2] - mus[m][2])
        Cov = V * 2
        cll = base_ll - 0.5 * np.einsum("ni,ij,nj->n", dv,
                                        np.linalg.inv(Cov), dv)
        total = 0.0
        for i in np.unique(atom_ids):
            per_atom = np.append(cll[atom_ids == i], out_ll)
            total += np.logaddexp.reduce(per_atom)
        post[m] = total
    post = np.exp(post - post.max())
    post = post / post.sum()

    order = np.argsort(-post)
    return [(float(post[m]), mus[m]) for m in order], (cands, votes, R)


# ----------------------------------------------------------------------------
# Stress worlds
# ----------------------------------------------------------------------------

def symmetric_template(n_half=5, seed=0, break_phase=None):
    """180-degree-symmetric template: atom pairs at (x, theta) and (-x, theta)
    with phase 0 (the degenerate value where phi == -phi). Rotating the
    object by pi maps the atom set onto itself -> pose genuinely ambiguous
    mod pi. `break_phase` gives ONE atom a generic phase, breaking it."""
    rng = np.random.default_rng(seed)
    half = make_template(n_half, rng)
    half[:, 6] = 0.0                                  # degenerate phases
    mirror = half.copy()
    mirror[:, 0:2] *= -1                              # antipodal positions
    tmpl = np.vstack([half, mirror])
    if break_phase is not None:
        tmpl[0, 6] = break_phase                      # one generic phase
    return tmpl


def run_symmetry_demo(rho_true=2.5, noise=0.004, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    xi_true = np.array([0.2, -0.15, rho_true, 0.1])
    results = {}
    for name, tmpl in (("symmetric", symmetric_template(seed=seed)),
                       ("broken", symmetric_template(seed=seed,
                                                     break_phase=1.3))):
        obs = transform_atoms(tmpl, xi_true)
        obs += rng.normal(0, noise, obs.shape) * np.array(
            [1, 1, 1, .3, .3, 20, 3, .5, .5, .5])
        obs[:, 3] = np.maximum(obs[:, 3], 0.012)
        obs[:, 4] = np.maximum(obs[:, 4], 0.008)
        obs[:, 5] = np.maximum(obs[:, 5], 0.5)
        modes, _ = mixture_bind(tmpl, obs, M=2)
        results[name] = modes
        if verbose:
            desc = ", ".join(f"pi={w:.2f} rho={mu[2]:+.3f}" for w, mu in modes)
            print(f"[{name:9s}] modes: {desc}   (true rho {rho_true:+.3f}, "
                  f"alias {wrap_pi(rho_true - math.pi):+.3f})")
    return results, xi_true


if __name__ == "__main__":
    run_symmetry_demo()
