#!/usr/bin/env python3
"""
BET 6 — Belief propagation object binding in Gabor packet space.
=================================================================

"objects that know what they know": objects are pose BELIEFS (Gaussians on
the Lie algebra of Sim(2)) plus intrinsic templates; atoms carry categorical
ASSIGNMENT beliefs. Binding, borders, and permanence are inference, not
architecture. The BP posture is Tindall et al. (arXiv:2503.05693) with the
physics removed: cheap local message passing as the engine, honesty about
where the graph makes it lie (double counting on loops -> cavity messages).

The Slapstack edge over capsules/GLOM-style routing-by-agreement: the
part->whole pose vote is CLOSED-FORM GROUP ALGEBRA on atom parameters
(verified exactly equivariant in Bet 5), not a learned transform. An atom's
Sim(2)-invariant signature (cycles-per-envelope su*f, aspect su/sv,
envelope-relative phase, color) identifies it; the group part poses it.

Experiments (all CPU, seconds each):

  6a  STATIC BINDING.  K known templates hidden in a scene at unknown poses
      with parameter noise and clutter atoms. Loopy BP over correspondence
      beliefs b_i(k,j) and pose beliefs b(g_k). GO: assignments recovered,
      poses recovered, clutter rejected to the outlier class.

  6b  COMMON FATE + CALIBRATION.  Two spatially interleaved atom clouds, no
      templates, no appearance cues. Motions are IDENTICAL for the first
      frames (evidence genuinely absent), then diverge. Online BP/EM from
      motion alone. GO: binding accuracy high AFTER divergence, and belief
      entropy is CALIBRATED: high during the ambiguous window, dropping
      when evidence arrives; low expected calibration error (ECE).

  6c  PERMANENCE BY INFERENCE.  Object A passes fully behind an occluder.
      Its pose belief coasts on a constant-velocity prior with honestly
      GROWING covariance, then re-binds its atoms on re-emergence. This is
      the probabilistic upgrade of Bet 5: not "the tractor cannot move"
      (frozen tensor) but "the system believes the tractor is still there,
      and knows how well it knows".

Usage:
  python bet6_bp_binding.py --exp all --out runs/bet6
  python tests.py            # assertion battery
"""

import argparse
import json
import math
import os

import numpy as np

# ----------------------------------------------------------------------------
# Sim(2) utilities.  Pose xi = (tx, ty, rho, lam), action: x' = e^lam R(rho) x + t
# Gaussian beliefs live in these coordinates (Euclidean approximation of the
# Lie algebra — honest small-covariance approximation, noted in README).
# ----------------------------------------------------------------------------

def rot(rho):
    c, s = math.cos(rho), math.sin(rho)
    return np.array([[c, -s], [s, c]])


def apply_pose(xi, xy):
    tx, ty, rho, lam = xi
    return math.exp(lam) * xy @ rot(rho).T + np.array([tx, ty])


def wrap_half_pi(d):
    """Gabor orientation is pi-periodic (atom(theta+pi, phi) == atom(theta, -phi)),
    so orientation differences are only defined mod pi. Wrap to [-pi/2, pi/2)."""
    return (d + math.pi / 2) % math.pi - math.pi / 2


# ----------------------------------------------------------------------------
# Atoms.  Structured array: xy(2), theta, su, sv, f, phase, color(3)
# ----------------------------------------------------------------------------

FIELDS = ["x", "y", "theta", "su", "sv", "f", "phase", "r", "g", "b"]


def make_template(n, rng):
    """Random template with DISTINCT Sim(2)-invariant signatures per atom."""
    a = np.zeros((n, len(FIELDS)))
    a[:, 0:2] = rng.uniform(-0.35, 0.35, (n, 2))            # canonical xy
    a[:, 2] = rng.uniform(-math.pi / 2, math.pi / 2, n)     # theta
    a[:, 3] = rng.uniform(0.04, 0.12, n)                    # su
    a[:, 4] = a[:, 3] * rng.uniform(0.4, 0.9, n)            # sv (aspect < 1)
    a[:, 5] = rng.uniform(4.0, 14.0, n)                     # f
    a[:, 6] = rng.uniform(0, 2 * math.pi, n)                # envelope-relative phase
    a[:, 7:10] = rng.uniform(0, 1, (n, 3))                  # color
    return a


def transform_atoms(atoms, xi):
    """Exact Sim(2) action on atom parameters (the Bet 5 algebra):
    xy -> s R xy + t, theta -> theta + rho, sigma -> s sigma, f -> f/s.
    Envelope-relative phase and color are INVARIANT."""
    out = atoms.copy()
    s = math.exp(xi[3])
    out[:, 0:2] = apply_pose(xi, atoms[:, 0:2])
    out[:, 2] = atoms[:, 2] + xi[2]
    out[:, 3:5] = atoms[:, 3:5] * s
    out[:, 5] = atoms[:, 5] / s
    return out


def signature(atoms):
    """Sim(2)-invariant intrinsic signature: (log(su*f), log aspect,
    cos/sin phase, color). Identity lives here; pose cannot touch it."""
    return np.stack([
        np.log(atoms[:, 3] * atoms[:, 5]),          # cycles per envelope width
        np.log(atoms[:, 3] / atoms[:, 4]),          # aspect
        np.cos(atoms[:, 6]), np.sin(atoms[:, 6]),   # phase (periodic-safe)
        atoms[:, 7], atoms[:, 8], atoms[:, 9],      # color
    ], axis=1)


def pose_vote(obs, tmpl):
    """CLOSED-FORM part->whole vote: the unique xi mapping template atom ->
    observed atom. Scale from three independent scale-carrying channels
    (su, sv, 1/f), rotation from theta (mod pi), translation from xy."""
    s = (obs[3] / tmpl[3] * obs[4] / tmpl[4] * tmpl[5] / obs[5]) ** (1 / 3)
    rho = wrap_half_pi(obs[2] - tmpl[2])
    t = obs[0:2] - s * rot(rho) @ tmpl[0:2]
    return np.array([t[0], t[1], rho, math.log(s)])


# ----------------------------------------------------------------------------
# Weighted Sim(2) fit (Umeyama) — closed-form M-step for motion binding
# ----------------------------------------------------------------------------

def fit_sim2(X, Y, w):
    """argmin_{s,R,t} sum_i w_i ||Y_i - (s R X_i + t)||^2. Returns xi."""
    w = w / (w.sum() + 1e-12)
    mx, my = w @ X, w @ Y
    Xc, Yc = X - mx, Y - my
    C = (w[:, None] * Yc).T @ Xc
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, d])
    Rm = U @ D @ Vt
    var_x = (w @ (Xc ** 2).sum(1)) + 1e-12
    s = np.trace(np.diag(S) @ D) / var_x
    t = my - s * Rm @ mx
    rho = math.atan2(Rm[1, 0], Rm[0, 0])
    return np.array([t[0], t[1], rho, math.log(max(s, 1e-9))])


# ----------------------------------------------------------------------------
# Experiment 6a — static template binding via loopy BP with cavity messages
# ----------------------------------------------------------------------------

def run_6a(seed=0, K=3, n_per=12, n_clutter=10, noise=0.01, iters=40,
           damping=0.5, cavity=True, out=None, verbose=True):
    rng = np.random.default_rng(seed)
    templates = [make_template(n_per, rng) for _ in range(K)]
    true_xi = [np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5),
                         rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)])
               for _ in range(K)]

    obs_list, gt = [], []
    for k in range(K):
        o = transform_atoms(templates[k], true_xi[k])
        o += rng.normal(0, noise, o.shape) * np.array(
            [1, 1, 1, .3, .3, 20, 3, .5, .5, .5])   # per-field noise scaling
        o[:, 3] = np.maximum(o[:, 3], 0.012)         # keep su, sv, f physical
        o[:, 4] = np.maximum(o[:, 4], 0.008)
        o[:, 5] = np.maximum(o[:, 5], 0.5)
        obs_list.append(o)
        gt += [k] * n_per
    clutter = make_template(n_clutter, rng)
    clutter[:, 0:2] = rng.uniform(-1, 1, (n_clutter, 2))
    obs_list.append(clutter)
    gt += [-1] * n_clutter
    obs = np.vstack(obs_list)
    gt = np.array(gt)
    N = len(obs)

    # correspondence candidates by invariant signature (identity channel)
    sig_o = signature(obs)
    sig_t = [signature(t) for t in templates]
    SIG_VAR = 0.05
    cands = []      # per atom: list of (k, j, sig_loglik)
    for i in range(N):
        c = []
        for k in range(K):
            d2 = ((sig_t[k] - sig_o[i]) ** 2).sum(1)
            for j in np.argsort(d2)[:3]:
                c.append((k, int(j), -0.5 * d2[j] / SIG_VAR))
        cands.append(c)

    OUT_LL = -12.0                              # outlier class log-likelihood
    V = np.diag([0.03, 0.03, 0.05, 0.05]) ** 2  # pose-vote noise covariance
    Vinv = np.linalg.inv(V)
    P0inv = np.diag([1e-2] * 4)                 # broad pose prior precision

    # beliefs: per atom, weights over candidates + outlier (last slot).
    # Initialize from the IDENTITY channel (invariant signatures) — uniform
    # init averages correct and incorrect pose votes into a garbage mean and
    # loopy BP converges into the wrong basin (seed-dependent failures).
    B = []
    for i in range(N):
        ll0 = np.array([sig_ll for (_, _, sig_ll) in cands[i]] + [OUT_LL])
        b0 = np.exp(ll0 - ll0.max()); b0 /= b0.sum()
        B.append(b0)
    votes = [[pose_vote(obs[i], templates[k][j]) for (k, j, _) in cands[i]]
             for i in range(N)]

    mu = [np.zeros(4) for _ in range(K)]
    for it in range(iters):
        # ---- atom -> object: precision-weighted fusion of pose votes ----
        Lam = [P0inv.copy() for _ in range(K)]
        eta = [np.zeros(4) for _ in range(K)]
        for i in range(N):
            for c_idx, (k, j, _) in enumerate(cands[i]):
                w = B[i][c_idx]
                Lam[k] = Lam[k] + w * Vinv
                eta[k] = eta[k] + w * (Vinv @ votes[i][c_idx])
        Sig = [np.linalg.inv(L) for L in Lam]
        mu = [Sig[k] @ eta[k] for k in range(K)]

        # ---- object -> atom: likelihood under (cavity) pose belief ----
        newB = []
        for i in range(N):
            ll = np.empty(len(cands[i]) + 1)
            for c_idx, (k, j, sig_ll) in enumerate(cands[i]):
                if cavity:  # remove atom i's own vote: no self-confirmation
                    L_c = Lam[k] - B[i][c_idx] * Vinv
                    e_c = eta[k] - B[i][c_idx] * (Vinv @ votes[i][c_idx])
                    S_c = np.linalg.inv(L_c)
                    m_c, S_k = S_c @ e_c, S_c
                else:
                    m_c, S_k = mu[k], Sig[k]
                r = votes[i][c_idx] - m_c
                Cov = S_k + V
                ll[c_idx] = (sig_ll - 0.5 * r @ np.linalg.solve(Cov, r)
                             - 0.5 * math.log(np.linalg.det(Cov)))
            ll[-1] = OUT_LL
            b = np.exp(ll - ll.max()); b /= b.sum()
            newB.append(damping * b + (1 - damping) * B[i])
        B = newB

    # ---- read out ----
    pred = np.empty(N, dtype=int)
    conf = np.empty(N)
    for i in range(N):
        pk = np.zeros(K + 1)
        for c_idx, (k, j, _) in enumerate(cands[i]):
            pk[k] += B[i][c_idx]
        pk[K] = B[i][-1]
        pred[i] = np.argmax(pk) if np.argmax(pk) < K else -1
        conf[i] = pk.max()
    acc = (pred == gt).mean()
    pose_err = float(np.mean([np.abs(mu[k] - true_xi[k]).max() for k in range(K)]))
    res = {"exp": "6a", "seed": seed, "accuracy": float(acc),
           "pose_err_max_abs": pose_err, "n_atoms": int(N),
           "cavity": cavity,
           "clutter_rejected": float((pred[gt == -1] == -1).mean())}
    if verbose:
        print(f"[6a seed {seed}] accuracy {acc:.3f}  pose_err {pose_err:.4f}  "
              f"clutter_rejected {res['clutter_rejected']:.2f}")
    if out:
        plot_6a(obs, gt, pred, conf, mu, K, os.path.join(out, "6a_binding.png"))
    return res


# ----------------------------------------------------------------------------
# Experiments 6b/6c — common-fate binding, calibration, occlusion permanence
# ----------------------------------------------------------------------------

def make_motion_world(seed=0, n_per=20, T=12, noise=0.006, ambiguous_until=6):
    """Two spatially INTERLEAVED clouds. Identical motion until frame
    `ambiguous_until` (evidence absent by construction), divergent after."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * math.pi, n_per)
    A0 = np.stack([0.35 * np.cos(ang), 0.35 * np.sin(ang)], 1) \
        + rng.normal(0, 0.05, (n_per, 2))
    B0 = np.stack([0.25 * np.cos(ang), 0.25 * np.sin(ang)], 1) \
        + rng.normal(0, 0.08, (n_per, 2))          # interleaved with A

    def step_xi(t, obj):
        if t < ambiguous_until or obj == "A":
            return np.array([0.05, 0.0, 0.0, 0.0])          # common drift
        return np.array([0.0, -0.05, 0.06, 0.0])            # B diverges

    XA, XB = [A0], [B0]
    for t in range(T - 1):
        XA.append(apply_pose(step_xi(t, "A"), XA[-1]))
        XB.append(apply_pose(step_xi(t, "B"), XB[-1]))
    XA = np.array(XA) + rng.normal(0, noise, (T, n_per, 2))
    XB = np.array(XB) + rng.normal(0, noise, (T, n_per, 2))
    X = np.concatenate([XA, XB], axis=1)     # (T, 2n, 2), tracks persistent
    gt = np.array([0] * n_per + [1] * n_per)
    return X, gt, ambiguous_until


def run_6b(seed=0, T=12, n_per=20, noise=0.006, n_anchor=3, em_iters=8,
           out=None, verbose=True):
    X, gt, amb = make_motion_world(seed, n_per, T, noise)
    N = X.shape[1]
    SIG2 = (2.5 * noise) ** 2 * 2       # residual variance, honestly inflated
    EPS = 1e-3                          # uniform mixture floor (anti-hubris)

    # anchors: n_anchor known atoms per object (a user click, in effect)
    L = np.zeros((N, 2))                # cumulative log-likelihoods
    anchor = np.full(N, -1)
    anchor[:n_anchor] = 0
    anchor[n_per:n_per + n_anchor] = 1

    def beliefs():
        b = np.exp(L - L.max(1, keepdims=True)); b /= b.sum(1, keepdims=True)
        b = (1 - EPS) * b + EPS / 2
        for i in range(N):
            if anchor[i] >= 0:
                b[i] = [1.0 - EPS, EPS] if anchor[i] == 0 else [EPS, 1.0 - EPS]
        return b

    ent_t, acc_t, snapshots = [], [], []
    for t in range(T - 1):
        for _ in range(em_iters):
            b = beliefs()
            xi = [fit_sim2(X[t], X[t + 1], b[:, k]) for k in range(2)]
            step_ll = np.stack([
                -((X[t + 1] - apply_pose(xi[k], X[t])) ** 2).sum(1) / (2 * SIG2)
                for k in range(2)], axis=1)
        L = L + step_ll                          # evidence accumulates
        b = beliefs()
        H = float(np.mean(-(b * np.log2(b)).sum(1)))
        acc = float((b.argmax(1) == gt).mean())
        ent_t.append(H); acc_t.append(acc)
        snapshots.append((b.max(1).copy(), (b.argmax(1) == gt).copy()))
        if verbose:
            tag = "AMBIGUOUS" if t < amb - 1 else "divergent"
            print(f"[6b t={t:2d} {tag:9s}] entropy {H:.3f} bits  acc {acc:.3f}")

    # calibration over all (atom, time) snapshots
    confs = np.concatenate([s[0] for s in snapshots])
    corrs = np.concatenate([s[1] for s in snapshots])
    bins = np.linspace(0.5, 1.0, 11)
    ece, rel = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs >= lo) & (confs < hi)
        if m.sum() > 0:
            ece += m.mean() * abs(corrs[m].mean() - confs[m].mean())
            rel.append((float((lo + hi) / 2), float(corrs[m].mean()),
                        int(m.sum())))
    res = {"exp": "6b", "seed": seed,
           "entropy_ambiguous_mean": float(np.mean(ent_t[:amb - 1])),
           "entropy_final": float(ent_t[-1]),
           "acc_final": float(acc_t[-1]), "ece": float(ece),
           "reliability": rel}
    if verbose:
        print(f"[6b] H(ambiguous) {res['entropy_ambiguous_mean']:.3f} -> "
              f"H(final) {res['entropy_final']:.3f} bits | "
              f"acc {res['acc_final']:.3f} | ECE {ece:.3f}")
    if out:
        plot_6b(ent_t, acc_t, amb, rel, os.path.join(out, "6b_calibration.png"))
        gif_6b(X, gt, T, seed, n_per, noise, os.path.join(out, "6b_binding.gif"))
    return res


def run_6c(seed=0, T=16, n_per=20, noise=0.006, occl=(5, 10), em_iters=8,
           out=None, verbose=True):
    """Object A fully occluded for frames occl[0]..occl[1]-1. Constant-velocity
    coasting with growing covariance; re-binding on emergence."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * math.pi, n_per)
    A0 = np.stack([0.3 * np.cos(ang) - 0.6, 0.3 * np.sin(ang)], 1)
    B0 = rng.uniform(-0.25, 0.25, (n_per, 2)) + np.array([0.0, -0.7])
    vA = np.array([0.09, 0.0, 0.0, 0.0])         # A crosses the scene
    vB = np.array([0.0, 0.02, 0.0, 0.0])
    XA, XB = [A0], [B0]
    for t in range(T - 1):
        XA.append(apply_pose(vA, XA[-1]))
        XB.append(apply_pose(vB, XB[-1]))
    XA = np.array(XA) + rng.normal(0, noise, (T, n_per, 2))
    XB = np.array(XB) + rng.normal(0, noise, (T, n_per, 2))
    visible_A = np.array([not (occl[0] <= t < occl[1]) for t in range(T)])

    Q = np.diag([2e-4, 2e-4, 1e-4, 1e-4])        # process noise per frame
    Rn = np.diag([1e-4] * 4)                     # fit-observation noise
    xi_vel, P = np.zeros(4), np.diag([1e-2] * 4)   # A's velocity belief
    cov_tr, pred_err = [], None
    centroidA_pred = XA[occl[0] - 1].mean(0)

    for t in range(T - 1):
        if visible_A[t] and visible_A[t + 1]:
            z = fit_sim2(XA[t], XA[t + 1], np.ones(n_per))   # observed step
            S = P + Q + Rn
            Kg = (P + Q) @ np.linalg.inv(S)
            xi_vel = xi_vel + Kg @ (z - xi_vel)
            P = (np.eye(4) - Kg) @ (P + Q)
        else:
            P = P + Q                                        # coast: honesty grows
            centroidA_pred = apply_pose(xi_vel, centroidA_pred[None])[0]
        cov_tr.append(float(np.trace(P)))

    # re-emergence: predicted vs actual centroid, and re-binding accuracy
    pred_err = float(np.linalg.norm(centroidA_pred - XA[occl[1]].mean(0)))
    # associate re-appeared atoms: step-likelihood under A's coasted motion
    # vs B's motion vs outlier, using the frame pair after emergence
    t0 = occl[1]
    zB = fit_sim2(XB[t0], XB[t0 + 1], np.ones(n_per))
    SIG2 = (3 * noise) ** 2 * 2
    allX0 = np.concatenate([XA[t0], XB[t0]])
    allX1 = np.concatenate([XA[t0 + 1], XB[t0 + 1]])
    llA = -((allX1 - apply_pose(xi_vel, allX0)) ** 2).sum(1) / (2 * SIG2)
    llB = -((allX1 - apply_pose(zB, allX0)) ** 2).sum(1) / (2 * SIG2)
    rebind = np.where(llA > llB, 0, 1)
    gt = np.array([0] * n_per + [1] * n_per)
    rebind_acc = float((rebind == gt).mean())

    grow = cov_tr[occl[1] - 2] / cov_tr[occl[0] - 1]
    res = {"exp": "6c", "seed": seed, "cov_growth_during_occlusion": float(grow),
           "centroid_pred_err_at_emergence": pred_err,
           "rebind_accuracy": rebind_acc, "occl_frames": list(occl)}
    if verbose:
        print(f"[6c] cov trace grew x{grow:.1f} during occlusion | "
              f"centroid prediction error at emergence {pred_err:.4f} | "
              f"re-binding accuracy {rebind_acc:.3f}")
    if out:
        plot_6c(cov_tr, occl, os.path.join(out, "6c_permanence.png"))
    return res


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def plot_6a(obs, gt, pred, conf, mu, K, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#d43a2f", "#2f7ed4", "#3fae5a", "#999999"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    for a, lab, lbl in ((ax[0], gt, "ground truth"), (ax[1], pred, "BP binding")):
        for k in list(range(K)) + [-1]:
            m = lab == k
            a.scatter(obs[m, 0], obs[m, 1], s=28,
                      c=cols[k if k >= 0 else -1],
                      alpha=0.9 if k >= 0 else 0.45,
                      label=f"obj {k}" if k >= 0 else "clutter")
        a.set_title(lbl); a.set_aspect("equal"); a.set_xlim(-1.1, 1.1)
        a.set_ylim(-1.1, 1.1)
    for k in range(K):
        ax[1].scatter(*mu[k][:2], marker="x", c="k", s=60)
    ax[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("6a: template binding by pose-vote BP (x = inferred pose center)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_6b(ent_t, acc_t, amb, rel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    t = np.arange(len(ent_t))
    ax[0].axvspan(-0.5, amb - 1.5, color="#f4d9a0", alpha=0.5,
                  label="identical motion\n(evidence absent)")
    ax[0].plot(t, ent_t, "o-", label="belief entropy (bits)")
    ax[0].plot(t, acc_t, "s--", label="binding accuracy")
    ax[0].set_xlabel("frame"); ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(fontsize=7); ax[0].set_title("knowing what it knows")
    if rel:
        c, a, _ = zip(*rel)
        ax[1].plot([0.5, 1], [0.5, 1], "k--", lw=1, label="perfect calibration")
        ax[1].plot(c, a, "o-", label="observed")
        ax[1].set_xlabel("confidence"); ax[1].set_ylabel("accuracy")
        ax[1].legend(fontsize=7); ax[1].set_title("reliability")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def gif_6b(X, gt, T, seed, n_per, noise, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    # rerun online BP capturing per-frame beliefs (cheap)
    frames = []
    res_frames = _replay_6b_beliefs(X, gt, n_per, noise)
    for t, b in res_frames:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        rgb = np.stack([b[:, 0], np.zeros(len(b)), b[:, 1]], 1)
        ax.scatter(X[t][:, 0], X[t][:, 1], c=np.clip(rgb, 0, 1), s=45)
        ax.set_xlim(-1.2, 1.6); ax.set_ylim(-1.6, 1.2)
        ax.set_title(f"frame {t}: red=belief obj0, blue=obj1, purple=unsure")
        ax.set_aspect("equal")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(buf.copy()))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=350, loop=0)


def _replay_6b_beliefs(X, gt, n_per, noise, n_anchor=3, em_iters=8):
    N = X.shape[1]; T = X.shape[0]
    SIG2 = (2.5 * noise) ** 2 * 2; EPS = 1e-3
    L = np.zeros((N, 2)); anchor = np.full(N, -1)
    anchor[:n_anchor] = 0; anchor[n_per:n_per + n_anchor] = 1
    outp = []

    def beliefs():
        b = np.exp(L - L.max(1, keepdims=True)); b /= b.sum(1, keepdims=True)
        b = (1 - EPS) * b + EPS / 2
        for i in range(N):
            if anchor[i] >= 0:
                b[i] = [1 - EPS, EPS] if anchor[i] == 0 else [EPS, 1 - EPS]
        return b

    for t in range(T - 1):
        for _ in range(em_iters):
            b = beliefs()
            xi = [fit_sim2(X[t], X[t + 1], b[:, k]) for k in range(2)]
            step_ll = np.stack([
                -((X[t + 1] - apply_pose(xi[k], X[t])) ** 2).sum(1) / (2 * SIG2)
                for k in range(2)], 1)
        L = L + step_ll
        outp.append((t + 1, beliefs()))
    return outp


def plot_6c(cov_tr, occl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.axvspan(occl[0] - 1, occl[1] - 1, color="#cccccc", alpha=0.6,
               label="occluded")
    ax.semilogy(cov_tr, "o-")
    ax.set_xlabel("frame"); ax.set_ylabel("tr(pose covariance)")
    ax.set_title("6c: honesty grows in the dark, collapses on re-emergence")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="all", choices=["all", "6a", "6b", "6c"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=5, help="6a accuracy trials")
    p.add_argument("--out", default="runs/bet6")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ledger = []
    if args.exp in ("all", "6a"):
        accs = []
        for s in range(args.trials):
            r = run_6a(seed=args.seed + s, out=args.out if s == 0 else None)
            accs.append(r["accuracy"]); ledger.append(r)
        print(f"[6a] accuracy over {args.trials} seeds: "
              f"{np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    if args.exp in ("all", "6b"):
        ledger.append(run_6b(seed=args.seed, out=args.out))
    if args.exp in ("all", "6c"):
        ledger.append(run_6c(seed=args.seed, out=args.out))
    with open(os.path.join(args.out, "ledger.json"), "w") as fh:
        json.dump(ledger, fh, indent=2)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
