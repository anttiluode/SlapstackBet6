#!/usr/bin/env python3
"""
Open problems 1 + 2 — real atoms as templates, borders as belief fields.
=========================================================================

#1 REAL ATOMS. Bet 6's binding ran on synthetic templates with conveniently
distinct signatures. Here the template library comes from actual Slapstack
reconstructions: two toy scenes (a tractor, a star) are fit with the Bet 5
torch model (hard-concrete gates, envelope-relative phase), the surviving
atoms are extracted, and BP must find both objects hidden in a cluttered
scene at unknown poses — including rotations far beyond +/-90 degrees,
which the pi-ambiguity fix (bet6_multimodal.pose_votes_2pi) now permits.
Real atoms have whatever signatures optimization gave them, collisions
included. That is the point.

#2 BORDERS. "Bounded entities with borders": the border is not drawn, it is
the decision boundary of the ownership field

    P(k | pixel)  ~  sum_i  b_i(k) * envelope_i(pixel)

computed through the atoms' ACTUAL Gabor envelopes. Soft where beliefs are
uncertain, sharp where they are confident — and this is quantified: mean
pixel-ownership entropy must fall as observation noise falls.

Also exports soft per-atom belief weights in Bet-5 model index space, the
input format for bet6d_sds_oracle.py (open problem #3).
"""

import math
import os

import numpy as np

from bet6_bp_binding import make_template, transform_atoms, signature, rot
from bet6_multimodal import pose_votes_2pi, wrap_pi, _density_peaks

FIELDS = ["x", "y", "theta", "su", "sv", "f", "phase", "r", "g", "b"]


# ----------------------------------------------------------------------------
# Template library from real Slapstack reconstructions (Bet 5 torch model)
# ----------------------------------------------------------------------------

def _draw_tractor(px=96):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (px, px), (40, 80, 160))
    d = ImageDraw.Draw(im)
    d.rectangle([15, 68, px, px], fill=(210, 180, 110))
    d.rectangle([26, 41, 64, 64], fill=(200, 40, 30))
    d.rectangle([45, 26, 64, 41], fill=(180, 30, 25))
    d.ellipse([22, 56, 45, 79], fill=(20, 20, 20))
    d.ellipse([54, 62, 69, 77], fill=(25, 25, 25))
    return im


def _draw_star(px=96):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (px, px), (25, 25, 60))
    d = ImageDraw.Draw(im)
    c, r1, r2 = px / 2, px * 0.42, px * 0.17
    pts = []
    for i in range(10):
        r = r1 if i % 2 == 0 else r2
        a = math.pi / 2 + i * math.pi / 5
        pts.append((c + r * math.cos(a), c - r * math.sin(a)))
    d.polygon(pts, fill=(240, 200, 60))
    d.ellipse([c - 9, c - 9, c + 9, c + 9], fill=(200, 90, 30))
    return im


def _train_recon(pil_img, n_atoms=96, iters=350, size=96, seed=0):
    import torch
    import torch.nn.functional as F
    from bet5_gabor_sds import GaborPacketImage
    dev = torch.device("cpu")
    tgt = torch.from_numpy(np.asarray(pil_img.resize((size, size)),
                                      dtype=np.float32).copy()
                           ).permute(2, 0, 1) / 255.0
    m = GaborPacketImage(n_atoms, seed=seed)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for it in range(iters):
        opt.zero_grad()
        loss = F.mse_loss(m.render(size, size, dev), tgt) \
            + 1e-3 * m.gates.l0().sum() / n_atoms
        loss.backward()
        if it < 60 and m.gates.logits.grad is not None:
            m.gates.logits.grad.zero_()
        opt.step()
    return m


def atoms_from_model(m):
    """Extract hard-open atoms into the Bet 6 field layout. Amplitude is
    folded into color (signed). xy is centered -> canonical template frame.
    Returns (template array, model-space indices of the kept atoms,
    model.n_atoms)."""
    import torch
    keep = np.where(m.gates.hard_open().numpy())[0]
    arr = np.zeros((len(keep), len(FIELDS)))
    arr[:, 0:2] = torch.tanh(m.xy_raw).detach().numpy()[keep]
    arr[:, 0:2] -= arr[:, 0:2].mean(0)
    arr[:, 2] = m.theta.detach().numpy()[keep]
    arr[:, 3] = np.clip(np.exp(m.log_sigma_u.detach().numpy()[keep]),
                        5e-3, 2.0)
    arr[:, 4] = np.clip(np.exp(m.log_sigma_v.detach().numpy()[keep]),
                        5e-3, 2.0)
    arr[:, 5] = np.log1p(np.exp(m.freq_raw.detach().numpy()[keep]))
    arr[:, 6] = np.mod(m.phase.detach().numpy()[keep], 2 * math.pi)
    amp = m.amp.detach().numpy()[keep]
    arr[:, 7:10] = m.color.detach().numpy()[keep] * amp[:, None]
    return arr, keep, m.n_atoms


def build_templates(cache="runs/templates.npz", force=False, verbose=True):
    if (not force) and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return [z["tractor"], z["star"]], z["meta"].item()
    if verbose:
        print("training template reconstructions (CPU, ~30 s)...")
    mA = _train_recon(_draw_tractor(), seed=0)
    mB = _train_recon(_draw_star(), seed=1)
    tA, keepA, nA = atoms_from_model(mA)
    tB, keepB, nB = atoms_from_model(mB)
    meta = {"keep_tractor": keepA, "n_model_tractor": nA,
            "keep_star": keepB, "n_model_star": nB}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, tractor=tA, star=tB, meta=meta)
    if verbose:
        print(f"templates: tractor {len(tA)} atoms, star {len(tB)} atoms")
    return [tA, tB], meta


# ----------------------------------------------------------------------------
# General BP binding with 2pi votes (upgrade of Bet 6's run_6a loop)
# ----------------------------------------------------------------------------

def bp_bind(templates, obs, iters=40, damping=0.5, cavity=True,
            sig_var=0.08, out_ll=-14.0):
    """Loopy BP binding of obs atoms to K templates at unknown poses.
    Candidates carry BOTH pi-hypotheses (phase-scored), so rotations are
    unrestricted. Returns per-atom object marginals (N, K+1), pose means,
    and internals."""
    K = len(templates)
    sig_t = [signature(t) for t in templates]
    sig_o = signature(obs)
    N = len(obs)
    V = np.diag([0.03, 0.03, 0.05, 0.05]) ** 2
    Vinv = np.linalg.inv(V)
    P0inv = np.diag([1e-2] * 4)

    cands, votes, base = [], [], []
    for i in range(N):
        c, v, b = [], [], []
        for k in range(K):
            d2 = ((sig_t[k] - sig_o[i]) ** 2).sum(1)
            for j in np.argsort(d2)[:3]:
                for (xi, pc) in pose_votes_2pi(obs[i], templates[k][int(j)]):
                    c.append((k, int(j)))
                    v.append(xi)
                    b.append(-0.5 * d2[int(j)] / sig_var + pc)
        cands.append(c); votes.append(v); base.append(np.array(b))

    # beliefs seeded from the identity+phase channel (Bet 6 lesson)
    B = []
    for i in range(N):
        ll0 = np.append(base[i], out_ll)
        b0 = np.exp(ll0 - ll0.max()); b0 /= b0.sum()
        B.append(b0)

    # pose init: density peak of each object's votes (angle-aware)
    mu = []
    for k in range(K):
        vk, wk = [], []
        for i in range(N):
            for c_idx, (kk, j) in enumerate(cands[i]):
                if kk == k:
                    vk.append(votes[i][c_idx]); wk.append(B[i][c_idx])
        mu.append(_density_peaks(vk, np.array(wk), 1)[0])

    for it in range(iters):
        Lam = [P0inv.copy() for _ in range(K)]
        eta = [np.zeros(4) for _ in range(K)]
        for i in range(N):
            for c_idx, (k, j) in enumerate(cands[i]):
                v = votes[i][c_idx].copy()
                v[2] = mu[k][2] + wrap_pi(v[2] - mu[k][2])   # branch-align
                w = B[i][c_idx]
                Lam[k] = Lam[k] + w * Vinv
                eta[k] = eta[k] + w * (Vinv @ v)
        Sig = [np.linalg.inv(L) for L in Lam]
        mu = [Sig[k] @ eta[k] for k in range(K)]
        for k in range(K):
            mu[k][2] = wrap_pi(mu[k][2])

        newB = []
        for i in range(N):
            ll = np.empty(len(cands[i]) + 1)
            for c_idx, (k, j) in enumerate(cands[i]):
                v = votes[i][c_idx].copy()
                v[2] = mu[k][2] + wrap_pi(v[2] - mu[k][2])
                if cavity:
                    L_c = Lam[k] - B[i][c_idx] * Vinv
                    e_c = eta[k] - B[i][c_idx] * (Vinv @ v)
                    S_c = np.linalg.inv(L_c)
                    m_c, S_k = S_c @ e_c, S_c
                else:
                    m_c, S_k = mu[k], Sig[k]
                r = v - m_c
                r[2] = wrap_pi(v[2] - m_c[2])
                Cov = S_k + V
                ll[c_idx] = (base[i][c_idx]
                             - 0.5 * r @ np.linalg.solve(Cov, r)
                             - 0.5 * math.log(np.linalg.det(Cov)))
            ll[-1] = out_ll
            b = np.exp(ll - ll.max()); b /= b.sum()
            newB.append(damping * b + (1 - damping) * B[i])
        B = newB

    marg = np.zeros((N, K + 1))
    for i in range(N):
        for c_idx, (k, j) in enumerate(cands[i]):
            marg[i, k] += B[i][c_idx]
        marg[i, K] = B[i][-1]
    return marg, mu, (cands, votes, B)


# ----------------------------------------------------------------------------
# Scene, experiment, ownership rendering
# ----------------------------------------------------------------------------

def make_scene(templates, xis, n_clutter=15, noise=0.006, seed=0):
    rng = np.random.default_rng(seed)
    parts, gt = [], []
    for k, (t, xi) in enumerate(zip(templates, xis)):
        parts.append(transform_atoms(t, np.array(xi)))
        gt += [k] * len(t)
    cl = make_template(n_clutter, rng)
    cl[:, 0:2] = rng.uniform(-0.95, 0.95, (n_clutter, 2))
    parts.append(cl); gt += [-1] * n_clutter
    obs = np.vstack(parts)
    obs += rng.normal(0, noise, obs.shape) * np.array(
        [1, 1, 1, .3, .3, 20, 3, .5, .5, .5])
    obs[:, 3] = np.maximum(obs[:, 3], 0.012)
    obs[:, 4] = np.maximum(obs[:, 4], 0.008)
    obs[:, 5] = np.maximum(obs[:, 5], 0.5)
    return obs, np.array(gt)


def render_scene(obs, H=220):
    """Numpy Gabor render of the observed atoms (visualization only)."""
    ys = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(ys, ys)
    pre = np.zeros((3, H, H))
    for a in obs:
        dx, dy = X - a[0], Y - a[1]
        u = math.cos(a[2]) * dx + math.sin(a[2]) * dy
        v = -math.sin(a[2]) * dx + math.cos(a[2]) * dy
        env = np.exp(-0.5 * ((u / a[3]) ** 2 + (v / a[4]) ** 2))
        car = np.cos(2 * math.pi * a[5] * u + a[6])
        for c in range(3):
            pre[c] += a[7 + c] * env * car
    return 1 / (1 + np.exp(-2.0 * pre))


def ownership_field(obs, marg, K, H=220):
    """P(k | pixel) through the atoms' actual Gabor envelopes. The border is
    the decision boundary of a belief field, not a drawn line."""
    ys = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(ys, ys)
    O = np.zeros((K + 1, H, H))
    for i, a in enumerate(obs):
        dx, dy = X - a[0], Y - a[1]
        u = math.cos(a[2]) * dx + math.sin(a[2]) * dy
        v = -math.sin(a[2]) * dx + math.cos(a[2]) * dy
        env = np.exp(-0.5 * ((u / a[3]) ** 2 + (v / a[4]) ** 2))
        energy = float(np.linalg.norm(a[7:10]))   # amplitude-weighted evidence:
        for k in range(K + 1):                    # louder atoms claim harder
            O[k] += marg[i, k] * energy * env
    tot = O.sum(0)
    P = O / (tot + 1e-6)
    support = tot > 0.05
    ent = -(P * np.log2(P + 1e-12)).sum(0)
    return P, ent, support


def run_real_atoms(noise=0.006, seed=0, out=None, verbose=True):
    templates, meta = build_templates(verbose=verbose)
    xis = [(0.25, -0.20, 2.0, 0.12),      # tractor: rotation WAY past pi/2
           (-0.40, 0.35, -1.2, -0.15)]    # star
    obs, gt = make_scene(templates, xis, noise=noise, seed=seed)
    marg, mu, _ = bp_bind(templates, obs)
    pred = np.where(marg.argmax(1) < len(templates), marg.argmax(1), -1)
    acc = float((pred == gt).mean())
    perr = float(max(
        max(abs(mu[k][0] - xis[k][0]), abs(mu[k][1] - xis[k][1]),
            abs(wrap_pi(mu[k][2] - xis[k][2])), abs(mu[k][3] - xis[k][3]))
        for k in range(2)))
    res = {"exp": "real_atoms", "noise": noise, "seed": seed,
           "n_template_atoms": [int(len(t)) for t in templates],
           "accuracy": acc, "pose_err_max_abs": perr,
           "rotations_true": [xis[0][2], xis[1][2]]}
    if verbose:
        print(f"[real atoms, noise {noise}] accuracy {acc:.3f}  "
              f"pose_err {perr:.4f}  (rotations {xis[0][2]}, {xis[1][2]} rad)")
    P, ent, support = ownership_field(obs, marg, len(templates))
    res["mean_ownership_entropy"] = float(ent[support].mean())
    if out:
        plot_borders(obs, P, ent, support, marg, gt,
                     os.path.join(out, f"borders_noise{noise}.png"))
    return res, (obs, marg, P, ent, support, meta)


def plot_borders(obs, P, ent, support, marg, gt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = P.shape[1]
    scene = render_scene(obs, H)
    K = P.shape[0] - 1
    cols = np.array([[0.85, 0.23, 0.18], [0.18, 0.45, 0.83], [0.6, 0.6, 0.6]])
    own = np.einsum("khw,kc->hwc", P, cols)
    own[~support] = 1.0
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.2))
    ax[0].imshow(scene.transpose(1, 2, 0), extent=[-1, 1, 1, -1])
    ax[0].set_title("scene (rendered from observed atoms)")
    ax[1].imshow(np.clip(own, 0, 1), extent=[-1, 1, 1, -1])
    ax[1].set_title("ownership field P(object | pixel)")
    im = ax[2].imshow(np.where(support, ent, np.nan),
                      extent=[-1, 1, 1, -1], cmap="magma")
    ax[2].set_title("border = high-entropy ridge of the belief field")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def export_belief_weights(meta, marg, k_obj, mode="edit_background",
                          path="runs/belief_weights_tractor.npy"):
    """Soft per-atom weights in Bet-5 MODEL index space for the tractor
    model. mode='edit_background': weight = 1 - b(object) so SDS gradients
    flow AROUND the believed object (bet6d_sds_oracle.py input)."""
    n_model = int(meta["n_model_tractor"])
    keep = meta["keep_tractor"]
    b_obj = marg[:len(keep), k_obj]           # tractor atoms lead the scene
    w = np.zeros(n_model)
    w[keep] = (1.0 - b_obj) if mode == "edit_background" else b_obj
    np.save(path, w)
    return w


if __name__ == "__main__":
    os.makedirs("runs/bet6_open", exist_ok=True)
    res_lo, pack = run_real_atoms(noise=0.006, out="runs/bet6_open")
    res_hi, _ = run_real_atoms(noise=0.06, out="runs/bet6_open")
    print(f"border softness (mean ownership entropy): "
          f"low-noise {res_lo['mean_ownership_entropy']:.3f}  vs  "
          f"high-noise {res_hi['mean_ownership_entropy']:.3f}")
    w = export_belief_weights(pack[5], pack[1], k_obj=0)
    print(f"exported soft weights for bet6d oracle: "
          f"{(w > 0.5).sum()} background-editable atoms of {len(w)}")
