#!/usr/bin/env python3
"""
BET 5 — Object permanence in Gabor packet space.
=================================================

Question: if we FREEZE the geometry channels of an existing atom set and let
SDS gradients touch only appearance (amp / color / phase / background), does
the object HOLD its shape while the scene relights?

    "at dusk, it turns to dusk, the tractor stays."

This is the empirical test of the fiber-bundle claim: illumination acts on
the amplitude/chroma fibers; geometry (x, y, theta, sigma, f) is the base
space and should be untouchable by a relighting edit.

Carries forward everything verified in Bet 4 (GO: recognizable tractor from
256 atoms under pure SDS) plus the fixes that run demanded:

  * SOFT clamp        pre = 4*tanh(pre/4)   -- no zero-gradient dead zones
                      (Bet 4's hard clamp fixed saturation but killed the
                      escape gradient; this keeps the corridor AND the slope)
  * Random Sim(2) cameras -- the DreamFusion trick, free in this
                      representation: zoom/shift/rotate are parameter
                      arithmetic on atoms. Kills the SDS zoom-crop trap:
                      only a complete, centered object scores well under
                      every random view.
  * Normalized SDS loss + gate warmup -- in Bet 4 the SDS loss dwarfed the
                      L0 term so gates never closed (256/256 all run). Loss
                      is now per-element normalized and gates get a warmup
                      before pruning pressure engages.
  * --init-atoms      load any previous atoms.pt (Bet 4 recon or SDS,
                      N inferred from the file)
  * --freeze          channel groups excluded from optimization entirely
  * --train-groups    per-atom masks: only listed groups receive gradients
                      (the two-slot tractor/background experiment is a flag)

Modes
-----
recon  : fit atoms to a target image (MSE). No diffusion model needed.
sds    : score distillation from frozen Stable Diffusion.
render : load atoms and render -- identity view, a chosen camera, or a
         camera sweep saved as GIF (the "glide" demo).

The permanence experiment (the actual bet)
------------------------------------------
  # 1. you already have runs/recon_tractor/atoms.pt  (23.5 dB, 205 atoms)
  # 2. relight it, geometry untouchable:
  python bet5_gabor_sds.py --mode sds \
      --init-atoms runs/recon_tractor/atoms.pt \
      --freeze geometry,gates \
      --prompt "a photo of a red tractor at dusk, golden hour, warm light" \
      --iters 1500 --render-size 512 --cfg 50 \
      --sd-model sd2-community/stable-diffusion-2-1-base \
      --out runs/bet5_dusk

GO  : final image reads as the SAME tractor, relit. Geometry channels are
      bitwise identical (the script verifies and prints this).
NO-GO: appearance channels alone cannot express the edit (tractor holds by
      construction -- geometry is frozen -- but the scene refuses to read
      as dusk after a seed retry and a cfg bump).

Honesty note: recon/render/freeze/group/camera machinery executed and
verified on CPU before shipping. The sds path reuses Bet 4's verified-on-
your-GPU loop with the fixes above; the fixes themselves have not run under
a real SDS gradient yet. First run = smoke test.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Hard-concrete gates (Louizos et al.) — Slapstack two-way doors
# ----------------------------------------------------------------------------

class HardConcreteGates(nn.Module):
    GAMMA, ZETA, BETA = -0.1, 1.1, 2.0 / 3.0

    def __init__(self, n, init_logit=2.0):
        super().__init__()
        self.logits = nn.Parameter(torch.full((n,), float(init_logit)))

    def forward(self, hard_eval=False):
        if self.training and not hard_eval:
            u = torch.rand_like(self.logits).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.logits) / self.BETA)
        else:
            s = torch.sigmoid(self.logits)
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)

    def l0(self):
        return torch.sigmoid(self.logits - self.BETA * math.log(-self.GAMMA / self.ZETA))

    @torch.no_grad()
    def hard_open(self):
        z = torch.sigmoid(self.logits) * (self.ZETA - self.GAMMA) + self.GAMMA
        return (z.clamp(0, 1) > 0.5)


# ----------------------------------------------------------------------------
# Differentiable Gabor packet image (Bet 4 renderer + soft clamp + cameras)
# ----------------------------------------------------------------------------

class GaborPacketImage(nn.Module):
    """
    Image = sigmoid( softclamp( bg_bias + sum_i g_i a_i c_i env_i carrier_i ) )

      env_i     = exp(-0.5 (u^2/su^2 + v^2/sv^2)), (u,v) atom-local rotated coords
      carrier_i = cos(2*pi*f_i*u + phi_i)     phi is ENVELOPE-RELATIVE:
                                              pose never touches phase.
    A camera g = (s, rho, tx, ty) in Sim(2) acts by parameter arithmetic:
      xy -> s*R(rho)*xy + t,  theta -> theta+rho,  sigma -> s*sigma,  f -> f/s
    phase and color are untouched — that IS the identity/pose factorization.
    """

    # per-atom parameter names (used by freeze / group-mask machinery)
    PER_ATOM = ["xy_raw", "theta", "log_sigma_u", "log_sigma_v",
                "freq_raw", "phase", "amp", "color", "gates.logits"]

    def __init__(self, n_atoms=256, coarse_frac=0.25, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        n = n_atoms
        nc = int(n * coarse_frac)

        self.xy_raw = nn.Parameter(torch.randn(n, 2, generator=g) * 0.7)
        self.theta = nn.Parameter(torch.rand(n, generator=g) * math.pi)

        log_s = torch.empty(n)
        log_s[:nc] = math.log(0.45) + 0.25 * torch.randn(nc, generator=g)
        log_s[nc:] = math.log(0.12) + 0.35 * torch.randn(n - nc, generator=g)
        self.log_sigma_u = nn.Parameter(log_s.clone())
        self.log_sigma_v = nn.Parameter(log_s + 0.2 * torch.randn(n, generator=g))

        f = torch.empty(n)
        f[:nc] = 0.25 + 0.5 * torch.rand(nc, generator=g)
        f[nc:] = 0.75 + 2.0 * torch.rand(n - nc, generator=g)
        self.freq_raw = nn.Parameter(torch.log(torch.expm1(f)))

        self.phase = nn.Parameter(2 * math.pi * torch.rand(n, generator=g))
        self.amp = nn.Parameter(0.35 + 0.15 * torch.randn(n, generator=g))
        self.color = nn.Parameter(0.30 * torch.randn(n, 3, generator=g))
        self.bg_bias = nn.Parameter(torch.zeros(3))

        self.gates = HardConcreteGates(n)
        # per-atom group id (0 = default). Buffer so it persists in atoms.pt.
        self.register_buffer("group", torch.zeros(n, dtype=torch.long))
        self.n_atoms = n

    # -- derived ---------------------------------------------------------------
    def xy(self):
        return torch.tanh(self.xy_raw)

    def freq(self):
        return F.softplus(self.freq_raw)

    def sigmas(self):
        return (self.log_sigma_u.exp().clamp(5e-3, 2.0),
                self.log_sigma_v.exp().clamp(5e-3, 2.0))

    # -- render ------------------------------------------------------------------
    def render(self, H, W, device, chunk=64, hard_gates=False, camera=None):
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")

        xy = self.xy().to(device)
        theta = self.theta.to(device)
        su, sv = self.sigmas()
        su, sv = su.to(device), sv.to(device)
        f = self.freq().to(device)
        phi = self.phase.to(device)
        amp = self.amp.to(device)
        col = self.color.to(device)
        z = self.gates(hard_eval=hard_gates).to(device)

        if camera is not None:                       # Sim(2): parameter arithmetic
            s, rho, tx, ty = camera
            c, sn = math.cos(rho), math.sin(rho)
            R = torch.tensor([[c, -sn], [sn, c]], device=device, dtype=xy.dtype)
            xy = s * xy @ R.T + torch.tensor([tx, ty], device=device, dtype=xy.dtype)
            theta = theta + rho
            su, sv = s * su, s * sv
            f = f / s
            # phase untouched by construction — envelope-relative.

        pre = torch.zeros(3, H, W, device=device) + self.bg_bias.to(device)[:, None, None]

        for i0 in range(0, self.n_atoms, chunk):
            sl = slice(i0, min(i0 + chunk, self.n_atoms))
            dx = X[None] - xy[sl, 0, None, None]
            dy = Y[None] - xy[sl, 1, None, None]
            ct = torch.cos(theta[sl])[:, None, None]
            st = torch.sin(theta[sl])[:, None, None]
            u = ct * dx + st * dy
            v = -st * dx + ct * dy
            env = torch.exp(-0.5 * ((u / su[sl, None, None]) ** 2 +
                                    (v / sv[sl, None, None]) ** 2))
            carrier = torch.cos(2 * math.pi * f[sl, None, None] * u + phi[sl, None, None])
            w = (z[sl] * amp[sl])[:, None, None] * env * carrier
            pre = pre + torch.einsum("nhw,nc->chw", w, col[sl])

        # Leaky soft clamp: tanh corridor + linear leak. Plain tanh still dies
        # numerically in fp32 at |pre|>~40 (caught by shipped test 5); the
        # 0.02*pre leak guarantees a nonzero escape gradient at ANY depth of
        # saturation — the solid-red trap always has an exit ramp.
        pre = 4.0 * torch.tanh(pre / 4.0) + 0.02 * pre
        return torch.sigmoid(pre)

    def ledger(self):
        return {"atoms_total": self.n_atoms,
                "atoms_open_hard": int(self.gates.hard_open().sum().item()),
                "expected_L0": float(self.gates.l0().sum().item()),
                "groups": {int(k): int(v) for k, v in
                           zip(*[t.tolist() for t in self.group.unique(return_counts=True)])}}


# ----------------------------------------------------------------------------
# Loading, freezing, group masks
# ----------------------------------------------------------------------------

FREEZE_MAP = {
    "position":    ["xy_raw"],
    "orientation": ["theta"],
    "scale":       ["log_sigma_u", "log_sigma_v"],
    "frequency":   ["freq_raw"],
    "phase":       ["phase"],
    "amp":         ["amp"],
    "color":       ["color"],
    "bg":          ["bg_bias"],
    "gates":       ["gates.logits"],
}
FREEZE_MAP["geometry"] = (FREEZE_MAP["position"] + FREEZE_MAP["orientation"]
                          + FREEZE_MAP["scale"] + FREEZE_MAP["frequency"])
FREEZE_MAP["appearance"] = (FREEZE_MAP["phase"] + FREEZE_MAP["amp"]
                            + FREEZE_MAP["color"] + FREEZE_MAP["bg"])


def load_atoms(path):
    sd = torch.load(path, map_location="cpu")
    n = sd["phase"].shape[0]
    model = GaborPacketImage(n_atoms=n)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # 'group' may be missing in Bet 4 files — that's fine (defaults to 0).
    assert not unexpected, f"unexpected keys in {path}: {unexpected}"
    print(f"loaded {path}: {n} atoms"
          + (f" (new buffers defaulted: {missing})" if missing else ""))
    return model


def resolve_frozen(freeze_arg):
    frozen = set()
    for tok in [t for t in (freeze_arg or "").split(",") if t.strip()]:
        tok = tok.strip()
        assert tok in FREEZE_MAP, f"unknown freeze channel '{tok}' " \
                                  f"(choose from {sorted(FREEZE_MAP)})"
        frozen.update(FREEZE_MAP[tok])
    return frozen


def make_optimizer(model, frozen, mode, lr_scale=1.0):
    # LRs that worked: recon = Bet 4 defaults; sds = the rebalance that
    # escaped the solid-red saturation trap (color 2e-3, bg 5e-4).
    lrs = {"xy_raw": 5e-3, "theta": 5e-3, "log_sigma_u": 5e-3,
           "log_sigma_v": 5e-3, "freq_raw": 5e-3, "phase": 2e-2,
           "amp": 1e-2, "color": 1e-2, "bg_bias": 1e-2, "gates.logits": 2e-2}
    if mode == "sds":
        lrs["color"], lrs["bg_bias"] = 2e-3, 5e-4
    groups, trained = [], []
    for name, lr in lrs.items():
        if name in frozen:
            continue
        p = dict(model.named_parameters())[name]
        groups.append({"params": [p], "lr": lr * lr_scale})
        trained.append(name)
    assert groups, "everything is frozen — nothing to optimize"
    print(f"training channels: {trained}")
    if frozen:
        print(f"frozen channels:   {sorted(frozen)}")
    return torch.optim.Adam(groups, betas=(0.9, 0.99))


def group_mask(model, train_groups):
    """None if all groups train; else float mask (n,) — 1 for trainable atoms."""
    if train_groups is None:
        return None
    ids = torch.tensor([int(t) for t in train_groups.split(",")])
    mask = torch.isin(model.group.cpu(), ids).float()
    print(f"group mask: {int(mask.sum())}/{model.n_atoms} atoms trainable "
          f"(groups {ids.tolist()})")
    return mask


def apply_grad_masks(model, mask, device):
    """Zero gradients of per-atom params for atoms outside trainable groups."""
    if mask is None:
        return
    m = mask.to(device)
    params = dict(model.named_parameters())
    for name in GaborPacketImage.PER_ATOM:
        p = params[name]
        if p.grad is not None:
            p.grad.mul_(m.view(-1, *([1] * (p.dim() - 1))))


def sample_camera(args):
    if args.no_camera:
        return None
    return (math.exp(random.uniform(-args.cam_zoom, args.cam_zoom)),
            random.uniform(-args.cam_rot, args.cam_rot),
            random.uniform(-args.cam_shift, args.cam_shift),
            random.uniform(-args.cam_shift, args.cam_shift))


def save_png(img_chw, path):
    from PIL import Image
    arr = (img_chw.detach().clamp(0, 1).cpu().numpy()
           .transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def geometry_fingerprint(model):
    """Hash of geometry channels — proves bitwise permanence after a run."""
    import hashlib
    h = hashlib.sha256()
    for name in FREEZE_MAP["geometry"]:
        h.update(dict(model.named_parameters())[name].detach().cpu()
                 .numpy().tobytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------
# Mode: recon
# ----------------------------------------------------------------------------

def run_recon(args, model, device):
    from PIL import Image
    tgt = Image.open(args.target).convert("RGB").resize((args.render_size,) * 2)
    target = torch.from_numpy(np.asarray(tgt).copy()).float().permute(2, 0, 1) / 255.0
    target = target.to(device)

    frozen = resolve_frozen(args.freeze)
    opt = make_optimizer(model, frozen, "recon")
    mask = group_mask(model, args.train_groups)
    model.train().to(device)

    os.makedirs(args.out, exist_ok=True)
    save_png(target, os.path.join(args.out, "target.png"))
    log, t0 = [], time.time()
    for it in range(args.iters):
        opt.zero_grad()
        img = model.render(args.render_size, args.render_size, device, chunk=args.chunk)
        mse = F.mse_loss(img, target)
        loss = mse + args.l0_weight * model.gates.l0().sum() / model.n_atoms
        loss.backward()
        if it < args.gate_warmup and model.gates.logits.grad is not None:
            model.gates.logits.grad.zero_()
        apply_grad_masks(model, mask, device)
        opt.step()
        if it % max(1, args.iters // 20) == 0 or it == args.iters - 1:
            psnr = -10 * math.log10(max(mse.item(), 1e-12))
            row = {"it": it, "mse": mse.item(), "psnr_db": psnr, **model.ledger()}
            log.append(row)
            print(f"[recon] it {it:5d}  mse {mse.item():.5f}  psnr {psnr:5.2f} dB  "
                  f"open {row['atoms_open_hard']}/{model.n_atoms}  ({time.time()-t0:.0f}s)")
            save_png(img, os.path.join(args.out, f"it_{it:05d}.png"))
    finish(model, args, device, log)


# ----------------------------------------------------------------------------
# Mode: sds
# ----------------------------------------------------------------------------

def run_sds(args, model, device):
    from diffusers import StableDiffusionPipeline, DDPMScheduler

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False)
    pipe.to(device)
    vae, unet, tok, te = pipe.vae, pipe.unet, pipe.tokenizer, pipe.text_encoder
    for m in (vae, unet, te):
        m.requires_grad_(False)
    sched = DDPMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")
    alphas = sched.alphas_cumprod.to(device)
    T = sched.config.num_train_timesteps

    def embed(text):
        ids = tok(text, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(device)
        return te(ids)[0]

    with torch.no_grad():
        emb = torch.cat([embed(args.negative_prompt), embed(args.prompt)])

    frozen = resolve_frozen(args.freeze)
    opt = make_optimizer(model, frozen, "sds")
    mask = group_mask(model, args.train_groups)
    model.train().to(device)
    fp_before = geometry_fingerprint(model)

    os.makedirs(args.out, exist_ok=True)
    log, t0 = [], time.time()
    for it in range(args.iters):
        opt.zero_grad()
        cam = sample_camera(args)
        img = model.render(args.render_size, args.render_size, device,
                           chunk=args.chunk, camera=cam)
        x = img[None] * 2 - 1
        if args.render_size != 512:
            x = F.interpolate(x, (512, 512), mode="bilinear", align_corners=False)

        latents = vae.encode(x.to(dtype)).latent_dist.sample() * vae.config.scaling_factor
        latents = latents.float()

        frac = it / max(1, args.iters - 1)
        t_max = args.t_max_start + (args.t_max_end - args.t_max_start) * frac
        t = torch.randint(int(args.t_min * T), int(t_max * T), (1,), device=device)

        noise = torch.randn_like(latents)
        noisy = sched.add_noise(latents, noise, t)
        with torch.no_grad():
            eps = unet(torch.cat([noisy] * 2).to(dtype), torch.cat([t] * 2),
                       encoder_hidden_states=emb).sample.float()
            eps_un, eps_tx = eps.chunk(2)
            eps_hat = eps_un + args.cfg * (eps_tx - eps_un)

        w = (1 - alphas[t]).view(-1, 1, 1, 1)
        grad = (w * (eps_hat - noise)).detach()
        sds_loss = (grad * latents).sum() / latents.numel()   # normalized (Bet 4 fix)
        l0_loss = model.gates.l0().sum() / model.n_atoms
        loss = sds_loss + args.l0_weight * l0_loss
        loss.backward()
        if it < args.gate_warmup and model.gates.logits.grad is not None:
            model.gates.logits.grad.zero_()
        apply_grad_masks(model, mask, device)
        torch.nn.utils.clip_grad_norm_(
            [p for g_ in opt.param_groups for p in g_["params"]], 1.0)
        opt.step()

        if it % max(1, args.iters // 30) == 0 or it == args.iters - 1:
            row = {"it": it, "sds": float(sds_loss.item()),
                   "l0": float(l0_loss.item()), "t_max": t_max, **model.ledger()}
            log.append(row)
            print(f"[sds] it {it:5d}  sds {sds_loss.item():+.4f}  t_max {t_max:.2f}  "
                  f"open {row['atoms_open_hard']}/{model.n_atoms}  ({time.time()-t0:.0f}s)")
            with torch.no_grad():
                save_png(model.render(args.render_size, args.render_size, device,
                                      chunk=args.chunk, hard_gates=True),
                         os.path.join(args.out, f"it_{it:05d}.png"))

    fp_after = geometry_fingerprint(model)
    if "geometry" in (args.freeze or ""):
        verdict = "IDENTICAL — permanence held by construction" \
            if fp_before == fp_after else "CHANGED — BUG, investigate"
        print(f"geometry fingerprint before/after: {fp_before} / {fp_after} -> {verdict}")
    finish(model, args, device, log,
           extra={"geometry_fp_before": fp_before, "geometry_fp_after": fp_after})


# ----------------------------------------------------------------------------
# Mode: render  (identity view, chosen camera, or a glide GIF)
# ----------------------------------------------------------------------------

def run_render(args, model, device):
    model.eval().to(device)
    os.makedirs(args.out, exist_ok=True)
    S = args.render_size
    with torch.no_grad():
        save_png(model.render(S, S, device, chunk=args.chunk, hard_gates=True),
                 os.path.join(args.out, "identity.png"))
        if args.camera:
            cam = tuple(float(v) for v in args.camera.split(","))
            assert len(cam) == 4, "--camera expects 's,rho,tx,ty'"
            save_png(model.render(S, S, device, chunk=args.chunk,
                                  hard_gates=True, camera=cam),
                     os.path.join(args.out, "camera.png"))
        if args.gif:
            from PIL import Image
            frames = []
            n = 40
            for i in range(n):
                p = i / (n - 1)
                cam = (1.0 + 0.15 * math.sin(2 * math.pi * p),   # gentle zoom breath
                       0.0,
                       -0.45 + 0.9 * p,                          # glide left -> right
                       0.10 * math.sin(4 * math.pi * p))         # slight bob
                im = model.render(S, S, device, chunk=args.chunk,
                                  hard_gates=True, camera=cam)
                frames.append(Image.fromarray(
                    (im.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255)
                    .astype(np.uint8)))
            frames[0].save(os.path.join(args.out, "glide.gif"), save_all=True,
                           append_images=frames[1:], duration=60, loop=0)
            print("wrote glide.gif — texture rides the envelopes; phase never moves")
    print(f"render -> {args.out}")


# ----------------------------------------------------------------------------

def finish(model, args, device, log, extra=None):
    model.eval()
    with torch.no_grad():
        img = model.render(args.render_size, args.render_size, device,
                           chunk=args.chunk, hard_gates=True)
    save_png(img, os.path.join(args.out, "final_hardgates.png"))
    torch.save(model.state_dict(), os.path.join(args.out, "atoms.pt"))
    ledger = {"mode": args.mode, "prompt": getattr(args, "prompt", None),
              "freeze": args.freeze, "train_groups": args.train_groups,
              "init_atoms": args.init_atoms,
              "camera": None if args.no_camera else
                        {"zoom": args.cam_zoom, "shift": args.cam_shift,
                         "rot": args.cam_rot},
              "final": model.ledger(), "log": log}
    if extra:
        ledger.update(extra)
    with open(os.path.join(args.out, "ledger.json"), "w") as fh:
        json.dump(ledger, fh, indent=2)
    print(f"done -> {args.out}  | open atoms: {model.ledger()['atoms_open_hard']}")


def assign_group_rect(model, spec):
    """--assign-group-rect 'x0,y0,x1,y1:gid' — atoms with canonical xy inside
    the rect get group gid. Coordinates in [-1,1]. Repeatable."""
    box, gid = spec.split(":")
    x0, y0, x1, y1 = (float(v) for v in box.split(","))
    xy = model.xy().detach()
    inside = ((xy[:, 0] >= x0) & (xy[:, 0] <= x1) &
              (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
    model.group[inside] = int(gid)
    print(f"assigned {int(inside.sum())} atoms in [{x0},{x1}]x[{y0},{y1}] "
          f"to group {gid}")


def main():
    p = argparse.ArgumentParser(description="Bet 5: permanence in Gabor packet space")
    p.add_argument("--mode", choices=["recon", "sds", "render"], required=True)
    p.add_argument("--out", default="runs/bet5")
    p.add_argument("--atoms", type=int, default=256)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--render-size", type=int, default=512)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--l0-weight", type=float, default=5e-3)
    p.add_argument("--gate-warmup", type=int, default=400,
                   help="iterations before L0 pruning pressure engages")
    # init / freeze / groups
    p.add_argument("--init-atoms", help="atoms.pt from any previous run (Bet 4 OK)")
    p.add_argument("--freeze", default="",
                   help=f"comma list from {sorted(FREEZE_MAP)}")
    p.add_argument("--train-groups",
                   help="comma list of group ids that receive gradients")
    p.add_argument("--assign-group-rect", action="append", default=[],
                   help="'x0,y0,x1,y1:gid' assign atoms in rect to group (repeatable)")
    # cameras
    p.add_argument("--no-camera", action="store_true",
                   help="disable random Sim(2) cameras in sds mode")
    p.add_argument("--cam-zoom", type=float, default=0.30, help="log-zoom range")
    p.add_argument("--cam-shift", type=float, default=0.25)
    p.add_argument("--cam-rot", type=float, default=0.15, help="radians")
    p.add_argument("--camera", help="render mode: fixed 's,rho,tx,ty'")
    p.add_argument("--gif", action="store_true", help="render mode: glide GIF")
    # recon
    p.add_argument("--target", help="target image (recon mode)")
    # sds
    p.add_argument("--prompt", default="a photo of a tractor")
    p.add_argument("--negative-prompt", default="blurry, low quality, deformed")
    p.add_argument("--sd-model", default="sd2-community/stable-diffusion-2-1-base")
    p.add_argument("--cfg", type=float, default=50.0)
    p.add_argument("--t-min", type=float, default=0.02)
    p.add_argument("--t-max-start", type=float, default=0.98)
    p.add_argument("--t-max-end", type=float, default=0.50)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = load_atoms(args.init_atoms) if args.init_atoms \
        else GaborPacketImage(args.atoms, seed=args.seed)
    for spec in args.assign_group_rect:
        assign_group_rect(model, spec)

    if args.mode == "recon":
        assert args.target, "--target required in recon mode"
        run_recon(args, model, device)
    elif args.mode == "sds":
        run_sds(args, model, device)
    else:
        run_render(args, model, device)


if __name__ == "__main__":
    main()
