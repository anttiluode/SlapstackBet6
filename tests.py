#!/usr/bin/env python3
"""
Open problem 3 — Stable Diffusion as an oracle node with BELIEF-weighted
gradient masks (GPU; NOT executed in the build environment — smoke-test me).
==========================================================================

Bet 5b edited "which atoms receive gradients" with a hand-declared rectangle
(hard 0/1 group mask). This script replaces the rectangle with the OUTPUT OF
INFERENCE: soft per-atom weights w_i in [0, 1] exported from Bet 6 BP
binding (bet6_open.export_belief_weights). The SDS gradient on every
per-atom parameter is scaled by w_i, so:

    w_i ~ 0   atom is confidently the protected object  -> untouchable
    w_i ~ 1   atom is confidently background/other      -> fully editable
    w_i ~ 0.5 the belief is genuinely unsure            -> half-strength
              edits, exactly at the soft border of the ownership field

This is the graphical-model division of labor made executable: BP knows
WHAT IS WHERE and HOW SURE, SD knows WHAT THINGS LOOK LIKE. An edit is a
conditioned inference, and the protection of the object is proportional to
the belief that it is the object.

Deployment shape (once repos are merged):
  1. bet5 recon or sds run       ->  runs/<scene>/atoms.pt
  2. bet6 bp_bind on those atoms ->  belief weights .npy (model index space)
  3. this script                 ->  belief-masked SDS edit

Example:
  python bet6d_sds_oracle.py \
      --init-atoms runs/recon_tractor/atoms.pt \
      --weights runs/belief_weights_tractor.npy \
      --prompt "a tractor on a beach in miami, ocean, sand, blue sky" \
      --iters 1500 --render-size 512 --cfg 50 --no-camera \
      --out runs/bet6d_beach_soft

Honesty: the loop below is Bet 5's verified-on-GPU SDS loop; the only new
mechanics are (a) loading float weights and (b) passing them where Bet 5
passed a hard mask — bet5.apply_grad_masks already multiplies grads by a
float tensor, so soft masking needs no new numerics. Still: first run is a
smoke test. The known open risk from Bet 5 carries over (SD 2.1
mode-seeking oversaturation).
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from bet5_gabor_sds import (load_atoms, resolve_frozen, make_optimizer,
                            apply_grad_masks, sample_camera, save_png,
                            geometry_fingerprint)


def run(args):
    from diffusers import StableDiffusionPipeline, DDPMScheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = load_atoms(args.init_atoms)

    w = np.load(args.weights).astype(np.float32)
    assert len(w) == model.n_atoms, \
        f"weights length {len(w)} != model atoms {model.n_atoms}"
    assert 0.0 <= w.min() and w.max() <= 1.0, "weights must be in [0,1]"
    mask = torch.from_numpy(w)
    print(f"belief mask: {(w > 0.9).sum()} editable, {(w < 0.1).sum()} "
          f"protected, {((w >= 0.1) & (w <= 0.9)).sum()} soft-border atoms")

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
    model.train().to(device)
    fp0 = geometry_fingerprint(model)

    os.makedirs(args.out, exist_ok=True)
    log, t0 = [], time.time()
    for it in range(args.iters):
        opt.zero_grad()
        cam = None if args.no_camera else sample_camera(args)
        img = model.render(args.render_size, args.render_size, device,
                           chunk=args.chunk, camera=cam)
        x = img[None] * 2 - 1
        if args.render_size != 512:
            x = F.interpolate(x, (512, 512), mode="bilinear",
                              align_corners=False)
        latents = vae.encode(x.to(dtype)).latent_dist.sample() \
            * vae.config.scaling_factor
        latents = latents.float()

        frac = it / max(1, args.iters - 1)
        t_max = args.t_max_start + (args.t_max_end - args.t_max_start) * frac
        t = torch.randint(int(args.t_min * T), int(t_max * T), (1,),
                          device=device)
        noise = torch.randn_like(latents)
        noisy = sched.add_noise(latents, noise, t)
        with torch.no_grad():
            eps = unet(torch.cat([noisy] * 2).to(dtype), torch.cat([t] * 2),
                       encoder_hidden_states=emb).sample.float()
            eps_un, eps_tx = eps.chunk(2)
            eps_hat = eps_un + args.cfg * (eps_tx - eps_un)
        wgt = (1 - alphas[t]).view(-1, 1, 1, 1)
        grad = (wgt * (eps_hat - noise)).detach()
        sds_loss = (grad * latents).sum() / latents.numel()
        loss = sds_loss + args.l0_weight * model.gates.l0().sum() / model.n_atoms
        loss.backward()
        if it < args.gate_warmup and model.gates.logits.grad is not None:
            model.gates.logits.grad.zero_()
        apply_grad_masks(model, mask, device)     # <-- soft belief mask
        torch.nn.utils.clip_grad_norm_(
            [p for g_ in opt.param_groups for p in g_["params"]], 1.0)
        opt.step()

        if it % max(1, args.iters // 30) == 0 or it == args.iters - 1:
            row = {"it": it, "sds": float(sds_loss.item()), "t_max": t_max,
                   **model.ledger()}
            log.append(row)
            print(f"[oracle] it {it:5d}  sds {sds_loss.item():+.4f}  "
                  f"t_max {t_max:.2f}  ({time.time()-t0:.0f}s)")
            with torch.no_grad():
                save_png(model.render(args.render_size, args.render_size,
                                      device, chunk=args.chunk,
                                      hard_gates=True),
                         os.path.join(args.out, f"it_{it:05d}.png"))

    model.eval()
    with torch.no_grad():
        save_png(model.render(args.render_size, args.render_size, device,
                              chunk=args.chunk, hard_gates=True),
                 os.path.join(args.out, "final_hardgates.png"))
    torch.save(model.state_dict(), os.path.join(args.out, "atoms.pt"))
    fp1 = geometry_fingerprint(model)
    ledger = {"mode": "sds_oracle", "prompt": args.prompt,
              "weights": args.weights, "freeze": args.freeze,
              "mask_stats": {"editable": int((w > 0.9).sum()),
                             "protected": int((w < 0.1).sum()),
                             "soft": int(((w >= 0.1) & (w <= 0.9)).sum())},
              "geometry_fp_before": fp0, "geometry_fp_after": fp1,
              "note": ("Whole-model fingerprint SHOULD change here (editable "
                       "atoms train). Per-atom protection is enforced by the "
                       "soft mask; verify visually that the protected object "
                       "is intact."),
              "final": model.ledger(), "log": log}
    with open(os.path.join(args.out, "ledger.json"), "w") as fh:
        json.dump(ledger, fh, indent=2)
    print(f"done -> {args.out}")


def main():
    p = argparse.ArgumentParser(description="belief-weighted SDS oracle")
    p.add_argument("--init-atoms", required=True)
    p.add_argument("--weights", required=True,
                   help=".npy soft weights in model index space, [0,1]")
    p.add_argument("--out", default="runs/bet6d")
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--render-size", type=int, default=512)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--l0-weight", type=float, default=5e-3)
    p.add_argument("--gate-warmup", type=int, default=10 ** 9,
                   help="default: gates never pruned during an edit")
    p.add_argument("--freeze", default="gates")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--cam-zoom", type=float, default=0.30)
    p.add_argument("--cam-shift", type=float, default=0.25)
    p.add_argument("--cam-rot", type=float, default=0.15)
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="blurry, low quality, deformed")
    p.add_argument("--sd-model", default="sd2-community/stable-diffusion-2-1-base")
    p.add_argument("--cfg", type=float, default=50.0)
    p.add_argument("--t-min", type=float, default=0.02)
    p.add_argument("--t-max-start", type=float, default=0.98)
    p.add_argument("--t-max-end", type=float, default=0.50)
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
