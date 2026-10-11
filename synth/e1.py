"""
Per-coordinate synthetic experiment for STE with binary and ternary quantizers.

Per-coordinate quadratic loss L(q) = (h/2)(q - w*)^2, h = 1, w* Laplacian (E|w|=1) or Gaussian (sigma=1).
Latent w updated by STE with SGD or Adam. Quantizer binary or ternary.

Measures, over a final window after burn-in, per coordinate, next to closed-form values:
  - time-averaged q            (closed form: equals w*)
  - snapshot penalty (q_T-w*)^2 (closed form: (w*-a)(b-w*))
  - flip rate                  (closed form: 1-|w*|/s binary, 2 min(|w*|, s-|w*|)/s ternary)
  - total penalty vs scale s   (ternary minimum at 1.504 E|w| Laplacian, 1.276 Gaussian)
  - mode-of-trajectory rounding vs snapshot vs EMA-latent requantize
  - non-flipping fraction under SGD + weight decay

Usage: python synth/e1.py --out synth/results/e1 --device cuda:0
"""
import argparse, json, math, os, time
import torch

def laplace(n, g, dev):
    u = torch.rand(n, generator=g, device=dev) - 0.5
    return -torch.sign(u) * torch.log1p(-2 * u.abs())  # scale b=1, E|w|=1

def gauss(n, g, dev):
    return torch.randn(n, generator=g, device=dev)   # sigma=1, E|w| = sqrt(2/pi)

def quantize(w, s, kind, dz):
    if kind == "binary":
        return s * torch.sign(w)
    # ternary: levels {-s,0,s}, deadzone |w| <= dz*s
    return s * torch.sign(w) * (w.abs() > dz * s).to(w.dtype)

def level_index(q, s, kind):
    # ternary -> 0,1,2 ; binary -> 0,2
    return (torch.round(q / s) + 1).long()

def theory_penalty(w, s, kind):
    a = w.abs()
    inside = a < s
    if kind == "binary":
        pen_in = s * s - a * a
    else:
        pen_in = a * (s - a)
    pen_out = (a - s) ** 2
    return torch.where(inside, pen_in, pen_out)

def theory_flip(w, s, kind):
    a = w.abs()
    inside = a < s
    if kind == "binary":
        f = 1 - a / s
    else:
        f = 2 * torch.minimum(a, s - a) / s
    return torch.where(inside, f, torch.zeros_like(a))

def theory_mode_penalty(w, s, kind):
    a = w.abs()
    inside = a < s
    if kind == "binary":
        pen_in = (s - a) ** 2
    else:
        pen_in = torch.minimum(a * a, (s - a) ** 2)
    return torch.where(inside, pen_in, (a - s) ** 2)

def run(cfg, n, steps, burn, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    wstar = (laplace if cfg["dist"] == "laplace" else gauss)(n, g, dev)
    Eabs = 1.0 if cfg["dist"] == "laplace" else math.sqrt(2 / math.pi)
    s = cfg["s_mult"] * Eabs
    kind, dz, opt, lr, wd = cfg["kind"], cfg["dz"], cfg["opt"], cfg["lr"], cfg["wd"]
    h = 1.0
    w = 0.5 * wstar + 0.1 * torch.randn(n, generator=g, device=dev)   # transient start
    m = torch.zeros_like(w); v = torch.zeros_like(w)
    b1, b2, eps = 0.9, 0.95, 1e-8
    qsum = torch.zeros_like(w); flips = torch.zeros_like(w)
    counts = torch.zeros(3, n, device=dev)
    ema = w.clone(); ema_beta = 0.99
    q_prev = quantize(w, s, kind, dz)
    win = 0
    for t in range(1, steps + 1):
        q = quantize(w, s, kind, dz)
        grad = h * (q - wstar)
        if opt == "sgd":
            w = w - lr * grad - lr * wd * w
        else:  # adamw, decoupled decay
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
            w = w - lr * (mh / (vh.sqrt() + eps) + wd * w)
        ema = ema_beta * ema + (1 - ema_beta) * w
        if t > burn:
            qsum += q; flips += (q != q_prev).to(w.dtype)
            counts.scatter_add_(0, level_index(q, s, kind).unsqueeze(0), torch.ones(1, n, device=dev))
            win += 1
        q_prev = q
    q_last = quantize(w, s, kind, dz)
    qbar = qsum / win
    flip_rate = flips / win
    mode_idx = counts.argmax(0)
    q_mode = (mode_idx.to(w.dtype) - 1) * s
    q_ema = quantize(ema, s, kind, dz)
    pen_snap = (q_last - wstar) ** 2
    pen_mode = (q_mode - wstar) ** 2
    pen_ema = (q_ema - wstar) ** 2
    pen_avg = (qbar - wstar) ** 2
    inside = wstar.abs() < s
    wi, qi = wstar[inside], qbar[inside]
    out = dict(cfg)
    out.update(
        s=s,
        inside_frac=float(inside.float().mean()),
        # time average vs w* (interior coordinates only; exterior ones sit pinned at +-s)
        slope_qbar_vs_wstar=float((qi * wi).sum() / (wi * wi).sum()),
        rmse_qbar_minus_wstar=float(pen_avg[inside].mean().sqrt()),
        ema_equals_mode_frac=float((q_ema[inside] == q_mode[inside]).float().mean()),
        # total penalties (mean of (q-w*)^2 == G(s) per coordinate)
        pen_snapshot=float(pen_snap.mean()),
        pen_snapshot_theory=float(theory_penalty(wstar, s, kind).mean()),
        pen_mode=float(pen_mode.mean()),
        pen_mode_theory=float(theory_mode_penalty(wstar, s, kind).mean()),
        pen_ema_requant=float(pen_ema.mean()),
        # flip rate
        flip_rate=float(flip_rate.mean()),
        flip_rate_theory=float(theory_flip(wstar, s, kind).mean()),
        zero_frac=float((q_last == 0).float().mean()) if kind == "ternary" else 0.0,
        zero_frac_theory=float((torch.clamp(1 - wstar.abs() / s, min=0)).mean()) if kind == "ternary" else 0.0,
        # never-flipping interior coordinates (closed form for SGD + coupled decay, ternary)
        nonflip_frac=float((flips[inside] == 0).float().mean()),
        nonflip_frac_theory=float(((wstar.abs() < wd * dz * s / h).float().sum() / inside.float().sum())) if (opt == "sgd" and wd > 0 and kind == "ternary") else 0.0,
        three_level_frac=float(((counts > 0).sum(0) == 3).float().mean()),
    )
    # curves binned by |w*|/s
    r = (wstar.abs() / s)
    edges = torch.linspace(0, 1.5, 31, device=dev)
    idx = torch.bucketize(r, edges) - 1
    idx = idx.clamp(0, 29)
    def binned(x):
        num = torch.zeros(30, device=dev).scatter_add_(0, idx, x)
        den = torch.zeros(30, device=dev).scatter_add_(0, idx, torch.ones_like(x))
        return (num / den.clamp(min=1)).tolist()
    out["bins"] = ((edges[:-1] + edges[1:]) / 2).tolist()
    out["pen_binned"] = binned(pen_snap)
    out["pen_binned_theory"] = binned(theory_penalty(wstar, s, kind))
    out["flip_binned"] = binned(flip_rate)
    out["flip_binned_theory"] = binned(theory_flip(wstar, s, kind))
    out["qbar_binned"] = binned(qbar * torch.sign(wstar) / s)   # normalized, should equal |w*|/s
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--burn", type=int, default=10000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    if args.quick:
        args.n, args.steps, args.burn = 200_000, 4000, 2000
    cfgs = []
    # total penalty vs s, both quantizers, both dists, both optimizers
    for dist in ["laplace", "gauss"]:
        for opt, lr in [("sgd", 1e-2), ("adam", 1e-3)]:
            for kind in ["binary", "ternary"]:
                for sm in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
                    cfgs.append(dict(dist=dist, opt=opt, lr=lr, kind=kind, s_mult=sm, dz=0.5, wd=0.0, tag="scan"))
    # Delta-insensitivity: ternary s=1, dz sweep
    for dz in [0.25, 0.75]:
        for opt, lr in [("sgd", 1e-2), ("adam", 1e-3)]:
            cfgs.append(dict(dist="laplace", opt=opt, lr=lr, kind="ternary", s_mult=1.0, dz=dz, wd=0.0, tag="dz"))
    # LR sensitivity at s=1
    for opt, lrs in [("sgd", [3e-3, 3e-2]), ("adam", [3e-4, 3e-3, 1e-2])]:
        for lr in lrs:
            for kind in ["binary", "ternary"]:
                cfgs.append(dict(dist="laplace", opt=opt, lr=lr, kind=kind, s_mult=1.0, dz=0.5, wd=0.0, tag="lr"))
    # SGD + coupled weight decay
    for wd in [0.01, 0.1]:
        for kind in ["binary", "ternary"]:
            cfgs.append(dict(dist="laplace", opt="sgd", lr=1e-2, kind=kind, s_mult=1.0, dz=0.5, wd=wd, tag="wd"))
    # AdamW decoupled decay 0.1 (realistic)
    for kind in ["binary", "ternary"]:
        cfgs.append(dict(dist="laplace", opt="adam", lr=1e-3, kind=kind, s_mult=1.0, dz=0.5, wd=0.1, tag="adamw"))

    results = []
    t0 = time.time()
    for i, cfg in enumerate(cfgs):
        r = run(cfg, args.n, args.steps, args.burn, dev)
        results.append(r)
        print(f"[{i+1}/{len(cfgs)} {time.time()-t0:6.0f}s] {cfg['tag']:5s} {cfg['dist']:7s} {cfg['opt']:4s} lr={cfg['lr']:.0e} {cfg['kind']:7s} s={cfg['s_mult']:.2f} dz={cfg['dz']} wd={cfg['wd']} | "
              f"slope={r['slope_qbar_vs_wstar']:.4f} rmse={r['rmse_qbar_minus_wstar']:.4f} ema=mode:{r['ema_equals_mode_frac']:.3f} pen={r['pen_snapshot']:.4f}/{r['pen_snapshot_theory']:.4f} "
              f"mode={r['pen_mode']:.4f}/{r['pen_mode_theory']:.4f} ema={r['pen_ema_requant']:.4f} "
              f"flip={r['flip_rate']:.3f}/{r['flip_rate_theory']:.3f} zero={r['zero_frac']:.3f}/{r['zero_frac_theory']:.3f} "
              f"nonflip={r['nonflip_frac']:.4f}/{r['nonflip_frac_theory']:.4f} 3lvl={r['three_level_frac']:.4f}", flush=True)
        with open(os.path.join(args.out, "e1_results.json"), "w") as f:
            json.dump(results, f)
    print("done", time.time() - t0)

if __name__ == "__main__":
    main()
