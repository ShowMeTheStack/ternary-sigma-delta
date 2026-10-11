"""synth/e1.py setting with minibatch-like gradient noise: grad = h(q - w*) + sigma * N(0,1), Adam, sigma in {0, 1, 10, 100}.
Checks whether the time-average slope and the snapshot penalty keep their noise-free values, and how the flip rate
scales in the noise-dominated regime.
python synth/e1_noise.py --device cpu --n 200000 --steps 6000 --burn 3000 --seed 0 --out synth/results/e1_noise
With --out, writes e1_noise.json with config, every run dict and the closed forms G_B(s), G_T(s) at s = 1.
"""
import argparse, json, math, torch
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1 import laplace, quantize, theory_penalty, theory_flip

def run(kind, sigma, lr, n, steps, burn, dev, s=1.0, dz=0.5, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    wstar = laplace(n, g, dev); w = 0.5 * wstar + 0.1 * torch.randn(n, generator=g, device=dev)
    m = torch.zeros_like(w); v = torch.zeros_like(w); b1, b2 = 0.9, 0.95
    qsum = torch.zeros_like(w); flips = torch.zeros_like(w); win = 0; qprev = quantize(w, s, kind, dz)
    for t in range(1, steps + 1):
        q = quantize(w, s, kind, dz); grad = (q - wstar) + sigma * torch.randn(n, generator=g, device=dev)
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad * grad
        w = w - lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + 1e-8)
        if t > burn: qsum += q; flips += (q != qprev).float(); win += 1
        qprev = q
    qbar = qsum / win; inside = wstar.abs() < s; q_last = quantize(w, s, kind, dz)
    pen = ((q_last - wstar) ** 2).mean(); fr = (flips / win)
    r = wstar.abs() / s; edges = torch.linspace(0, 1.0, 11, device=dev); idx = (torch.bucketize(r, edges) - 1).clamp(0, 9)
    def binned(x):
        num = torch.zeros(10, device=dev).scatter_add_(0, idx, x); den = torch.zeros(10, device=dev).scatter_add_(0, idx, torch.ones_like(x)); return (num / den.clamp(min=1)).tolist()
    return dict(kind=kind, sigma=sigma, lr=lr, slope=float((qbar[inside] * wstar[inside]).sum() / (wstar[inside] ** 2).sum()),
                pen=float(pen), pen_theory=float(theory_penalty(wstar, s, kind).mean()), flip=float(fr.mean()),
                flip_binned=[round(x, 4) for x in binned(fr)], zero=float((q_last == 0).float().mean()))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--burn", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None); a = ap.parse_args()
    torch.set_num_threads(4)
    res = []; s0 = 1.0
    closed = dict(G_B=s0 ** 2 - 2 + (2 * s0 + 4) * math.exp(-s0), G_T=s0 - 2 + (s0 + 4) * math.exp(-s0))
    for sigma in [0.0, 1.0, 10.0, 100.0]:
        for kind in ["binary", "ternary"]:
            r = run(kind, sigma, 1e-3, a.n, a.steps, a.burn, a.device, seed=a.seed); res.append(r)
            if a.out:
                os.makedirs(a.out, exist_ok=True)
                with open(os.path.join(a.out, "e1_noise.json"), "w") as f:
                    json.dump(dict(config=dict(n=a.n, steps=a.steps, burn=a.burn, seed=a.seed, lr=1e-3, s=s0, dz=0.5, opt="adam", betas=[0.9, 0.95],
                                               dist="laplace", wstar_Eabs=1.0), closed_form_laplace_s1=closed, runs=res), f, indent=1)
            print(f"sigma={sigma:6.1f} {kind:7s} slope={r['slope']:.4f} pen={r['pen']:.4f}/{r['pen_theory']:.4f} zero={r['zero']:.3f} flip={r['flip']:.4f} binned={r['flip_binned']}", flush=True)
