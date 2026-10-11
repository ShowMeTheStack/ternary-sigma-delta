"""Regime map: dither gap vs scale s under gradient noise (Adam), sigma in {0, 3, 10}, ternary and binary.
The gap is snapshot penalty minus the penalty of the average over the last 64 steps (--burn is not used).
python synth/e1_noise_gamma.py --device cpu --n 100000 --steps 8000 --burn 4000
"""
import argparse, math, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1 import laplace, quantize

def run(kind, sigma, s, lr, n, steps, burn, dev, win=64, dz=0.5, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    wstar = laplace(n, g, dev); w = 0.5 * wstar + 0.1 * torch.randn(n, generator=g, device=dev)
    m = torch.zeros_like(w); v = torch.zeros_like(w); b1, b2 = 0.9, 0.95
    qsum = torch.zeros_like(w); q2 = torch.zeros_like(w); cnt = 0
    for t in range(1, steps + 1):
        q = quantize(w, s, kind, dz); grad = (q - wstar) + sigma * torch.randn(n, generator=g, device=dev)
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad * grad
        w = w - lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + 1e-8)
        if t > steps - win: qsum += q; q2 += q * q; cnt += 1      # short stationary window like the LM probe
    qbar = qsum / cnt; varq = q2 / cnt - qbar ** 2; q_last = quantize(w, s, kind, dz)
    pen = ((q_last - wstar) ** 2).mean(); gap = pen - ((qbar - wstar) ** 2).mean()   # snapshot minus dither-free
    return float(pen), float(gap), float(varq.mean() / s ** 2), float((q_last == 0).float().mean())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--burn", type=int, default=4000); a = ap.parse_args()
    torch.set_num_threads(4)
    for sigma in [0.0, 3.0, 10.0]:
        for kind in ["ternary", "binary"]:
            for s in [0.75, 1.0, 1.25, 1.5, 2.0]:
                pen, gap, varq, zero = run(kind, sigma, s, 1e-3, a.n, a.steps, a.burn, a.device)
                print(f"sigma={sigma:5.1f} {kind:7s} s={s:4.2f} pen={pen:.4f} gap64={gap:.4f} varq64/s2={varq:.4f} zero={zero:.3f}", flush=True)
