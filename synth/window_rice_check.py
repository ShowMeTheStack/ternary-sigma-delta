"""Synthetic window and level-crossing checks. Adam, ternary, Laplacian w*, s=1, dz=0.5.
(a) finite-window error |qbar_W - w*| (interior, RMS over coords) vs W for sigma in {0, 10}: expect ~1/W and ~1/sqrt(W).
(b) Rice level-crossing law at sigma=10: flip rate vs 2*(rho(+dz)+rho(-dz))*E|dw| measured from the simulation (rho from latent histogram).
python synth/window_rice_check.py
"""
import math, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1 import laplace, quantize
torch.set_num_threads(4); dev = "cpu"; n = 100000; s, dz, lr = 1.0, 0.5, 1e-3; b1, b2 = 0.9, 0.95
for sigma in [0.0, 10.0]:
    g = torch.Generator(device=dev).manual_seed(0)
    wstar = laplace(n, g, dev); w = 0.5 * wstar + 0.1 * torch.randn(n, generator=g, device=dev)
    m = torch.zeros_like(w); v = torch.zeros_like(w); burn = 4000
    Ws = [16, 64, 256, 1024, 4096]; total = burn + max(Ws)
    qsum = {W: torch.zeros_like(w) for W in Ws}; flips = 0; absdw = 0.0; nsteps = 0; qprev = None; hist = torch.zeros(160)
    for t in range(1, total + 1):
        q = quantize(w, s, "ternary", dz); grad = (q - wstar) + sigma * torch.randn(n, generator=g, device=dev)
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad * grad
        step = lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + 1e-8); w = w - step
        if t > burn:
            k = t - burn
            for W in Ws:
                if k <= W: qsum[W] += q
            if qprev is not None: flips += float((q != qprev).float().mean())
            absdw += float(step.abs().mean()); nsteps += 1
            hist += torch.histc(w.clamp(-4, 4), bins=160, min=-4, max=4) / n
        qprev = q
    inside = wstar.abs() < s
    errs = {W: float((((qsum[W] / W) - wstar)[inside] ** 2).mean().sqrt()) for W in Ws}
    hist /= nsteps; width = 8 / 160; rho = hist / width
    def rho_at(x): i = int((x + 4) / width); return float(rho[min(max(i, 0), 159)])
    dbar = absdw / nsteps; r_meas = flips / (nsteps - 1); r_pred = 2 * (rho_at(dz) + rho_at(-dz)) * dbar
    print(f"sigma={sigma:4.1f}  RMS |qbar_W - w*| interior: " + "  ".join(f"W={W}:{errs[W]:.4f}" for W in Ws) + f"   | Rice: E|dw|={dbar:.5f} rho(+-dz)={rho_at(dz):.3f},{rho_at(-dz):.3f} r_pred={r_pred:.4f} r_meas={r_meas:.4f} ratio={r_meas/r_pred:.2f}", flush=True)
