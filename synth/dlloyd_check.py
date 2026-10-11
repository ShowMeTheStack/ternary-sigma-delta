"""Synthetic check: STE-Adam with a k-level quantizer using dither-optimal (synth/dither_lloyd.py) vs Lloyd-Max levels
(Laplacian, E|w|=1, k = 4 and 7). Reports the measured snapshot penalty (should be lower for the dither-optimal levels
by the predicted D gain) and the mode-rounded penalty, next to the D and MSE functionals of each level set.
python synth/dlloyd_check.py
"""
import torch, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1 import laplace
torch.set_num_threads(4); dev = "cpu"; n = 200000; steps, burn = 20000, 10000; lr = 1e-3; b1, b2 = 0.9, 0.95
LEVELS = {
    (4, "dl"): [-2.018, -0.522, 0.522, 2.018], (4, "lm"): [-2.575, -0.601, 0.601, 2.575],
    (7, "dl"): [-3.263, -1.752, -0.745, 0.0, 0.745, 1.752, 3.263], (7, "lm"): [-4.029, -2.043, -0.851, 0.0, 0.851, 2.043, 4.029],
}
def quantize(w, L):  # nearest level
    d = (w[:, None] - L[None, :]).abs(); return L[d.argmin(1)]
for (k, kind), lv in LEVELS.items():
    L = torch.tensor(lv); g = torch.Generator(device=dev).manual_seed(0)
    wstar = laplace(n, g, dev); w = 0.5 * wstar + 0.1 * torch.randn(n, generator=g, device=dev)
    m = torch.zeros_like(w); v = torch.zeros_like(w); counts = torch.zeros(len(lv), n); win = 0
    for t in range(1, steps + 1):
        q = quantize(w, L); grad = q - wstar
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad * grad
        w = w - lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + 1e-8)
        if t > burn:
            idx = (w[:, None] - L[None, :]).abs().argmin(1); counts[idx, torch.arange(n)] += 1; win += 1
    q_last = quantize(w, L); pen = float(((q_last - wstar) ** 2).mean())
    mode = L[counts.argmax(0)]; pen_mode = float(((mode - wstar) ** 2).mean())
    # theoretical D and MSE for these levels under the Laplacian
    X = torch.linspace(-12, 12, 240001); p = torch.exp(-X.abs()) / 2; dx = X[1] - X[0]
    idx = torch.searchsorted(L, X).clamp(1, len(lv) - 1); lo = L[idx - 1]; hi = L[idx]
    inside = (X >= L[0]) & (X <= L[-1]); dith = torch.where(inside, (X - lo) * (hi - X), torch.minimum((X - L[0]) ** 2, (X - L[-1]) ** 2))
    D = float((dith * p).sum() * dx); near = (X[:, None] - L[None, :]).abs().min(1).values; MSE = float((near ** 2 * p).sum() * dx)
    print(f"k={k} {kind}: measured snapshot penalty {pen:.4f} (theory D {D:.4f})  mode-rounded {pen_mode:.4f} (theory MSE {MSE:.4f})", flush=True)
