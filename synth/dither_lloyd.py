"""Dither-optimal k-level quantizers vs Lloyd-Max, Laplacian (E|w|=1) and Gaussian (sigma=1).
Dither functional D(l) = sum_i int_{l_i}^{l_{i+1}} (w-l_i)(l_{i+1}-w) p dw + tails (w-l_edge)^2; MSE functional = Lloyd-Max.
Symmetric levels {0, +-l_1, ...} (k odd) or {+-l_1, ...} (k even). Coordinate descent on a fine grid.
python synth/dither_lloyd.py
"""
import numpy as np, sys
def dens(name, x):
    return np.exp(-np.abs(x)) / 2 if name == "laplace" else np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
X = np.linspace(-10, 10, 40001); DX = X[1] - X[0]
def penalties(levels, p):
    L = np.sort(levels); idx = np.searchsorted(L, X)                  # cell index: X between L[idx-1] and L[idx]
    lo = np.where(idx > 0, L[np.clip(idx - 1, 0, len(L) - 1)], -np.inf); hi = np.where(idx < len(L), L[np.clip(idx, 0, len(L) - 1)], np.inf)
    inside = np.isfinite(lo) & np.isfinite(hi)
    dither = np.where(inside, (X - lo) * (hi - X), np.where(np.isfinite(lo), (X - lo) ** 2, (hi - X) ** 2))
    near = np.minimum(np.abs(X - np.where(np.isfinite(lo), lo, hi)), np.abs(X - np.where(np.isfinite(hi), hi, lo)))
    return float((dither * p).sum() * DX), float((near ** 2 * p).sum() * DX)
def optimize(k, name, functional):
    p = dens(name, X); half = k // 2; odd = k % 2 == 1
    mags = np.linspace(0.5, 2.5, half) if half else np.array([])
    def build(m): return np.concatenate([[-x for x in m[::-1]], [0.0] if odd else [], m])
    for sweep in range(12):
        for i in range(len(mags)):
            best = None
            lo_b = mags[i - 1] + 1e-3 if i > 0 else (1e-3 if odd else 1e-3); hi_b = mags[i + 1] - 1e-3 if i + 1 < len(mags) else 8.0
            for v in np.linspace(lo_b, hi_b, 121):
                m = mags.copy(); m[i] = v; val = penalties(build(m), p)[0 if functional == "dither" else 1]
                if best is None or val < best[0]: best = (val, v)
            mags[i] = best[1]
        # refine grid once converged coarse
    return build(mags), penalties(build(mags), p)
if __name__ == "__main__":
    for name in ["laplace", "gauss"]:
        Eabs = 1.0 if name == "laplace" else np.sqrt(2 / np.pi)
        for k in [2, 3, 4, 5, 7, 15]:
            ld, (dd, md) = optimize(k, name, "dither"); ll, (dl, ml) = optimize(k, name, "mse")
            pos = lambda L: [round(float(x) / Eabs, 3) for x in L if x > 1e-9]
            print(f"{name:7s} k={k:2d} dither-Lloyd levels/E|w| {pos(ld)}  D={dd:.4f} MSE={md:.4f} | Lloyd-Max levels {pos(ll)}  D={dl:.4f} MSE={ml:.4f} | D gain {100*(dl-dd)/dl:.1f}%", flush=True)
