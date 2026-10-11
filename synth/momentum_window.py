"""Finite-window error of the duty-cycle average, SGD vs heavy-ball momentum, noise-free ternary model.
Reports RMS(qbar_W - w*) over interior coordinates for windows W; the log-log slopes tell 1/W vs 1/W^2.
python synth/momentum_window.py
"""
import numpy as np, sys
rng = np.random.default_rng(0); N = 50000; eta, h, s, dz = 1e-2, 1.0, 1.0, 0.5
ws = rng.laplace(0, 1, N); ws = ws[np.abs(ws) < 0.95 * s]        # interior targets only
def q(w): return s * np.sign(w) * (np.abs(w) > dz * s)
def run(beta, T=6000, burn=3000, windows=(16, 64, 256, 1024)):
    w = ws.copy(); u = np.zeros_like(w); hist = []
    for t in range(T):
        qt = q(w); g = h * (qt - ws)
        u = beta * u + g; w = w - eta * u
        if t >= burn: hist.append(qt)
    H = np.array(hist); out = {}
    for W in windows:
        errs = []
        for start in range(0, H.shape[0] - W, max(1, (H.shape[0] - W) // 20)):
            errs.append(np.sqrt(np.mean((H[start:start + W].mean(0) - ws) ** 2)))
        out[W] = float(np.mean(errs))
    return out
for beta in (0.0, 0.9):
    r = run(beta); Ws = sorted(r); e = [r[W] for W in Ws]
    slopes = [np.log(e[i + 1] / e[i]) / np.log(Ws[i + 1] / Ws[i]) for i in range(len(Ws) - 1)]
    print(f"beta={beta}: " + ", ".join(f"W={W}: {r[W]:.5f}" for W in Ws) + f"  log-slopes {np.round(slopes, 2)}")
