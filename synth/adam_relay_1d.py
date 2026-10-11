"""One-dimensional deterministic Adam relay.

A single coordinate with loss (h/2)(q - w*)^2, h = 1, binary or ternary quantizer with s = 1, STE gradient
g_t = h (q_t - w*), Adam exactly as synth/e1.py (lr 1e-3, beta = (0.9, 0.95), eps 1e-8, bias correction),
no noise, no weight decay. Run 20000 steps, measure the flip rate (fraction of steps where q changes) and the
period of the steady-state q sequence over the last 10000 steps (same burn-in and window as synth/e1.py),
for w* = 0.0, 0.1, ..., 0.9.

The initial latent follows synth/e1.py (w0 = 0.5 w* + 0.1 z); since the system is deterministic we report several z
(z = +-1, +-0.5, 0.1, 0.01) to show whether the locked rate depends on the start. sign(0) = 0 would freeze
the coordinate at w = 0 exactly, so z = 0 is not used.

python synth/adam_relay_1d.py --out synth/results/adam_relay_1d.json
"""
import argparse, json, math, os
import numpy as np


def relay(wstar, z, s=1.0, lr=1e-3, b1=0.9, b2=0.95, eps=1e-8, steps=20000, burn=10000, kind="binary", dz=0.5):
    w = 0.5 * wstar + 0.1 * z; m = v = 0.0
    def quant(w):
        if kind == "binary": return s * float(np.sign(w))
        return s * float(np.sign(w)) * float(abs(w) > dz * s)
    qs = []
    for t in range(1, steps + 1):
        q = quant(w); g = q - wstar
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        w = w - lr * (m / (1 - b1 ** t)) / (math.sqrt(v / (1 - b2 ** t)) + eps)
        qs.append(q)
    qs = np.array(qs)
    win = qs[burn - 1:]   # include the last pre-window step so flips are counted over exactly steps - burn transitions
    flips = int((win[1:] != win[:-1]).sum()); rate = flips / (steps - burn)
    tail = qs[burn:]
    period = None
    for P in range(1, 2001):
        if np.array_equal(tail[P:], tail[:-P]): period = P; break
    qbar = float(tail.mean())
    return dict(wstar=wstar, z=z, kind=kind, flip_rate=rate, flips=flips, period=period,
                flips_per_period=(int((np.roll(tail[:period], 1) != tail[:period]).sum()) if period else None),
                rate_from_period=(int((np.roll(tail[:period], 1) != tail[:period]).sum()) / period if period else None),
                qbar=qbar, qbar_minus_wstar=qbar - wstar)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="synth/results/adam_relay_1d.json"); a = ap.parse_args()
    res = []
    for kind in ["binary", "ternary"]:
        for ws in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for z in [1.0, -1.0, 0.5, -0.5, 0.1, 0.01]:
                r = relay(ws, z, kind=kind); res.append(r)
                print(f"{kind:7s} w*={ws:.2f} z={z:+.2f} flip_rate={r['flip_rate']:.4f} period={r['period']} "
                      f"(flips/period {r['flips_per_period']}, rate {r['rate_from_period']}) qbar-w*={r['qbar_minus_wstar']:+.4f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(config=dict(s=1.0, lr=1e-3, betas=[0.9, 0.95], eps=1e-8, steps=20000, burn=10000, h=1.0, init="w0 = 0.5 w* + 0.1 z"),
                   runs=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
