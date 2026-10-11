"""Predicted duty-cycle slope drop under decoupled AdamW weight decay.

Model: under AdamW, w <- w - eta (m^/sqrt(v^) + lambda w). A bounded trajectory has zero mean update, so
mean(g)/sqrt(v^) = -lambda w_bar and q_bar - w* = -(lambda sqrt(v^)/h) w_bar. For an interior ternary weight
(0 < u = |w*| < s) the latent oscillates about the threshold, w_bar ~ Delta = dz*s, and in the two-level regime
v^ ~ E[g^2] = h^2 u (s - u) (duty u/s at level s, g = h(s-u) or -h u). Hence
    q_bar = w* - lambda Delta sqrt(u (s - u)) sign(w*),
and the through-origin regression slope of q_bar on w* over interior coordinates drops, relative to the
no-decay Adam baseline, by
    drop = lambda Delta E[u sqrt(u (s - u))] / E[u^2]   (moments of |w*| truncated to [0, s)).
For comparison, coupled decay under SGD: slope = 1 - lambda Delta E[u]/E[u^2] (q_bar = w* - lambda Delta / h).

Laplacian w* with E|w| = 1 (|w*| ~ Exp(1)), s = 1, dz = 0.5, h = 1, lambda = 0.1 (AdamW) / 0.01, 0.1 (SGD).
Computes the moments by fine midpoint quadrature (no scipy) and on the synth/e1.py draw (seed 0, n = 1e6, torch,
same generator), and, if synth/results/e1/e1_results.json exists, the measured drop
(slope of Adam ternary s=1 wd=0 minus slope of AdamW ternary s=1 wd=0.1).

python synth/adamw_slope_drop.py --out synth/results/adamw_slope_drop.json
"""
import argparse, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def moments_quad(s=1.0, N=2_000_000):
    u = (np.arange(N) + 0.5) * (s / N); p = np.exp(-u); du = s / N
    Z = (p * du).sum()
    E = lambda f: float((f * p * du).sum() / Z)
    return dict(Eu=E(u), Eu2=E(u * u), Eu_sqrt=E(u * np.sqrt(u * (s - u))))


def moments_sample(s=1.0, n=1_000_000, seed=0):
    import torch, sys
    sys.path.insert(0, HERE); from e1 import laplace
    g = torch.Generator(device="cpu").manual_seed(seed); w = laplace(n, g, "cpu").double().numpy()
    u = np.abs(w); u = u[u < s]
    return dict(Eu=float(u.mean()), Eu2=float((u * u).mean()), Eu_sqrt=float((u * np.sqrt(u * (s - u))).mean()), n_interior=int(u.size))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "results/adamw_slope_drop.json")); a = ap.parse_args()
    s, dz = 1.0, 0.5; D = dz * s
    out = dict(config=dict(dist="laplace E|w|=1", s=s, dz=dz, h=1.0))
    srcs = dict(quadrature=moments_quad(s))
    try: srcs["e1_sample_seed0_n1e6"] = moments_sample(s)
    except Exception as e: print("sample moments skipped:", e)
    for k, mo in srcs.items():
        mo["sgd_slope_wd0.01"] = 1 - 0.01 * D * mo["Eu"] / mo["Eu2"]
        mo["sgd_slope_wd0.1"] = 1 - 0.1 * D * mo["Eu"] / mo["Eu2"]
        mo["adamw_drop_wd0.1"] = 0.1 * D * mo["Eu_sqrt"] / mo["Eu2"]
        print(k, {kk: round(v, 5) if isinstance(v, float) else v for kk, v in mo.items()})
    out["predicted"] = srcs
    ej = os.path.join(HERE, "results/e1/e1_results.json")
    if os.path.exists(ej):
        R = json.load(open(ej))
        sel = lambda tag, wd: [r for r in R if r["tag"] == tag and r["dist"] == "laplace" and r["opt"] == "adam" and r["kind"] == "ternary"
                               and r["s_mult"] == 1.0 and r["dz"] == 0.5 and r["lr"] == 1e-3 and r["wd"] == wd][0]
        b, w = sel("scan", 0.0), sel("adamw", 0.1)
        out["measured_e1"] = dict(slope_adam=b["slope_qbar_vs_wstar"], slope_adamw=w["slope_qbar_vs_wstar"],
                                  drop=b["slope_qbar_vs_wstar"] - w["slope_qbar_vs_wstar"], zero_adam=b["zero_frac"], zero_adamw=w["zero_frac"])
        print("measured (E1, n=1e6):", out["measured_e1"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
