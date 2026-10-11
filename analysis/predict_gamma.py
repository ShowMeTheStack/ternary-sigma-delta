"""Predict the dither-optimal scale multiplier gamma* from an independent estimate of the
full-precision weight distribution: the final weights of a BF16 run, layer by layer.

For each quantized layer with FP weights W (|W| = u), s = gamma * E|W|:
  snapshot penalty (ternary)  G_T(gamma) = mean_u [ u(s-u) if u<s else (u-s)^2 ]
  snapshot penalty (binary)   G_B(gamma) = mean_u [ s^2-u^2 if u<s else (u-s)^2 ]
  mode-rounded (ternary)      M_T(gamma) = mean_u [ min(u^2,(s-u)^2) if u<s else (u-s)^2 ]
Uniform curvature is assumed across weights within and across layers (layers are weighted by size).
Also reports the fitted Laplacian/Gaussian shape via excess kurtosis and the predicted zero fraction.

python analysis/predict_gamma.py runs/S_fp_s0      -> runs/S_fp_s0/predict_gamma.json
"""
import argparse, json, os, sys
import numpy as np, torch

def penalties(u, gammas, Eabs):
    out = {"G_T": [], "G_B": [], "M_T": [], "zero": []}
    for g in gammas:
        s = g * Eabs; inside = u < s; tail = ((u - s) ** 2)[~inside].sum()
        ui = u[inside]
        out["G_T"].append((ui * (s - ui)).sum() + tail)
        out["G_B"].append((s * s - ui * ui).sum() + tail)
        out["M_T"].append(np.minimum(ui * ui, (s - ui) ** 2).sum() + tail)
        out["zero"].append((1 - ui / s).sum())
    return {k: np.array(v) for k, v in out.items()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fp_run"); ap.add_argument("--gammas", type=float, nargs="*", default=list(np.round(np.arange(0.5, 2.55, 0.05), 2)))
    a = ap.parse_args()
    st = torch.load(os.path.join(a.fp_run, "final_state.pt"), weights_only=False)
    tot = {"G_T": 0, "G_B": 0, "M_T": 0, "zero": 0}; n = 0; kurt = []
    for name, p in st.items():
        if not (name.startswith("blocks.") and name.endswith(".weight") and p.dim() == 2): continue
        w = p.float().numpy().ravel(); u = np.abs(w); Eabs = u.mean()
        k = ((w - w.mean()) ** 4).mean() / (w.var() ** 2) - 3; kurt.append(k)
        pen = penalties(u, a.gammas, Eabs)
        for key in tot: tot[key] = tot[key] + pen[key] / (Eabs ** 2 if key != "zero" else 1)   # normalize by layer scale^2
        n += u.size
    g = np.array(a.gammas)
    res = {k: (v / n) for k, v in tot.items()}
    out = dict(n_weights=n, mean_excess_kurtosis=float(np.mean(kurt)),
               gamma_star_snapshot_ternary=float(g[np.argmin(res["G_T"])]), gamma_star_snapshot_binary=float(g[np.argmin(res["G_B"])]),
               gamma_star_mode_ternary=float(g[np.argmin(res["M_T"])]),
               ratio_B_over_T_at_gamma1=float(res["G_B"][np.argmin(abs(g - 1))] / res["G_T"][np.argmin(abs(g - 1))]),
               predicted_zero_frac_at_gamma1=float(res["zero"][np.argmin(abs(g - 1))]),
               predicted_zero_frac_at_gamma_star=float(res["zero"][np.argmin(res["G_T"])]),
               curve={f"{gg:.2f}": dict(G_T=float(a_), G_B=float(b_), M_T=float(c_)) for gg, a_, b_, c_ in zip(g, res["G_T"], res["G_B"], res["M_T"])})
    json.dump(out, open(os.path.join(a.fp_run, "predict_gamma.json"), "w"), indent=1)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k != "curve"})
    print("Laplacian excess kurtosis = 3, Gaussian = 0; Laplacian predicts gamma*_T=1.50, Gaussian 1.28")

if __name__ == "__main__":
    main()
