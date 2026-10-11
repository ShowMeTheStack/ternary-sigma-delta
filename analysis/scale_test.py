"""Measured dither gap (probe snapshot - dither-free) and mode gain vs gamma for a sweep of runs under --root
(given alphabet, dz = 0.5, deterministic rounding, no wd switch-off, BF16 activations), against the prediction
G_T(gamma) or G_B(gamma) and M_T(gamma) computed from a BF16 run's weights (predict_gamma.json in --fp),
both normalized at gamma = 1. Gap and mode gain come from probe_K.json (analysis/probe.py --steps K).

python analysis/scale_test.py --root runs/gamma_sweep --fp runs/S_fp_s0 --alphabet ternary --K 64
"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--fp", required=True)
ap.add_argument("--alphabet", default="ternary"); ap.add_argument("--K", type=int, default=64); a = ap.parse_args()
pred = json.load(open(os.path.join(a.fp, "predict_gamma.json")))["curve"]
key = "G_T" if a.alphabet == "ternary" else "G_B"; mkey = "M_T"
rows = []
for d in sorted(glob.glob(os.path.join(a.root, "*"))):
    args_f, probe_f, final_f = os.path.join(d, "args.json"), os.path.join(d, f"probe_{a.K}.json"), os.path.join(d, "final.json")
    if not (os.path.exists(args_f) and os.path.exists(final_f)): continue
    ar = json.load(open(args_f))
    if ar["alphabet"] != a.alphabet or ar["dz"] != 0.5 or ar.get("stochastic") or ar.get("wd_off_frac", 0) or ar.get("act_bits"): continue
    fin = json.load(open(final_f)); g = ar["gamma"]
    row = dict(gamma=g, seed=ar["seed"], val=fin["val_snapshot"], zero=fin.get("mean_zero_frac"))
    if os.path.exists(probe_f):
        p = json.load(open(probe_f)); row.update(gap=p["val_post_snapshot"] - p["val_dither_free"], mode_gain=p["val_post_snapshot"] - p["val_mode"], varq=p["mean_var_q_over_s2"], probe_snap=p["val_post_snapshot"], probe_mode=p["val_mode"])
    rows.append(row)
rows.sort(key=lambda r: (r["gamma"], r["seed"]))
g1 = [r for r in rows if r["gamma"] == 1.0 and "gap" in r]
gap1 = np.mean([r["gap"] for r in g1]) if g1 else np.nan
pk = lambda g: pred[f"{g:.2f}"]
print(f"{'gamma':>5s} {'seed':>4s} {'val':>7s} {'zero':>6s} | {'gap':>7s} {'pred_gap':>8s} | {'mode_gain':>9s} {'probe_mode':>10s} | {'varq':>7s} {'predG_rel':>9s} {'predM_rel':>9s}")
for r in rows:
    pg = pk(r["gamma"])[key] / pk(1.0)[key] * gap1 if g1 else np.nan
    print(f"{r['gamma']:5.2f} {r['seed']:4d} {r['val']:7.4f} {r['zero'] or 0:6.3f} | {r.get('gap', np.nan):7.4f} {pg:8.4f} | {r.get('mode_gain', np.nan):9.4f} {r.get('probe_mode', np.nan):10.4f} | {r.get('varq', np.nan):7.4f} {pk(r['gamma'])[key]/pk(1.0)[key]:9.3f} {pk(r['gamma'])[mkey]/pk(1.0)[mkey]:9.3f}")
