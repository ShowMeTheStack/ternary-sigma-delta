"""Stochastic-rounding variance prediction. SR makes each interior weight Bernoulli(u), u=|w|/s
fractional, so Var/s^2 = E[u(1-u)] over interior weights, from the latent histogram (units of E|W|, s = gamma*E|W|,
gamma = 1 assumed). Deterministic reference: the measured Var(q)/s^2 of the deterministic run's probe. The
predicted variance ratio is compared with the measured loss-gap ratio. Both runs need probe_64.json.
python analysis/sr_predict.py runs/S_ter_g1.0_stoch_s0 runs/S_ter_g1.0_s0
"""
import json, os, sys, numpy as np
sr, det = sys.argv[1], sys.argv[2]
h = [json.loads(l) for l in open(os.path.join(sr, "hist.jsonl"))][-1]["hist"]
edges = np.linspace(-4, 4, 65); mid = (edges[:-1] + edges[1:]) / 2
tot = 0.0; n = 0
for layer, hist in h.items():
    hist = np.array(hist); u = np.abs(mid)                # latent in units of E|W|; s = 1.0*E|W| for gamma=1
    inside = u < 1.0
    tot += float((hist[inside] * (u[inside] * (1 - u[inside]))).sum()); n += 1
Eu = tot / n
pd = json.load(open(os.path.join(det, "probe_64.json")))
ps = json.load(open(os.path.join(sr, "probe_64.json")))
print(f"SR predicted Var/s^2 = E[u(1-u)] over interior = {Eu:.4f}; deterministic probe Var/s^2 = {pd['mean_var_q_over_s2']:.4f}; SR probe Var/s^2 measured = {ps['mean_var_q_over_s2']:.4f}")
gd = pd["val_post_snapshot"] - pd["val_dither_free"]; gs = ps["val_post_snapshot"] - ps["val_dither_free"]
print(f"predicted variance ratio SR/det = {Eu/pd['mean_var_q_over_s2']:.2f};  measured loss-gap ratio = {gs/gd:.2f} ({gs:.4f}/{gd:.4f});  measured Var ratio = {ps['mean_var_q_over_s2']/pd['mean_var_q_over_s2']:.2f}")
