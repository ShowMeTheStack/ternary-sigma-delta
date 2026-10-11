"""Level-crossing (Rice) prediction of the per-layer flip rate in the noise-dominated regime.
r_pred(layer) = 2 * rho(theta) * delta_bar, with rho the latent density at the threshold (per unit of E|W|, from the last
histogram in hist.jsonl: 64 bins on [-4,4] of w/E|W|), theta = dz*gamma (ternary; two thresholds +-theta) or 0 (binary),
and delta_bar = eta_final * sqrt((1-b1)/(1+b1)) / E|W| the Adam random-walk step for noise-dominated gradients
(|m_hat|/sqrt(v_hat) ~ sqrt((1-b1)/(1+b1)) for white-noise gradients), in units of E|W| (per-layer scale s = gamma*E|W|).
Measured: per-layer flip rate over the cool-down window (final.json per_layer.flip_rate_window); needs --cooldown > 0.
python analysis/rice.py runs/M_ter_g1.0_s0 runs/M_bin_g1.0_s0 ... [--plot rice.pdf]
--plot FILE also saves a scatter plot of predicted vs measured per-layer flip rates (needs matplotlib).
"""
import json, os, sys, numpy as np
b1 = 0.9; fac = np.sqrt((1 - b1) / (1 + b1))
plot_out = None
args = sys.argv[1:]
if "--plot" in args:
    i = args.index("--plot"); plot_out = args[i + 1]; args = args[:i] + args[i + 2:]
allpts = []
for run in args:
    a = json.load(open(os.path.join(run, "args.json"))); fin = json.load(open(os.path.join(run, "final.json")))
    if fin.get("cooldown", 0) == 0: print(run, "no live cool-down"); continue
    eta = a["lr"] * a["min_lr_frac"]; gamma = a["gamma"]; dz = a["dz"]
    hist = [json.loads(l) for l in open(os.path.join(run, "hist.jsonl"))][-1]["hist"]
    edges = np.linspace(-4, 4, 65); width = edges[1] - edges[0]
    rows = []
    for layer, h in hist.items():
        h = np.array(h) / width                                   # density per unit of E|W|
        pl = fin["per_layer"][layer]; s = pl["scale"]; Eabs = s / gamma
        theta = dz * gamma if a["alphabet"] == "ternary" else 0.0
        def rho(t): return float(np.interp(t, (edges[:-1] + edges[1:]) / 2, h))
        rho_sum = rho(theta) + rho(-theta) if a["alphabet"] == "ternary" else rho(0.0)   # crossings of each threshold
        delta = eta * fac / Eabs                                    # step in units of E|W|
        r_pred = 2 * rho_sum * delta            # discrete Rice: crossings per step = 2 rho(theta) E|dw| per threshold, summed over thresholds
        r_meas = pl["flip_rate_window"]
        rows.append((layer, rho_sum, r_pred, r_meas))
    rp = np.array([r[2] for r in rows]); rm = np.array([r[3] for r in rows])
    c = float((rp * rm).sum() / (rp * rp).sum()); corr = float(np.corrcoef(rp, rm)[0, 1])
    print(f"{os.path.basename(run):22s} layers={len(rows)} mean r_pred={rp.mean():.4f} mean r_meas={rm.mean():.4f} ratio meas/pred={rm.mean()/rp.mean():.2f} fit c={c:.2f} corr={corr:.3f}")
    allpts.append((os.path.basename(run), a["alphabet"], rp, rm))
    for layer, rs, p_, m_ in rows[:4]: print(f"    {layer:14s} rho(theta)={rs:.3f} pred={p_:.4f} meas={m_:.4f}")

if plot_out:                                                   # optional scatter plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # ---- plot style -------------------------------------------------------------------------------------------
    TW = 345.0 / 72.27                      # 345 pt in inches
    CW = 252.0 / 72.27                      # 252 pt in inches (plot width)
    PW = 522.0 / 72.27                      # 522 pt in inches
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 7, "axes.labelsize": 7,
                         "axes.titlesize": 7.5, "axes.titlelocation": "left", "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
                         "lines.markersize": 3.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    TER, BIN, BF16, GREY = "#0072B2", "#D55E00", "#000000", "#767676"   # Okabe-Ito blue / vermillion, black, grey
    C = {"ternary": TER, "binary": BIN}; MK = {"ternary": "o", "binary": "s", "BF16": "^"}
    # -------------------------------------------------------------------------------------------------------------
    def config(name):                     # run family -> (marker, legend label); ternary filled, binary open
        if name.startswith("M_"): return "o", "size M"
        if "owt" in name: return "^", "OWT"
        if "gpt2" in name: return "D", "GPT-2-style"
        if "_a8" in name: return "v", "INT8 act."
        if "g1.25" in name: return "v", r"$\gamma=1.25$"
        return "X", name
    fig, ax = plt.subplots(figsize=(CW, 3.0), layout="constrained")
    hand = {"ternary": [], "binary": []}
    for name, alph, rp, rm in allpts:
        mk, lab = config(name)
        ax.plot(1e3 * rp, 1e3 * rm, mk, ms=3.2, mew=0.7, alpha=0.85, color=C[alph], mfc=C[alph] if alph == "ternary" else "white", zorder=3)
        hand[alph].append(Line2D([], [], ls="none", marker=mk, ms=3.5, mew=0.7, color=C[alph], mfc=C[alph] if alph == "ternary" else "white", label=f"{alph}, {lab}"))
    lim = [0, 1e3 * max(max(rp.max(), rm.max()) for _, _, rp, rm in allpts) * 1.05]
    ax.plot(lim, lim, "k-", lw=0.7, zorder=1); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(r"Rice-law prediction ($10^{-3}$ per step)"); ax.set_ylabel(r"measured flip rate ($10^{-3}$ per step)")
    ax.set_xticks([0, 4, 8, 12]); ax.set_yticks([0, 4, 8, 12])
    h = hand["ternary"] + hand["binary"]
    fig.legend(handles=h, loc="outside lower center", ncol=2, columnspacing=0.6, handletextpad=0.2, handlelength=1.0)
    fig.savefig(plot_out); print("saved", plot_out)
