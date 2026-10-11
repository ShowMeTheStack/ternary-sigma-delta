"""
Summarize finished language-model runs.

python analysis/lm_summary.py runs/sweep            -> table of final metrics per run directory under runs/sweep
python analysis/lm_summary.py runs/sweep --curves   -> also dither variance and flip-rate curves from final_state.pt

Per run (from final.json): val loss for snapshot / mode / dither-free / ema-requant / latent-fp,
mean zero fraction, mean flip rate in the final window, fraction of two-level and static weights.
Curves (from final_state.pt): for each quant layer, duty-cycle estimate  w*_hat = qbar = s*(c+ - c-)/W,
flip rate per weight = flipc/(W-1), binned by |w*_hat|/s, averaged over layers, printed next to the
noise-free closed form (1 - x binary, 2 min(x, 1 - x) ternary).
"""
import argparse, glob, json, os, sys
import numpy as np


def load_runs(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        f = os.path.join(d, "final.json")
        if not os.path.isdir(d) or not os.path.exists(f):
            continue
        a = json.load(open(os.path.join(d, "args.json")))
        r = json.load(open(f))
        r["name"] = os.path.basename(d); r["args"] = a
        # best val during training from log
        vals = [json.loads(l).get("val_loss") for l in open(os.path.join(d, "log.jsonl")) if "val_loss" in l]
        r["val_min_train"] = min(vals) if vals else None
        out.append(r)
    return out


def table(runs):
    hdr = f"{'run':28s} {'alph':7s} {'lr':>8s} {'gam':>4s} {'snap':>7s} {'mode':>7s} {'dfree':>7s} {'ema':>7s} {'latFP':>7s} {'zero':>6s} {'flipW':>7s} {'2lvl':>6s} {'static':>6s}"
    print(hdr); print("-" * len(hdr))
    for r in runs:
        a = r["args"]; pl = r.get("per_layer", {})
        two = np.mean([v["frac_two_level"] for v in pl.values()]) if pl else float("nan")
        st = np.mean([v["frac_static"] for v in pl.values()]) if pl else float("nan")
        g = lambda k: f"{r[k]:7.4f}" if k in r else f"{'-':>7s}"
        print(f"{r['name']:28s} {a['alphabet']:7s} {a['lr']:8.1e} {a['gamma']:4.2f} {g('val_snapshot')} {g('val_mode')} {g('val_dither_free')} "
              f"{g('val_ema_requant')} {g('val_latent_fp')} {r.get('mean_zero_frac', float('nan')):6.3f} {r.get('mean_flip_rate_window', float('nan')):7.4f} {two:6.3f} {st:6.3f}")


def dither_stats(root, runs):
    """Per-weight dither variance Var(q)/s^2 from window counts, averaged per run, printed next to the
    measured loss gap (snapshot - dither-free)."""
    import torch
    out = {}
    for r in runs:
        if r["args"]["alphabet"] == "none":
            continue
        st = torch.load(os.path.join(root, r["name"], "final_state.pt"), weights_only=False)
        tot = 0.0; n = 0
        for nme, d in st.items():
            if not isinstance(d, dict) or "counts" not in d: continue
            c = d["counts"].float(); W = c.sum(0)[0]
            p_minus, p_plus = c[0] / W, c[2] / W
            qbar = p_plus - p_minus                       # in units of s
            var = (p_plus + p_minus) - qbar ** 2           # E[q^2]-qbar^2 in units of s^2
            tot += float(var.sum()); n += var.numel()
        out[r["name"]] = tot / n
        gap = r.get("val_snapshot", float("nan")) - r.get("val_dither_free", float("nan"))
        print(f"{r['name']:28s} mean Var(q)/s^2 = {tot/n:.5f}   measured loss gap snapshot-ditherfree = {gap:+.5f}")
    return out


def curves(root, runs, nbins=30):
    import torch
    for r in runs:
        if r["args"]["alphabet"] == "none":
            continue
        st = torch.load(os.path.join(root, r["name"], "final_state.pt"), weights_only=False)
        edges = np.linspace(0, 1.5, nbins + 1); num = np.zeros(nbins); den = np.zeros(nbins); numz = np.zeros(nbins)
        for n, d in st.items():
            if not isinstance(d, dict) or "counts" not in d: continue
            c = d["counts"].float(); W = int(c.sum(0)[0]); s = r["per_layer"][n]["scale"]
            qbar = (c[2] - c[0]) / W                       # in units of s
            flip = d["flipc"].float() / max(1, W - 1)
            duty0 = c[1] / W
            x = qbar.abs().numpy(); idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
            np.add.at(num, idx, flip.numpy()); np.add.at(den, idx, 1); np.add.at(numz, idx, duty0.numpy())
        mid = (edges[:-1] + edges[1:]) / 2
        kind = r["args"]["alphabet"]
        th = (1 - mid) if kind == "binary" else 2 * np.minimum(mid, 1 - mid)
        th = np.where(mid < 1, th, 0)
        print(f"\n== {r['name']} flip rate vs |w*_hat|/s (window duty-cycle estimate) ==")
        print("bin   ", " ".join(f"{m:5.2f}" for m in mid[:20]))
        print("flip  ", " ".join(f"{v:5.3f}" for v in (num / np.maximum(den, 1))[:20]))
        print("theory", " ".join(f"{v:5.3f}" for v in th[:20]))
        print("P(q=0)", " ".join(f"{v:5.3f}" for v in (numz / np.maximum(den, 1))[:20]))
        print("count ", " ".join(f"{int(v):5d}" for v in den[:20]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--curves", action="store_true")
    a = ap.parse_args()
    runs = load_runs(a.root)
    table(runs)
    if a.curves:
        dither_stats(a.root, runs)
        curves(a.root, runs)
