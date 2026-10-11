"""Predicted ratio of mode-rounding gain to dither gap from duty cycles, R = E[min(p,1-p)^2] / E[p(1-p)]
over dithering weights (0<p<1), vs the measured ratio (snapshot-mode)/(snapshot-dither_free) from final.json.
Needs runs trained with --cooldown > 0.
python analysis/mode_ratio.py runs/M_ter_g1.0_s0 [more runs...]
"""
import json, os, sys, torch, numpy as np
for run in sys.argv[1:]:
    fin = json.load(open(os.path.join(run, "final.json")))
    if fin.get("cooldown", 0) == 0: print(run, "no live cool-down; skip"); continue
    st = torch.load(os.path.join(run, "final_state.pt"), weights_only=False)
    num = 0.0; den = 0.0; n_dith = 0; n_all = 0; per_layer = {}
    for n, d in st.items():
        if n == "_other": continue
        c = d["counts"].float(); W = c.sum(0)[0]
        half = c.shape[0] // 2
        # two-level assumption: p = fraction at the upper visited level; use visited pair (levels with count>0)
        visited = (c > 0).sum(0)
        two = visited == 2
        if two.sum() == 0: continue
        cc = c[:, two]                                # levels x n2
        # upper visited level fraction: take the larger-index visited level's share
        idx_sorted = torch.argsort(cc, dim=0, descending=True)  # sorted by count (unused); p below is the share of the higher visited level
        lv = torch.arange(cc.shape[0], dtype=torch.float32)[:, None].expand_as(cc)
        hi_mask = (cc > 0) & (lv == (lv * (cc > 0)).max(0).values[None, :])
        p = (cc * hi_mask).sum(0) / W
        var = p * (1 - p); mode_err = torch.minimum(p, 1 - p) ** 2
        per_layer[n] = (float(mode_err.mean() / var.mean()), int(two.sum()))
        num += float(mode_err.sum()); den += float(var.sum()); n_dith += int(two.sum()); n_all += c.shape[1]
    R_pred = num / den
    gap = fin["val_snapshot"] - fin["val_dither_free"]; gain = fin["val_snapshot"] - fin["val_mode"]
    print(f"{os.path.basename(run):22s} dithering {n_dith/n_all:.3f}  R_pred={R_pred:.3f}  measured gain/gap={gain/gap:.3f}  (gain {gain:.4f}, gap {gap:.4f})")
