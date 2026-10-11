"""Text summary of training curves from log.jsonl. For each run directory given, prints the training loss at
steps 10, 30, 50, 100, 150, 200, 250, 285, all validation losses, the mean flip rate at steps 100 and 250 and
the mean zero fraction at step 250.
python analysis/curves.py runs/S_ter_g1.0_s0 runs/S_bin_g1.0_s0
"""
import json, sys, math
for n in sys.argv[1:]:
    L = [json.loads(l) for l in open(f"{n}/log.jsonl")]
    tr = {r["step"]: r["loss"] for r in L if "loss" in r}
    va = {r["step"]: r["val_loss"] for r in L if "val_loss" in r}
    fl = {r["step"]: r.get("flip_mean", 0) for r in L if "loss" in r}
    zs = {r["step"]: r.get("zero_mean", 0) for r in L if "loss" in r}
    pts = [10, 30, 50, 100, 150, 200, 250, 285]
    print(f"{n.split('/')[-1]:16s} train:", " ".join(f"{s}:{tr.get(s, math.nan):.2f}" for s in pts),
          "| val:", {k: round(v, 3) for k, v in va.items()}, "| flip@100/250:", round(fl.get(100, 0), 4), round(fl.get(250, 0), 4), "| zero@250:", round(zs.get(250, 0), 3))
