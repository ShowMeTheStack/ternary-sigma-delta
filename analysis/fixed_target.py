"""Dither and rounding penalties on a fixed target. Take the weights of a BF16 run (or, with --target, the
dither-free weights saved by probe.py --save_bars for a ternary run) and, for each gamma, evaluate a simulated
duty-cycle snapshot (each weight drawn from its two bracketing ternary levels a < b with probability
p = (w-a)/(b-a) for the upper level, --draws draws) and nearest-level rounding. The excess val loss over the
unquantized target as a function of gamma can be compared with G_T(gamma) (dither) and M_T(gamma) (rounding)
from predict_gamma.py, up to a curvature constant.
Writes fixed_target_bf16.json into the BF16 run, or fixed_target_bars.json into --target_run.

python analysis/fixed_target.py runs/S_fp_s0 --gammas 0.5 0.75 1.0 1.25 1.5 2.0 2.5 --draws 3
"""
import argparse, json, os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.llama_small import Llama, SIZES
from quant.ste import QuantLinear
from train import Shards, evaluate
ap = argparse.ArgumentParser(); ap.add_argument("fp_run"); ap.add_argument("--gammas", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5])
ap.add_argument("--draws", type=int, default=3); ap.add_argument("--eval_batches", type=int, default=200)
ap.add_argument("--target", default=None, help="bars_K.pt from a ternary run's probe: use its dither-free weights as the fixed target (and its _other params)")
ap.add_argument("--target_run", default=None, help="the ternary run directory that produced --target"); a = ap.parse_args()
args = json.load(open(os.path.join(a.fp_run, "args.json"))); device = "cuda"
model = Llama(seq=args["seq"], quant_kwargs=dict(alphabet="ternary"), arch=args.get("arch", "llama"), **SIZES[args["size"]]).to(device).eval()
qm = {n: m for n, m in model.named_modules() if isinstance(m, QuantLinear)}
sd = model.state_dict()
if a.target:
    tst = torch.load(os.path.join(a.target_run, "final_state.pt"), weights_only=False)
    for n, p in tst.pop("_other").items(): sd[n].copy_(p.float())
    bars = torch.load(a.target, weights_only=False)
    for n, b in bars.items(): qm[n].weight.data.copy_(b.float().to(device))
    tag = "bars"
else:
    st = torch.load(os.path.join(a.fp_run, "final_state.pt"), weights_only=False)
    for n, p in st.items(): sd[n].copy_(p.float())
    tag = "bf16"
val = Shards(args["data"], "val"); g = torch.Generator(device=device).manual_seed(0)
def ev(): return evaluate(model, val, args["micro_bs"], args["seq"], a.eval_batches, device)
for m in qm.values(): m.override = m.weight.detach()
out = {"target": ev(), "target_kind": tag}
with torch.no_grad():
    for gam in a.gammas:
        dith = []; 
        for d in range(a.draws):
            for n, m in qm.items():
                w = m.weight; s = gam * w.abs().mean(); u = (w / s).clamp(-1, 1)
                lo = torch.floor(u); hi = lo + 1; p = u - lo                       # bracketing levels in {-1,0,1}, duty cycle p
                lv = torch.where(torch.rand(w.shape, generator=g, device=device) < p, hi, lo).clamp(-1, 1)
                m.override = lv * s
            dith.append(ev())
        for n, m in qm.items():
            w = m.weight; s = gam * w.abs().mean(); m.override = torch.round((w / s).clamp(-1, 1)) * s   # nearest level = mode rounding
        rnd = ev()
        out[f"{gam:.2f}"] = dict(dither_mean=float(np.mean(dith)), dither_std=float(np.std(dith)), rounded=rnd,
                                 zero_frac=float(np.mean([float((torch.round((m.weight / (gam * m.weight.abs().mean())).clamp(-1, 1)) == 0).float().mean()) for m in qm.values()])))
        print(f"gamma={gam:.2f} dither={np.mean(dith):.4f}±{np.std(dith):.4f} rounded={rnd:.4f} zero={out[f'{gam:.2f}']['zero_frac']:.3f} (target {out['target']:.4f})", flush=True)
for m in qm.values(): m.override = None
json.dump(out, open(os.path.join(a.target_run or a.fp_run, f"fixed_target_{tag}.json"), "w"), indent=1)
