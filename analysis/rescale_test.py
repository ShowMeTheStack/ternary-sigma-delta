"""Scale-invariance test: evaluate a trained run with all quantized (snapshot) weights multiplied by c, for each
c in --scales, and with only the attention, only the MLP, or only the query/key weights multiplied by 2.
Writes <run>/rescale.json.
python analysis/rescale_test.py runs/S_ter_g1.0_s0
"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.llama_small import Llama, SIZES
from quant.ste import quant_modules
from train import Shards, evaluate
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--eval_batches", type=int, default=400)
ap.add_argument("--scales", type=float, nargs="*", default=[0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]); a = ap.parse_args()
args = json.load(open(os.path.join(a.run, "args.json"))); device = "cuda"
qkw = dict(alphabet=args["alphabet"], scale_mult=args["gamma"], deadzone_frac=args["dz"], stochastic=False, act_bits=args.get("act_bits"))
model = Llama(seq=args["seq"], quant_kwargs=qkw, arch=args.get("arch", "llama"), **SIZES[args["size"]]).to(device).eval()
st = torch.load(os.path.join(a.run, "final_state.pt"), weights_only=False); sd = model.state_dict()
for n, p in st.pop("_other").items(): sd[n].copy_(p.float())
qm = dict(quant_modules(model))
for n, d in st.items(): qm[n].weight.data.copy_(d["latent"].float().to(device))
val = Shards(args["data"], "val"); out = {}
with torch.no_grad():
    wq = {n: m.quantize(m.weight)[0] for n, m in qm.items()}
    for c in a.scales:
        for n, m in qm.items(): m.override = wq[n] * c
        out[str(c)] = evaluate(model, val, args["micro_bs"], args["seq"], a.eval_batches, device)
        for m in qm.values(): m.override = None
    # attention-only and MLP-only rescale at c=2 to locate the residual scale dependence
    for tag, keys in [("attn", ("wq", "wk", "wv", "wo")), ("mlp", ("w1", "w2", "w3")), ("qk", ("wq", "wk"))]:
        for n, m in qm.items(): m.override = wq[n] * (2.0 if n.split(".")[-1] in keys else 1.0)
        out[f"c2_{tag}"] = evaluate(model, val, args["micro_bs"], args["seq"], a.eval_batches, device)
        for m in qm.values(): m.override = None
json.dump(out, open(os.path.join(a.run, "rescale.json"), "w"), indent=1)
print("RESCALE", {k: round(v, 4) for k, v in out.items()})
