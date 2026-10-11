"""Per-block attribution of the dither gap: apply dither-free (time-averaged) weights to a subset of layers only.
final_state.pt holds no counts for a fresh window, so this script reloads the run's final weights, runs a K-step
cool-down at the final learning rate with level counting on, and evaluates dither-free overrides for: no layers
(snapshot), all layers, attention only, MLP only, and each block on its own. Writes <run>/attribute_K.json.
python analysis/attribute.py runs/S_ter_g1.0_s0 --steps 64
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.llama_small import Llama, SIZES
from quant.ste import quant_modules
from train import Shards, evaluate

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--steps", type=int, default=64); ap.add_argument("--eval_batches", type=int, default=400)
a = ap.parse_args(); args = json.load(open(os.path.join(a.run, "args.json"))); device = "cuda"
torch.manual_seed(args["seed"] + 2000)
qkw = dict(alphabet=args["alphabet"], scale_mult=args["gamma"], deadzone_frac=args["dz"], stochastic=False, act_bits=args.get("act_bits"))
model = Llama(seq=args["seq"], quant_kwargs=qkw, arch=args.get("arch", "llama"), **SIZES[args["size"]]).to(device)
st = torch.load(os.path.join(a.run, "final_state.pt"), weights_only=False); sd = model.state_dict()
for n, p in st.pop("_other").items(): sd[n].copy_(p.float())
qm = dict(quant_modules(model))
for n, d in st.items(): qm[n].weight.data.copy_(d["latent"].float().to(device))
lr = args["lr"] * args["min_lr_frac"]; wd = 0.0 if args.get("wd_off_frac", 0) > 0 else args["wd"]
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]; nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95), fused=True)
train, val = Shards(args["data"], "train"), Shards(args["data"], "val"); rng = np.random.default_rng(args["seed"] + 2000)
accum = args["batch_tokens"] // (args["micro_bs"] * args["seq"])
for m in qm.values(): m.start_counting()
model.train()
for step in range(a.steps):
    for m in qm.values(): m.tick = True
    for _ in range(accum):
        x, y = train.batch(rng, args["micro_bs"], args["seq"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16): loss = model(x, y) / accum
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
bars = {n: m.duty_cycle_tensors()[1] for n, m in qm.items()}
def ev(names):
    for n, m in qm.items(): m.override = bars[n] if n in names else None
    v = evaluate(model, val, args["micro_bs"], args["seq"], a.eval_batches, device)
    for m in qm.values(): m.override = None
    return v
out = dict(snapshot=ev(set()), all=ev(set(qm)), attn=ev({n for n in qm if n.split(".")[-1] in ("wq", "wk", "wv", "wo")}), mlp=ev({n for n in qm if n.split(".")[-1] in ("w1", "w2", "w3")}))
nblk = SIZES[args["size"]]["n_layer"]
for b in range(nblk): out[f"block{b}"] = ev({n for n in qm if n.startswith(f"blocks.{b}.")})
# per-layer fraction of weights that visited exactly two levels in the window
out["two_level_frac"] = {n: float(((m.counts > 0).sum(0) == 2).float().mean()) for n, m in qm.items()}
json.dump(out, open(os.path.join(a.run, f"attribute_{a.steps}.json"), "w"), indent=1)
print({k: (round(v, 4) if isinstance(v, float) else "...") for k, v in out.items()})
