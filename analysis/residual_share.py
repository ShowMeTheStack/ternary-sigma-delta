"""Stream power entering each normalization of a trained decoder.
For each block, S_attn = power of (r + y_attn) entering n2, S_mlp = power of (r + y_attn + y_mlp) entering the next n1,
and the block-output shares mu = ||y||^2 / S. With two runs (e.g. ternary, binary) it also prints the mean ratio
<S_run0/S_run1> over layers.  python analysis/residual_share.py runs/S_ter_g1.0_s0 runs/S_bin_g1.0_s0 --device cpu
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.llama_small import Llama, SIZES
from quant.ste import quant_modules
from train import Shards
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--device", default="cuda")
ap.add_argument("--batches", type=int, default=4); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--data", default=None)
a = ap.parse_args()

def load(run):
    args = json.load(open(os.path.join(run, "args.json")))
    qkw = dict(alphabet=args["alphabet"], scale_mult=args["gamma"], deadzone_frac=args["dz"], stochastic=False, act_bits=args.get("act_bits"), scale_rule=args.get("scale_rule", "absmean"))
    model = Llama(seq=args["seq"], quant_kwargs=qkw, arch=args.get("arch", "llama"), **SIZES[args["size"]]).to(a.device)
    st = torch.load(os.path.join(run, "final_state.pt"), map_location=a.device, weights_only=False)
    sd = model.state_dict()
    if "_other" in st:
        for n, p in st.pop("_other").items(): sd[n].copy_(p.float())
        qm = dict(model.named_modules())
        for n, d in st.items(): qm[n].weight.data.copy_(d["latent"].float())
    else:
        for n, p in st.items(): sd[n].copy_(p.float())
    model.eval(); return model, args

def measure(model, args):
    val = Shards(a.data or args["data"], "val")
    P = {}   # per block: [S_in, y_attn, S_attn, y_mlp, S_mlp] power sums
    hooks = []
    for i, b in enumerate(model.blocks):
        P[i] = np.zeros(5)
        def mk(i):
            store = {}
            def pre(mod, inp): store["x"] = inp[0].detach().float()
            def post(mod, inp, out):
                x = store["x"]; xo = out.detach().float()
                # recompute the attention branch output to split the two residual additions
                B, T, D = x.shape
                h = mod.n1(x)
                q = mod.wq(h).view(B, T, mod.nh, mod.hd).transpose(1, 2); k = mod.wk(h).view(B, T, mod.nh, mod.hd).transpose(1, 2); v = mod.wv(h).view(B, T, mod.nh, mod.hd).transpose(1, 2)
                if mod.arch == "llama":
                    from model.llama_small import apply_rope
                    q, k = apply_rope(q, inp[1], inp[2]), apply_rope(k, inp[1], inp[2])
                att = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                ya = mod.wo(att.transpose(1, 2).reshape(B, T, D)).float()
                xa = x + ya; ym = xo - xa
                pw = lambda t: float((t ** 2).mean())
                P[i] += np.array([pw(x), pw(ya), pw(xa), pw(ym), pw(xo)])
            hooks.append(b.register_forward_pre_hook(pre)); hooks.append(b.register_forward_hook(post))
        mk(i)
    with torch.no_grad():
        for x, y in val.sequential(a.bs, args["seq"], a.batches):
            with torch.autocast(a.device, dtype=torch.bfloat16, enabled=(a.device == "cuda")):
                model(x.to(a.device))
    for h in hooks: h.remove()
    out = {}
    for i, v in P.items():
        v = v / a.batches
        out[f"block{i}"] = dict(S_in=v[0], y_attn=v[1], S_attn=v[2], y_mlp=v[3], S_mlp=v[4], mu_attn=v[1] / v[2], mu_mlp=v[3] / v[4])
    return out

res = {}
for run in a.runs:
    model, args = load(run); r = measure(model, args); res[run] = r
    print(run)
    for k, v in r.items(): print(f"  {k}: S_attn {v['S_attn']:.4f} S_mlp {v['S_mlp']:.4f} mu_attn {v['mu_attn']:.3f} mu_mlp {v['mu_mlp']:.3f}")
    json.dump(r, open(os.path.join(run, "residual_share.json"), "w"), indent=1)
if len(a.runs) == 2:
    r0, r1 = res[a.runs[0]], res[a.runs[1]]
    ratios = [r0[k]["S_attn"] / r1[k]["S_attn"] for k in r0] + [r0[k]["S_mlp"] / r1[k]["S_mlp"] for k in r0]
    print(f"<S_{{run0}}/S_{{run1}}> over layers: {np.mean(ratios):.4f} (min {min(ratios):.3f}, max {max(ratios):.3f})")
