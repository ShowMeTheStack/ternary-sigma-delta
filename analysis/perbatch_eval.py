"""Per-batch validation loss of a finished run, for the standard error of the reported validation loss.

Loads the run's final_state.pt exactly as analysis/probe.py and analysis/zeroshot.py do (fp16 latents of the
quantized layers + fp16 other parameters; BF16 runs store a flat state dict), then evaluates the same validation
micro-batches the trainer's final evaluation used (train.Shards.sequential over the first val shard, micro_bs and
seq from args.json, final_eval_batches = 400, BF16 autocast) and records the mean loss of every micro-batch.

Reports mean, SE = std(batch means, ddof=1) / sqrt(n), the lag-1 autocorrelation of the batch means, and a
blocked SE (batch means of 20 consecutive micro-batches) as a check on correlation between adjacent batches
(the batches are contiguous windows of one shard, so adjacent batches can share a document). With --which
snapshot mode it also records the mode-rounded network and the SE of the paired per-batch difference.

Resumable: the per-batch list is flushed to the output file every 25 batches and reused on restart.

python analysis/perbatch_eval.py runs/S_ter_g1.0_s0 --device cuda:0
Writes <run>/perbatch_eval.json.
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.llama_small import Llama, SIZES
from quant.ste import quant_modules
from train import Shards


def stats(x, block=20):
    x = np.asarray(x, dtype=np.float64); n = len(x)
    se = float(x.std(ddof=1) / math.sqrt(n))
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    nb = n // block; bm = x[:nb * block].reshape(nb, block).mean(1)
    return dict(n=n, mean=float(x.mean()), std_batch=float(x.std(ddof=1)), se=se, lag1_autocorr=r1,
                se_blocked=float(bm.std(ddof=1) / math.sqrt(nb)), block=block, n_blocks=nb)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--which", nargs="+", default=["snapshot"], choices=["snapshot", "mode"])
    ap.add_argument("--n_batches", type=int, default=None, help="default: args.json final_eval_batches (400)")
    ap.add_argument("--data", default=None); ap.add_argument("--out", default="perbatch_eval.json")
    ap.add_argument("--loss_chunk", type=int, default=None, help="rows per head/CE chunk (1 for large models)")
    a = ap.parse_args()
    args = json.load(open(os.path.join(a.run, "args.json"))); device = a.device
    nb = a.n_batches or args.get("final_eval_batches", 400)
    bs, seq = args["micro_bs"], args["seq"]
    qkw = dict(alphabet=args["alphabet"], scale_mult=args["gamma"], deadzone_frac=args["dz"], stochastic=False,
               act_bits=args.get("act_bits"), scale_rule=args.get("scale_rule") or "absmean")
    model = Llama(seq=seq, quant_kwargs=qkw, arch=args.get("arch") or "llama", **SIZES[args["size"]]).to(device).eval()
    if a.loss_chunk: model.loss_chunk = a.loss_chunk
    elif args.get("lean"): model.loss_chunk = 1
    st = torch.load(os.path.join(a.run, "final_state.pt"), map_location="cpu", weights_only=False)
    sd = model.state_dict(); qm = dict(quant_modules(model))
    variants = {}
    if args["alphabet"] == "none":
        for n, p in st.items(): sd[n].copy_(p.float())
        variants["snapshot"] = None
    else:
        for n, p in st.pop("_other").items(): sd[n].copy_(p.float())
        for n, d in st.items(): qm[n].weight.data.copy_(d["latent"].float().to(device))
        for w in a.which:
            if w == "snapshot": variants[w] = None
            else:
                ov = {}
                for n, d in st.items():
                    m = qm[n]; s = m.scale(m.weight)
                    ov[n] = ((d["counts"].to(device).float().argmax(0).float() - 1) * s).view_as(m.weight).detach()
                variants[w] = ov
    del st
    val = Shards(a.data or args["data"], "val")
    outp = os.path.join(a.run, a.out)
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    meta = dict(run=os.path.abspath(a.run), size=args["size"], micro_bs=bs, seq=seq, n_batches=nb,
                tokens=nb * bs * seq, val_file=val.files[0], weights="final_state.pt (fp16 latents)", autocast="bf16")
    res["meta"] = meta
    t0 = time.time()
    with torch.no_grad():
        for w, ov in variants.items():
            rec = res.get(w, {}); losses = rec.get("per_batch", [])
            if len(losses) >= nb: continue
            for n, m in qm.items(): m.override = ov[n] if ov else None
            for i, (x, y) in enumerate(val.sequential(bs, seq, nb)):
                if i < len(losses): continue
                x, y = x.to(device), y.to(device)
                with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
                    losses.append(float(model(x, y)))
                if (i + 1) % 25 == 0 or i + 1 == nb:
                    res[w] = dict(per_batch=losses); json.dump(res, open(outp, "w"), indent=1)
                    print(f"{w} {i+1}/{nb} running mean {np.mean(losses):.4f} {time.time()-t0:.0f}s", flush=True)
            for m in qm.values(): m.override = None
    for w in variants:
        res[w].update(stats(res[w]["per_batch"]))
    if "snapshot" in variants and "mode" in variants:
        d = np.array(res["mode"]["per_batch"]) - np.array(res["snapshot"]["per_batch"])
        res["mode_minus_snapshot"] = stats(d)
    fj = os.path.join(a.run, "final.json")
    if os.path.exists(fj):
        f = json.load(open(fj)); res["final_json"] = {k: f.get("val_" + k) for k in variants}
    json.dump(res, open(outp, "w"), indent=1)
    for w in variants:
        r = res[w]; print(f"{a.run} {w}: mean {r['mean']:.4f} (final.json {res.get('final_json', {}).get(w)}) SE {r['se']:.4f} "
                          f"blocked SE {r['se_blocked']:.4f} lag1 {r['lag1_autocorr']:.3f} tokens {meta['tokens']}", flush=True)
    if "mode_minus_snapshot" in res:
        r = res["mode_minus_snapshot"]; print(f"mode - snapshot: {r['mean']:.4f} SE {r['se']:.4f} blocked {r['se_blocked']:.4f}")


if __name__ == "__main__":
    main()
