"""Zero-shot evaluation of a trained run: LAMBADA (last-word accuracy, OpenAI variant), HellaSwag, PIQA and
ARC-Easy (length-normalized log-likelihood, multiple choice) and WinoGrande (partial scoring).
Applies snapshot / mode / ema weights.

python analysis/zeroshot.py runs/L_ter_g1.0_s0 --which snapshot mode ema --n 2000
"""
import argparse, json, os, sys
import numpy as np, torch, tiktoken
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.llama_small import Llama, SIZES
from quant.ste import quant_modules
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")

@torch.no_grad()
def logprob_continuation(model, ctx_ids, cont_ids, device):
    ids = torch.tensor([ctx_ids + cont_ids], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids[:, :-1]).float()
    lp = torch.log_softmax(logits[0], -1)
    tgt = ids[0, 1:]
    n = len(cont_ids)
    return float(lp[-n:].gather(1, tgt[-n:, None]).sum()), n

def lambada(model, device, n):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    correct = 0; total = 0
    for ex in ds.select(range(min(n, len(ds)))):
        text = ex["text"]; ctx, last = text.rsplit(" ", 1)
        ctx_ids = enc.encode_ordinary(ctx); cont_ids = enc.encode_ordinary(" " + last)
        ids = torch.tensor([ctx_ids], device=device)
        gen = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(len(cont_ids)):
                nxt = int(model(ids)[0, -1].argmax()); gen.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], 1)
        correct += int(gen == cont_ids); total += 1
    return correct / total

def hellaswag(model, device, n):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    correct = 0; total = 0
    for ex in ds.select(range(min(n, len(ds)))):
        ctx_ids = enc.encode_ordinary(ex["ctx"])
        scores = []
        for end in ex["endings"]:
            lp, k = logprob_continuation(model, ctx_ids, enc.encode_ordinary(" " + end), device)
            scores.append(lp / k)
        correct += int(int(np.argmax(scores)) == int(ex["label"])); total += 1
    return correct / total

def _mc(model, device, items, n):
    """Multiple choice by length-normalized continuation log-likelihood. items: iterable of (ctx, [choices], label)."""
    correct = 0; total = 0
    for ctx, choices, label in items:
        if total >= n: break
        ctx_ids = enc.encode_ordinary(ctx)
        scores = [logprob_continuation(model, ctx_ids, enc.encode_ordinary(" " + c), device) for c in choices]
        scores = [lp / k for lp, k in scores]
        correct += int(int(np.argmax(scores)) == int(label)); total += 1
    return correct / max(1, total)

def piqa(model, device, n):
    ds = load_dataset("ybisk/piqa", split="validation", revision="refs/convert/parquet")
    return _mc(model, device, ((ex["goal"], [ex["sol1"], ex["sol2"]], ex["label"]) for ex in ds), n)

def arc_easy(model, device, n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    def items():
        for ex in ds:
            labels = ex["choices"]["label"]; yield ("Question: " + ex["question"] + "\nAnswer:", ex["choices"]["text"], labels.index(ex["answerKey"]))
    return _mc(model, device, items(), n)

def winogrande(model, device, n):
    """Partial-evaluation scoring: the option fills the blank, the text after the blank is the scored continuation."""
    ds = load_dataset("allenai/winogrande", "default", split="validation", revision="refs/convert/parquet")   # all sizes share the 1267 validation items
    correct = 0; total = 0; seen = set()
    for ex in ds:
        if total >= n: break
        if ex["sentence"] in seen: continue
        seen.add(ex["sentence"])
        pre, post = ex["sentence"].split("_", 1)
        scores = []
        for opt in (ex["option1"], ex["option2"]):
            ctx_ids = enc.encode_ordinary(pre + opt); cont_ids = enc.encode_ordinary(post)
            lp, k = logprob_continuation(model, ctx_ids, cont_ids, device); scores.append(lp)
        correct += int(int(np.argmax(scores)) + 1 == int(ex["answer"])); total += 1
    return correct / max(1, total)

TASKS = dict(lambada=lambada, hellaswag=hellaswag, piqa=piqa, arc_easy=arc_easy, winogrande=winogrande)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--which", nargs="+", default=["snapshot", "mode"])
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--tasks", nargs="+", default=["lambada", "hellaswag"])
    ap.add_argument("--out", default="zeroshot.json"); a = ap.parse_args()
    args = json.load(open(os.path.join(a.run, "args.json"))); device = "cuda"
    qkw = dict(alphabet=args["alphabet"], scale_mult=args["gamma"], deadzone_frac=args["dz"], stochastic=False, act_bits=args["act_bits"], scale_rule=args.get("scale_rule", "absmean"))
    model = Llama(seq=args["seq"], quant_kwargs=qkw, arch=args.get("arch", "llama"), **SIZES[args["size"]]).to(device).eval()
    st = torch.load(os.path.join(a.run, "final_state.pt"), weights_only=False)
    qm = dict(quant_modules(model))
    sd = model.state_dict()
    if args["alphabet"] == "none":
        for n, p in st.items(): sd[n].copy_(p.float())
        variants = {"snapshot": None}
    else:
        for n, p in st.pop("_other").items(): sd[n].copy_(p.float())
        for n, d in st.items(): qm[n].weight.data.copy_(d["latent"].float().to(device))
        variants = {}
        for w in a.which:
            if w == "ema" and any(d.get("ema") is None for d in st.values()): print("no EMA state in this run; skipping ema", flush=True); continue
            ov = {}
            for n, d in st.items():
                m = qm[n]; s = m.scale(m.weight)
                if w == "snapshot": ov[n] = None
                elif w == "mode": ov[n] = ((d["counts"].to(device).float().argmax(0).float() - 1) * s).view_as(m.weight)
                elif w == "ema": ov[n] = m.quantize(d["ema"].float().to(device))[0]
            variants[w] = ov
    outp = os.path.join(a.run, a.out)
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    for w, ov in variants.items():
        if ov:
            for n, m in qm.items(): m.override = ov[n]
        res.setdefault(w, {})
        for t in a.tasks:
            res[w][t] = TASKS[t](model, device, a.n); print(w, t, res[w][t], flush=True)
            json.dump(res, open(outp, "w"), indent=1)
        for m in qm.values(): m.override = None

if __name__ == "__main__":
    main()
