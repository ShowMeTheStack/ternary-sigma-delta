"""Offline re-evaluation of a finished CIFAR run from final_state.pt. For each deployed variant (snapshot / mode /
dither-free / EMA / latent) the BN statistics are re-estimated on *augmented* training data before measuring
test accuracy and loss.
python analysis/cifar_reeval.py runs/cifar/ter_s0 [--batches 200]   -> runs/cifar/ter_s0/reeval.json
"""
import argparse, json, os, sys
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cifar.train_cifar as tc
from cifar.train_cifar import ResNet18, load_cifar, augment
norm = lambda x: tc.norm(x)
import torch.nn.functional as F

@torch.no_grad()
def acc_loss(model, X, Y, device, bs=1000):
    model.eval(); c = 0; tot = 0.0
    for i in range(0, len(X), bs):
        x = norm(X[i:i + bs].to(device)); y = Y[i:i + bs].to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = model(x)
        c += int((out.argmax(1) == y).sum()); tot += float(F.cross_entropy(out.float(), y, reduction="sum"))
    model.train(); return c / len(X), tot / len(X)
from quant.ste import quant_modules
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--batches", type=int, default=200); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=123)
a = ap.parse_args()
args = json.load(open(os.path.join(a.run, "args.json"))); device = a.device
(Xtr, Ytr), (Xte, Yte) = load_cifar(args["data"], args.get("dataset", "cifar10")); Xtr = Xtr.to(device)
qkw = dict(alphabet=args["alphabet"], scale_mult=args["gamma"], deadzone_frac=args["dz"], scale_rule=args.get("scale_rule", "absmean"))
model = ResNet18(qkw, n_cls=tc.N_CLS).to(device).to(memory_format=torch.channels_last)
st = torch.load(os.path.join(a.run, "final_state.pt"), map_location=device)
if "_other" in st:
    model.load_state_dict({k: v.float() for k, v in st["_other"].items()}, strict=False)
else:   # full-precision run: final_state.pt holds every parameter
    model.load_state_dict({k: v.float() for k, v in st.items()}, strict=False)
qmods = quant_modules(model)
srcs = {"snapshot": {}, "mode": {}, "dither_free": {}, "ema_requant": {}, "latent_fp": {}}
with torch.no_grad():
    for n, m in qmods:
        if n not in st: continue
        v = st[n]; lat = v["latent"].float().to(device); m.weight.copy_(lat)
        s = m.scale(m.weight).detach(); c = v["counts"].float().to(device); W = int(c.sum(0)[0])
        levels = torch.arange(-m.half, m.half + 1, device=device, dtype=c.dtype)
        srcs["snapshot"][n] = m.quantize(m.weight)[0]
        srcs["mode"][n] = ((c.argmax(0).float() - m.half) * s).view_as(m.weight)
        srcs["dither_free"][n] = ((levels[:, None] * c).sum(0) / W * s).view_as(m.weight)
        srcs["ema_requant"][n] = m.quantize(v["ema"].float().to(device))[0]
        srcs["latent_fp"][n] = lat
gen = torch.Generator(device=device); gen.manual_seed(a.seed)
@torch.no_grad()
def recal():
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d): mod.reset_running_stats(); mod.momentum = None
    model.train()
    perm = torch.randperm(len(Xtr), device=device, generator=gen)
    for i in range(a.batches):
        idx = perm[i * a.bs:(i + 1) * a.bs]
        x = norm(augment(Xtr[idx], gen)).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")): model(x)
out = {"batches": a.batches, "augmented": True}
for k, src in srcs.items():
    if not src and k != "snapshot": continue
    for n, m in qmods: m.override = src[n]
    recal(); out[k], out[k + "_loss"] = acc_loss(model, Xte, Yte, device)
    for n, m in qmods: m.override = None
json.dump(out, open(os.path.join(a.run, "reeval.json"), "w"), indent=1)
print(a.run, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
