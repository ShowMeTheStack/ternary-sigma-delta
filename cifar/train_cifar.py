"""
ResNet-18 on CIFAR-10 / CIFAR-100 with STE-quantized convolutions.

python cifar/train_cifar.py --alphabet ternary --seed 0 --out runs/cifar/ter_s0
Data: the python version of the dataset is downloaded into --data (default ~/data/cifar10) on first use;
no torchvision dependency.

Recipe: ResNet-18 (CIFAR stem 3x3/1, no max-pool), first conv and FC full precision, SGD momentum 0.9 (nesterov),
lr 0.1 cosine to lr*min_lr_frac, coupled weight decay, batch 128, random crop (pad 4) + horizontal flip, AMP bf16.
A cool-down of `cooldown` epochs at the final LR follows the schedule; level counts, flip counts and the latent EMA
are collected there and the deployment rules (snapshot / mode (most-visited level) / dither-free / EMA / latent) are
evaluated, each without and with BN recalibration (one pass over 20000 training images in train mode, no augmentation).
"""
import argparse, json, os, pickle, sys, tarfile, time, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.ste import QuantConv2d, quant_modules

URLS = {"cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "cifar100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"}
STATS = {"cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]), "cifar100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])}
MEAN = torch.tensor(STATS["cifar10"][0]).view(1, 3, 1, 1); STD = torch.tensor(STATS["cifar10"][1]).view(1, 3, 1, 1)
N_CLS = 10


def load_cifar(root, dataset="cifar10"):
    global MEAN, STD, N_CLS
    MEAN = torch.tensor(STATS[dataset][0]).view(1, 3, 1, 1); STD = torch.tensor(STATS[dataset][1]).view(1, 3, 1, 1)
    N_CLS = 10 if dataset == "cifar10" else 100
    root = os.path.expanduser(root)
    sub = "cifar-10-batches-py" if dataset == "cifar10" else "cifar-100-python"
    d = os.path.join(root, sub)
    if not os.path.isdir(d):
        os.makedirs(root, exist_ok=True); tgz = os.path.join(root, os.path.basename(URLS[dataset]))
        if not os.path.exists(tgz): urllib.request.urlretrieve(URLS[dataset], tgz)
        tarfile.open(tgz).extractall(root)
    key = "labels" if dataset == "cifar10" else "fine_labels"
    def rd(names):
        xs, ys = [], []
        for n in names:
            b = pickle.load(open(os.path.join(d, n), "rb"), encoding="latin1")
            xs.append(b["data"].reshape(-1, 3, 32, 32)); ys += b[key]
        return torch.from_numpy(np.concatenate(xs)), torch.tensor(ys, dtype=torch.long)
    if dataset == "cifar10": return rd([f"data_batch_{i}" for i in range(1, 6)]), rd(["test_batch"])
    return rd(["train"]), rd(["test"])


def norm(x_u8):
    return (x_u8.float() / 255.0 - MEAN.to(x_u8.device)) / STD.to(x_u8.device)


def augment(x_u8, gen):
    """Random crop with 4-pixel zero padding and horizontal flip, on the GPU."""
    B = x_u8.shape[0]
    xp = F.pad(x_u8, (4, 4, 4, 4))
    ox = torch.randint(0, 9, (B,), device=x_u8.device, generator=gen); oy = torch.randint(0, 9, (B,), device=x_u8.device, generator=gen)
    ar = torch.arange(32, device=x_u8.device)
    crop = torch.gather(xp, 2, (oy[:, None] + ar[None]).view(B, 1, 32, 1).expand(B, 3, 32, 40))
    crop = torch.gather(crop, 3, (ox[:, None] + ar[None]).view(B, 1, 1, 32).expand(B, 3, 32, 32))
    flip = torch.rand(B, device=x_u8.device, generator=gen) < 0.5
    crop = torch.where(flip[:, None, None, None], crop.flip(3), crop)
    return crop


# ---------------------------------------------------------------- model
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, mk):
        super().__init__()
        self.c1 = mk(cin, cout, 3, stride, 1); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = mk(cout, cout, 3, 1, 1); self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(mk(cin, cout, 1, stride, 0), nn.BatchNorm2d(cout))
    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + (x if self.sc is None else self.sc(x)))


class ResNet18(nn.Module):
    def __init__(self, qkw, n_cls=10):
        super().__init__()
        self._i = 0
        def mk(cin, cout, k, s, p):
            self._i += 1
            return QuantConv2d(cin, cout, k, s, p, name=f"conv{self._i}", **qkw)
        self.stem = nn.Conv2d(3, 64, 3, 1, 1, bias=False); self.bn = nn.BatchNorm2d(64)     # full precision
        cfg = [(64, 1), (128, 2), (256, 2), (512, 2)]; layers = []; cin = 64
        for cout, s in cfg:
            layers += [BasicBlock(cin, cout, s, mk), BasicBlock(cout, cout, 1, mk)]; cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, n_cls)                                                        # full precision
    def forward(self, x):
        x = F.relu(self.bn(self.stem(x)))
        x = self.layers(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


@torch.no_grad()
def accuracy(model, X, Y, device, bs=1000):
    model.eval(); c = 0
    for i in range(0, len(X), bs):
        x = norm(X[i:i + bs].to(device)); y = Y[i:i + bs].to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            c += int((model(x).argmax(1) == y).sum())
    model.train(); return c / len(X)


@torch.no_grad()
def bn_recalibrate(model, X, device, bs=500, n=20000):
    """Reset BN running stats and re-estimate them with the current (deployed) weights; no augmentation."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.reset_running_stats(); m.momentum = None   # cumulative average
    model.train()
    for i in range(0, n, bs):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            model(norm(X[i:i + bs].to(device)))
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.momentum = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphabet", default="ternary")
    ap.add_argument("--gamma", type=float, default=1.0); ap.add_argument("--dz", type=float, default=0.5)
    ap.add_argument("--scale_rule", default="absmean", choices=["absmean", "lsq", "twn"])
    ap.add_argument("--epochs", type=int, default=160); ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--min_lr_frac", type=float, default=0.01); ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9); ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--data", default="~/data/cifar10")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_batches", type=int, default=None)   # debugging
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    device = a.device; torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    (Xtr, Ytr), (Xte, Yte) = load_cifar(a.data, a.dataset)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    qkw = dict(alphabet=a.alphabet, scale_mult=a.gamma, deadzone_frac=a.dz, scale_rule=a.scale_rule)
    model = ResNet18(qkw, n_cls=N_CLS).to(device).to(memory_format=torch.channels_last)
    qmods = quant_modules(model)
    n_q = sum(m.weight.numel() for _, m in qmods); n_all = sum(p.numel() for p in model.parameters())
    print(f"params={n_all/1e6:.2f}M quantized={n_q/1e6:.2f}M layers={len(qmods)}", flush=True)
    decay = [p for n, p in model.named_parameters() if p.dim() > 1]; nodecay = [p for n, p in model.named_parameters() if p.dim() <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                          lr=a.lr, momentum=a.momentum, nesterov=True)
    n_per_epoch = len(Xtr) // a.bs
    if a.max_batches: n_per_epoch = min(n_per_epoch, a.max_batches)
    total = a.epochs * n_per_epoch; cool = a.cooldown * n_per_epoch; warm = a.warmup_epochs * n_per_epoch
    def lr_at(t):
        if t < warm: return a.lr * (t + 1) / warm
        if t >= total: return a.lr * a.min_lr_frac
        pr = (t - warm) / max(1, total - warm)
        return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + np.cos(np.pi * pr)))
    gen = torch.Generator(device=device); gen.manual_seed(a.seed)
    logf = open(os.path.join(a.out, "log.jsonl"), "a"); t0 = time.time(); step = 0; ema = None; final = {}
    for ep in range(a.epochs + a.cooldown):
        if ep == a.epochs:
            final["test_snapshot_end_of_schedule"] = accuracy(model, Xte, Yte, device)
            for _, m in qmods: m.start_counting(); m.record_traj = False
            ema = {n: m.weight.detach().float().clone() for n, m in qmods}; ema_beta = 1 - 2.0 / cool
        perm = torch.randperm(len(Xtr), device=device, generator=gen)
        model.train(); tl = 0.0; tc = 0
        for bi in range(n_per_epoch):
            idx = perm[bi * a.bs:(bi + 1) * a.bs]
            x = norm(augment(Xtr[idx], gen)).contiguous(memory_format=torch.channels_last); y = Ytr[idx]
            lr = lr_at(step)
            for g in opt.param_groups: g["lr"] = lr
            for _, m in qmods: m.tick = True
            with torch.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                out = model(x); loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            if ema is not None:
                with torch.no_grad():
                    for n, m in qmods: ema[n].mul_(ema_beta).add_(m.weight.float(), alpha=1 - ema_beta)
            tl += float(loss); tc += int((out.argmax(1) == y).sum())
        rec = dict(epoch=ep, step=step, lr=lr, train_loss=tl / n_per_epoch, train_acc=tc / (n_per_epoch * a.bs),
                   test_acc=accuracy(model, Xte, Yte, device), elapsed=time.time() - t0)
        if qmods:
            rec["flip_mean"] = float(np.mean([m.pop_flip_rate() for _, m in qmods]))
            rec["zero_mean"] = float(np.mean([float(m.last_zero_frac) for _, m in qmods]))
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        print(f"ep {ep} loss {rec['train_loss']:.3f} train {rec['train_acc']:.4f} test {rec['test_acc']:.4f} lr {lr:.4f} "
              f"flip {rec.get('flip_mean', 0):.5f} zero {rec.get('zero_mean', 0):.3f} {rec['elapsed']/60:.1f}min", flush=True)

    # ------------------------------------------------------------ final evaluations
    final.update(n_params=n_all, n_quant=n_q, steps=step, elapsed=time.time() - t0)
    final["test_snapshot"] = accuracy(model, Xte, Yte, device)
    if qmods:
        per_layer, modes, bars, emas = {}, {}, {}, {}
        with torch.no_grad():
            for n, m in qmods:
                mode, qbar = m.duty_cycle_tensors(); modes[n], bars[n] = mode, qbar
                emas[n] = m.quantize(ema[n])[0]; wq = m.quantize(m.weight)[0]
                c = m.counts.float(); W = m.count_steps; duty = c / W
                w = m.weight.detach().float().flatten(); wn = (w - w.mean()) / w.std()
                sc = float(m.scale(m.weight)); thr = m.threshold(m.weight, m.scale(m.weight)) if m.alphabet == "ternary" else 0.0
                per_layer[n] = dict(
                    numel=m.weight.numel(), scale=float(m.scale(m.weight)), zero_frac=float((wq == 0).float().mean()),
                    zero_frac_from_duty=float(duty[m.half].mean()), flip_rate_window=float(m.flipc.float().mean() / max(1, W - 1)),
                    frac_two_level=float(((c > 0).sum(0) == 2).float().mean()), frac_three_level=float(((c > 0).sum(0) == 3).float().mean()),
                    frac_static=float(((c > 0).sum(0) == 1).float().mean()), mode_eq_snapshot=float((mode == wq).float().mean()),
                    ema_eq_mode=float((torch.sign(emas[n]) == torch.sign(mode)).float().mean()),   # compares levels, not values (scales differ)
                    mean_abs_step=m.mean_abs_step(), kurtosis=float((wn ** 4).mean()),
                    abs_mean=float(w.abs().mean()), std=float(w.std()),
                    dens_at_threshold=(float(((w.abs() - float(thr)).abs() < 0.02 * sc).float().mean() / (0.04 * sc)) if m.alphabet == "ternary" else None),
                    dens_at_zero=float((w.abs() < 0.02 * sc).float().mean() / (0.04 * sc)),
                )
        bn_state = {k: v.clone() for k, v in model.state_dict().items() if "running" in k or "num_batches" in k}
        def with_override(src, recal):
            for n, m in qmods: m.override = src[n]
            if recal: bn_recalibrate(model, Xtr, device)
            v = accuracy(model, Xte, Yte, device)
            for n, m in qmods: m.override = None
            if recal: model.load_state_dict(bn_state, strict=False)
            return v
        srcs = dict(mode=modes, dither_free=bars, ema_requant=emas, latent_fp={n: m.weight.detach() for n, m in qmods},
                    snapshot={n: m.quantize(m.weight)[0] for n, m in qmods})
        for k, src in srcs.items():
            final[f"test_{k}"] = with_override(src, False); final[f"test_{k}_bnrecal"] = with_override(src, True)
        final["per_layer"] = per_layer
        for key in ("zero_frac", "flip_rate_window", "mean_abs_step", "kurtosis", "frac_static", "mode_eq_snapshot"):
            final[f"mean_{key}"] = float(np.mean([v[key] for v in per_layer.values()]))
        qnames = {n + ".weight" for n, _ in qmods}
        state = {n: dict(latent=m.weight.detach().half().cpu(), counts=m.counts.cpu(), flipc=m.flipc.cpu(), ema=ema[n].half().cpu()) for n, m in qmods}
        state["_other"] = {n: p.detach().half().cpu() for n, p in model.named_parameters() if n not in qnames}
        torch.save(state, os.path.join(a.out, "final_state.pt"))
    else:
        torch.save({n: p.detach().half().cpu() for n, p in model.named_parameters()}, os.path.join(a.out, "final_state.pt"))
    json.dump(final, open(os.path.join(a.out, "final.json"), "w"), indent=1)
    print("FINAL", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in final.items() if not isinstance(v, dict)}, flush=True)


if __name__ == "__main__":
    main()
