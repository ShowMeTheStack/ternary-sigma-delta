"""
Language-model trainer with straight-through (STE) quantized linear layers.

python train.py --size S --alphabet ternary --gamma 1.0 --dz 0.5 --tokens 0.6e9 \
    --lr 1.2e-3 --seed 0 --data ~/data/fineweb_edu_gpt2 --out runs/S_ter_g1.0_s0

Runs with the same seed draw the same sequence of training batches, so runs that differ only in
the quantizer settings see identical data.

Writes into --out: args.json, ckpt.pt (periodic checkpoint, deleted at the end), log.jsonl (per-step
records and validation losses), hist.jsonl (latent histograms), final.json (snapshot / mode / dither-free /
ema-requant / latent val losses, per-layer stats) and final_state.pt (latent weights fp16, counts int16,
flip counts int16 for quant layers).
"""
import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.llama_small import Llama, SIZES
from quant.ste import QuantLinear, quant_modules


# ---------------------------------------------------------------- data
class Shards:
    def __init__(self, path, split):
        pat = "val.bin" if split == "val" else "train_*.bin"
        self.files = sorted(glob.glob(os.path.join(os.path.expanduser(path), pat)))
        assert self.files, f"no shards at {path}/{pat}"
        self.mm = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        self.sizes = np.array([len(m) for m in self.mm], dtype=np.int64)
        self.total = int(self.sizes.sum())

    def batch(self, rng, bs, seq, device):
        # seeded random windows: identical sequence for every run with the same seed
        p = self.sizes / self.sizes.sum()
        si = rng.choice(len(self.mm), size=bs, p=p)
        x = np.empty((bs, seq + 1), dtype=np.int64)
        for i, s in enumerate(si):
            off = rng.integers(0, self.sizes[s] - seq - 1)
            x[i] = self.mm[s][off:off + seq + 1].astype(np.int64)
        t = torch.from_numpy(x).to(device, non_blocking=True)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def sequential(self, bs, seq, n_batches):
        m = self.mm[0]; pos = 0
        for _ in range(n_batches):
            x = np.stack([m[pos + i * (seq + 1): pos + (i + 1) * (seq + 1)].astype(np.int64) for i in range(bs)])
            pos += bs * (seq + 1)
            t = torch.from_numpy(x)
            yield t[:, :-1].contiguous(), t[:, 1:].contiguous()


# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, val, bs, seq, n_batches, device):
    model.eval(); tot = 0.0; n = 0
    for x, y in val.sequential(bs, seq, n_batches):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            loss = model(x, y)
        tot += float(loss); n += 1
    model.train()
    return tot / n


def latent_hist(m, bins=64, lim=4.0):
    w = m.weight.detach().float()
    r = (w / w.abs().mean().clamp_min(1e-8)).clamp(-lim, lim)
    h = torch.histc(r, bins=bins, min=-lim, max=lim)
    return (h / h.sum()).tolist()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="S", choices=list(SIZES))
    ap.add_argument("--alphabet", default="ternary")           # none|binary|ternary|uniform_k|levels:a,b,...
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--dz", type=float, default=0.5)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--act_bits", type=int, default=None)
    ap.add_argument("--tokens", type=float, default=0.6e9)
    ap.add_argument("--batch_tokens", type=int, default=524288)
    ap.add_argument("--micro_bs", type=int, default=32)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.02)
    ap.add_argument("--min_warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--wd_off_frac", type=float, default=0.0, help="disable weight decay for the last fraction of steps (BitNet recipe)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="~/data/fineweb_edu_gpt2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--hist_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_batches", type=int, default=40)
    ap.add_argument("--final_eval_batches", type=int, default=400)
    ap.add_argument("--final_window", type=int, default=64)
    ap.add_argument("--cooldown", type=int, default=0, help="extra steps after the schedule at the final LR with the live optimizer; duty-cycle stats and deployment evals use this window (production recipe)")
    ap.add_argument("--arch", default="llama", choices=["llama", "gpt2"])
    ap.add_argument("--ckpt_every", type=int, default=100)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)    # debugging
    # alternative scale rules, oscillation dampening and iterative freezing; all off by default
    ap.add_argument("--scale_rule", default="absmean", choices=["absmean", "lsq", "twn"])
    ap.add_argument("--osc_damp", type=float, default=0.0, help="Nagel et al. 2022 dampening weight lambda on sum (w-q)^2")
    ap.add_argument("--damp_start_frac", type=float, default=0.5, help="lambda ramps linearly from 0 at this fraction of the schedule to lambda at its end")
    ap.add_argument("--freeze_fth", type=float, default=0.0, help="Nagel et al. 2022 iterative freezing: EMA flip-frequency threshold (0 = off)")
    ap.add_argument("--freeze_start_frac", type=float, default=0.2)
    ap.add_argument("--freeze_momentum", type=float, default=0.99)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lean", action="store_true", help="memory-lean cool-down for large models: no latent EMA, no |dw| telemetry, uncompiled eval with 1-row loss chunks")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    device = a.device
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    qkw = dict(alphabet=a.alphabet, scale_mult=a.gamma, deadzone_frac=a.dz, stochastic=a.stochastic, act_bits=a.act_bits,
               scale_rule=a.scale_rule)
    model = Llama(seq=a.seq, quant_kwargs=qkw, arch=a.arch, **SIZES[a.size]).to(device)
    n_params = model.n_params()
    qmods = quant_modules(model)
    cmods = [(n, m) for n, m in qmods if m.half == 1 or m.level_table is not None]   # counting for 2/3-level and custom-level quantizers
    print(f"params(non-emb)={n_params/1e6:.1f}M quant layers={len(qmods)} counted={len(cmods)}", flush=True)

    sched_steps = a.max_steps or int(a.tokens // a.batch_tokens)
    steps = sched_steps + a.cooldown
    accum = a.batch_tokens // (a.micro_bs * a.seq)
    warm = max(a.min_warmup, int(a.warmup_frac * sched_steps))
    def lr_at(t):
        if t < warm: return a.lr * (t + 1) / warm
        if t >= sched_steps: return a.lr * a.min_lr_frac          # cool-down: constant final LR
        pr = (t - warm) / max(1, sched_steps - warm)
        return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * pr)))

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=(device == "cuda"))
    if a.scale_rule == "lsq":
        with torch.no_grad():
            for _, m in qmods: m.scale(m.weight)      # initialise the learned scales before compiling
    cmodel = torch.compile(model) if a.compile else model

    train, val = Shards(a.data, "train"), Shards(a.data, "val")
    rng = np.random.default_rng(a.seed)
    print(f"steps={steps} accum={accum} train_tokens={train.total/1e9:.2f}B", flush=True)

    # resume
    step0 = 0
    ck = os.path.join(a.out, "ckpt.pt")
    if os.path.exists(ck):
        sd = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"]); step0 = sd["step"]
        rng = np.random.default_rng(); rng.bit_generator.state = sd["rng"]
        if sd.get("freeze"):
            for n, m in cmods:
                if n in sd["freeze"]:
                    fz = sd["freeze"][n]; m.start_freezing(a.freeze_fth, a.freeze_momentum)
                    m.frz_mask.copy_(fz["mask"]); m.frz_level.copy_(fz["level"]); m.osc_f.copy_(fz["osc"]); m.lv_ema.copy_(fz["lv"])
            print(f"restored freezing state for {len(sd['freeze'])} layers", flush=True)
        print(f"resumed at step {step0}", flush=True)

    ema = None; final = {}
    logf = open(os.path.join(a.out, "log.jsonl"), "a"); histf = open(os.path.join(a.out, "hist.jsonl"), "a")
    t0 = time.time(); tok_seen = step0 * a.batch_tokens
    for step in range(step0, steps):
        if step == sched_steps and a.cooldown > 0:
            if a.lean: torch.cuda.empty_cache(); model.loss_chunk = 1
            final["val_snapshot_end_of_schedule"] = evaluate(model if a.lean else cmodel, val, a.micro_bs, a.seq, a.final_eval_batches, device)
            logf.write(json.dumps(dict(step=step, val_snapshot_end_of_schedule=final["val_snapshot_end_of_schedule"])) + "\n"); logf.flush()
        window = a.cooldown if a.cooldown > 0 else a.final_window
        if step == max(0, steps - window) and cmods:
            for _, m in cmods:
                m.start_counting(); m.record_traj = True
            ema = None if a.lean else {n: m.weight.detach().float().clone() for n, m in cmods}
            ema_beta = 1 - 2.0 / window
            if a.lean:
                for _, m in cmods: m.w_prev = None
                torch.cuda.empty_cache()
        lr = lr_at(step)
        if a.freeze_fth > 0 and step == int(a.freeze_start_frac * sched_steps):
            for _, m in cmods:
                if m.frz_mask is None: m.start_freezing(a.freeze_fth, a.freeze_momentum)
        t0d = int(a.damp_start_frac * sched_steps)
        lam = a.osc_damp * min(1.0, max(0.0, (step - t0d) / max(1, sched_steps - t0d))) if a.osc_damp > 0 else 0.0
        for g in opt.param_groups:
            g["lr"] = lr
            if a.wd_off_frac > 0 and step >= int((1 - a.wd_off_frac) * steps) and g["weight_decay"] > 0:
                g["weight_decay"] = 0.0
        log_now = step % a.log_every == 0 or step == steps - 1
        loss_acc = 0.0
        for _, m in qmods: m.tick = True
        for mi in range(accum):
            x, y = train.batch(rng, a.micro_bs, a.seq, device)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
                loss = cmodel(x, y) / accum
            loss.backward()
            if log_now: loss_acc += float(loss)
        if lam > 0:
            for _, m in cmods: m.damp_grad(lam)
        if a.freeze_fth > 0:
            for _, m in cmods:
                if m.frz_mask is not None and m.weight.grad is not None: m.weight.grad.masked_fill_(m.frz_mask, 0.0)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if a.freeze_fth > 0:
            for _, m in cmods: m.pin_frozen()
        tok_seen += a.batch_tokens
        if ema is not None:
            with torch.no_grad():
                for n, m in cmods:
                    ema[n].mul_(ema_beta).add_(m.weight.float(), alpha=1 - ema_beta)

        if log_now:
            rec = dict(step=step, tokens=tok_seen, lr=lr, loss=loss_acc,
                       gnorm=float(gn), tps=tok_seen / (time.time() - t0 + 1e-9) if step0 == 0 else None,
                       elapsed=time.time() - t0)
            if qmods:
                rec["flip"] = {n: m.pop_flip_rate() for n, m in qmods}
                rec["zero"] = {n: float(m.last_zero_frac) for n, m in qmods}
                rec["scale"] = {n: float(m.last_scale) for n, m in qmods}
                rec["flip_mean"] = float(np.mean(list(rec["flip"].values())))
                rec["zero_mean"] = float(np.mean(list(rec["zero"].values())))
                if a.scale_rule == "lsq":
                    rec["scale_ratio"] = {n: float(m.last_scale / m.weight.detach().abs().mean()) for n, m in qmods}
                if a.freeze_fth > 0:
                    rec["frozen_mean"] = float(np.mean([m.frozen_frac() for _, m in cmods]))
                if lam > 0: rec["lam"] = lam
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if step % (a.log_every * 10) == 0:
                print(f"step {step}/{steps} loss {rec['loss']:.4f} lr {lr:.2e} flip {rec.get('flip_mean', 0):.4f} "
                      f"zero {rec.get('zero_mean', 0):.3f} {tok_seen/(time.time()-t0)/1e3:.1f}k tok/s", flush=True)
        if qmods and step % a.hist_every == 0:
            histf.write(json.dumps(dict(step=step, hist={n: latent_hist(m) for n, m in qmods})) + "\n"); histf.flush()
        if step % a.eval_every == 0 and step > 0:
            vl = evaluate(cmodel, val, a.micro_bs, a.seq, a.eval_batches, device)
            logf.write(json.dumps(dict(step=step, val_loss=vl)) + "\n"); logf.flush()
            print(f"  val {vl:.4f}", flush=True)
        if step % a.ckpt_every == 0 and step > 0 and step < steps - window:
            fz = {n: dict(mask=m.frz_mask, level=m.frz_level, osc=m.osc_f, lv=m.lv_ema) for n, m in cmods if m.frz_mask is not None}
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1, rng=rng.bit_generator.state, freeze=fz), ck + ".tmp")
            os.replace(ck + ".tmp", ck)   # atomic: an interruption mid-write leaves the previous checkpoint intact

    # ------------------------------------------------------------ final evaluations
    final.update(n_params=n_params, steps=steps, sched_steps=sched_steps, cooldown=a.cooldown, tokens=tok_seen, elapsed=time.time() - t0)
    eb = a.final_eval_batches
    if a.lean: torch.cuda.empty_cache(); model.loss_chunk = 1
    final["val_snapshot"] = evaluate(model if a.lean else cmodel, val, a.micro_bs, a.seq, eb, device)
    if cmods:
        qmods = cmods
        per_layer = {}
        modes, bars, emas = {}, {}, {}
        with torch.no_grad():
            for n, m in qmods:
                mode, qbar = m.duty_cycle_tensors()
                modes[n], bars[n] = mode, qbar
                wq_placeholder = None
                emas[n] = m.quantize(ema[n])[0] if ema is not None else wq_placeholder
                wq = m.quantize(m.weight)[0]
                c = m.counts.float(); W = m.count_steps
                duty = c / W
                per_layer[n] = dict(
                    scale=float(m.scale(m.weight)), zero_frac=float((wq == 0).float().mean()),
                    zero_frac_from_duty=float(duty[m.half].mean()),
                    flip_rate_window=float(m.flipc.float().mean() / max(1, W - 1)),
                    frac_two_level=float(((c > 0).sum(0) == 2).float().mean()),
                    frac_three_level=float(((c > 0).sum(0) == 3).float().mean()),
                    frac_static=float(((c > 0).sum(0) == 1).float().mean()),
                    mode_eq_snapshot=float((mode == wq).float().mean()),
                    ema_eq_mode=(float((torch.sign(emas[n]) == torch.sign(mode)).float().mean()) if emas[n] is not None else None),
                    dither_var_proxy=float(((wq - qbar) ** 2).mean()),   # (q_snap - qbar)^2 ~ (w*-a)(b-w*) on average
                    mean_abs_step=m.mean_abs_step(), scale_ratio=float(m.scale(m.weight).detach() / m.weight.detach().abs().mean()),
                    frozen_frac=m.frozen_frac(),
                )
        def with_override(src):
            for n, m in qmods: m.override = src[n]
            v = evaluate(model, val, a.micro_bs, a.seq, eb, device)   # uncompiled model to avoid recompiles
            for n, m in qmods: m.override = None
            return v
        final["val_mode"] = with_override(modes)
        final["val_dither_free"] = with_override(bars)
        final["val_ema_requant"] = with_override(emas) if ema is not None else None
        final["val_latent_fp"] = with_override({n: m.weight.detach() for n, m in qmods})
        final["per_layer"] = per_layer
        final["mean_zero_frac"] = float(np.mean([v["zero_frac"] for v in per_layer.values()]))
        final["mean_flip_rate_window"] = float(np.mean([v["flip_rate_window"] for v in per_layer.values()]))
        final["mean_frozen_frac"] = float(np.mean([v["frozen_frac"] for v in per_layer.values()]))
        qnames = {n + ".weight" for n, _ in qmods}
        state = {n: dict(latent=m.weight.detach().half().cpu(), counts=m.counts.cpu(), flipc=m.flipc.cpu(),
                         traj=torch.stack(m.traj) if m.traj else None, traj_idx=m.traj_idx,
                         ema=(ema[n].half().cpu() if ema is not None else None))
                 for n, m in qmods}
        state["_other"] = {n: p.detach().half().cpu() for n, p in model.named_parameters() if n not in qnames}
        torch.save(state, os.path.join(a.out, "final_state.pt"))
    else:
        torch.save({n: p.detach().half().cpu() for n, p in model.named_parameters()}, os.path.join(a.out, "final_state.pt"))
    json.dump(final, open(os.path.join(a.out, "final.json"), "w"), indent=1)
    print("FINAL", {k: v for k, v in final.items() if not isinstance(v, dict)}, flush=True)
    if os.path.exists(ck): os.remove(ck)


if __name__ == "__main__":
    main()
