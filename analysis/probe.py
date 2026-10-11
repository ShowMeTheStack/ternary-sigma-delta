"""Stationary cool-down probe for duty-cycle deployment rules.

Reloads a finished run's final weights, trains K more steps at a constant (final) learning rate with
level counting on for every step, then evaluates: pre-probe snapshot, post-probe snapshot, mode-rounded,
dither-free (time-averaged), EMA-requantized. Writes probe_K.json into the run directory.

python analysis/probe.py runs/S_ter_g1.0_s0 --steps 64
"""
import argparse, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.llama_small import Llama, SIZES
from quant.ste import quant_modules
from train import Shards, evaluate

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="default: run lr * min_lr_frac (the final LR)")
    ap.add_argument("--eval_batches", type=int, default=400); ap.add_argument("--data", default=None)
    ap.add_argument("--save_bars", action="store_true", help="save time-averaged (dither-free) weights to bars_K.pt")
    ap.add_argument("--ckpt_every", type=int, default=100, help="checkpoint the probe state every N steps (resumable after a power cut); 0 = off")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    args = json.load(open(os.path.join(a.run, "args.json"))); device = a.device
    torch.manual_seed(args["seed"] + 1000)
    qkw = dict(alphabet=args["alphabet"], scale_mult=args["gamma"], deadzone_frac=args["dz"], stochastic=args.get("stochastic", False), act_bits=args.get("act_bits"))
    model = Llama(seq=args["seq"], quant_kwargs=qkw, arch=args.get("arch", "llama"), **SIZES[args["size"]]).to(device)
    st = torch.load(os.path.join(a.run, "final_state.pt"), map_location=device, weights_only=False)
    sd = model.state_dict()
    for n, p in st.pop("_other").items(): sd[n].copy_(p.float())
    qm = dict(quant_modules(model))
    for n, d in st.items(): qm[n].weight.data.copy_(d["latent"].float().to(device))
    cm = {n: m for n, m in qm.items() if m.half == 1}
    lr = a.lr if a.lr is not None else args["lr"] * args["min_lr_frac"]
    wd = 0.0 if args.get("wd_off_frac", 0) > 0 else args["wd"]
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]; nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=(device == "cuda"))
    data = a.data or args["data"]; train, val = Shards(data, "train"), Shards(data, "val")
    rng = np.random.default_rng(args["seed"] + 1000)
    accum = args["batch_tokens"] // (args["micro_bs"] * args["seq"])
    out = dict(run=a.run, steps=a.steps, lr=lr, wd=wd)
    ck = os.path.join(a.run, f"probe_{a.steps}_ckpt.pt"); step0 = 0; t_prev = 0.0
    if os.path.exists(ck):
        sd_ck = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(sd_ck["model"]); opt.load_state_dict(sd_ck["opt"]); rng.bit_generator.state = sd_ck["rng"]
        out["val_pre_snapshot"] = sd_ck["val_pre_snapshot"]; step0 = sd_ck["step"]; t_prev = sd_ck["elapsed"]
        for n, m in cm.items():
            m.start_counting(); m.record_traj = False; c = sd_ck["layers"][n]
            m.counts.copy_(c["counts"]); m.flipc.copy_(c["flipc"]); m.q_prev.copy_(c["q_prev"]); m.count_steps = c["count_steps"]
        ema = {n: sd_ck["layers"][n]["ema"].float().to(device) for n in cm}
        print(f"probe resumed at step {step0}", flush=True)
    else:
        out["val_pre_snapshot"] = evaluate(model, val, args["micro_bs"], args["seq"], a.eval_batches, device)
        for m in cm.values(): m.start_counting(); m.record_traj = False
        ema = {n: m.weight.detach().float().clone() for n, m in cm.items()}
    eb = 1 - 2.0 / a.steps
    model.train(); t0 = time.time() - t_prev
    for step in range(step0, a.steps):
        for m in cm.values(): m.tick = True
        for _ in range(accum):
            x, y = train.batch(rng, args["micro_bs"], args["seq"], device)
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = model(x, y) / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n, m in cm.items(): ema[n].mul_(eb).add_(m.weight.float(), alpha=1 - eb)
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            layers = {n: dict(counts=m.counts, flipc=m.flipc, q_prev=m.q_prev, count_steps=m.count_steps, ema=ema[n].half()) for n, m in cm.items()}
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), rng=rng.bit_generator.state, step=step + 1, elapsed=time.time() - t0,
                            val_pre_snapshot=out["val_pre_snapshot"], layers=layers), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    out["probe_seconds"] = time.time() - t0
    out["val_post_snapshot"] = evaluate(model, val, args["micro_bs"], args["seq"], a.eval_batches, device)
    modes, bars, emas, per = {}, {}, {}, {}
    with torch.no_grad():
        for n, m in cm.items():
            mode, qbar = m.duty_cycle_tensors(); modes[n], bars[n] = mode, qbar
            emas[n] = m.quantize(ema[n])[0]; wq = m.quantize(m.weight)[0]
            c = m.counts.float(); W = m.count_steps; pm, pp = c[0] / W, c[2] / W; qb = pp - pm
            per[n] = dict(zero_frac=float((wq == 0).float().mean()), frac_static=float(((c > 0).sum(0) == 1).float().mean()),
                          frac_two=float(((c > 0).sum(0) == 2).float().mean()), frac_three=float(((c > 0).sum(0) == 3).float().mean()),
                          flip_rate=float(m.flipc.float().mean() / max(1, W - 1)), var_q_over_s2=float(((pp + pm) - qb ** 2).mean()),
                          mode_eq_snapshot=float((torch.round(mode / m.scale(m.weight)) == torch.round(wq / m.scale(m.weight))).float().mean()),
                          ema_eq_mode=float((m.levels(ema[n])[0] == torch.round(mode / m.scale(m.weight))).float().mean()))
    def with_override(src):
        for n, m in cm.items(): m.override = src[n]
        v = evaluate(model, val, args["micro_bs"], args["seq"], a.eval_batches, device)
        for m in cm.values(): m.override = None
        return v
    out["val_mode"] = with_override(modes); out["val_dither_free"] = with_override(bars); out["val_ema_requant"] = with_override(emas)
    out["per_layer"] = per
    for k in ["zero_frac", "frac_static", "frac_two", "frac_three", "flip_rate", "var_q_over_s2", "mode_eq_snapshot", "ema_eq_mode"]:
        out["mean_" + k] = float(np.mean([v[k] for v in per.values()]))
    if a.save_bars:
        torch.save({n: b.half().cpu() for n, b in bars.items()}, os.path.join(a.run, f"bars_{a.steps}.pt"))
    json.dump(out, open(os.path.join(a.run, f"probe_{a.steps}.json"), "w"), indent=1)
    if os.path.exists(ck): os.remove(ck)
    print("PROBE", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if not isinstance(v, dict)}, flush=True)

if __name__ == "__main__":
    main()
