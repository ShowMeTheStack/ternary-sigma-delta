"""
Straight-through (STE) quantized layers with duty-cycle telemetry.

alphabet: "none" | "binary" | "ternary" | "uniform_k" (k levels symmetric, k odd)
          | "levels:a,b,..." symmetric custom levels +-a*s, +-b*s, ... (no zero level; nearest-level rounding; s = gamma*mean|W|)
scale_rule:
  "absmean"  s = gamma * mean|W| (per tensor, detached)                      [default; BitNet b1.58 at gamma=1, dz=0.5]
  "lsq"      s learned per tensor (Esser et al. 2020), init gamma * mean|W|, gradient scaled by 1/sqrt(N*Q_P)
  "twn"      threshold 0.7 * mean|W|, level = mean of |W| above the threshold (Li et al. 2016); ternary only
deadzone_frac dz:   ternary zero iff |W| <= dz * s   (absmean/lsq rules)
stochastic:         stochastic rounding instead of deterministic rounding
act_bits:           None (bf16 activations) or 8 (per-token absmax int8, STE)

Oscillation baselines (Nagel et al. 2022), both off by default:
  damp_grad(lam)        oscillation dampening: adds the gradient of lam * sum (w - q)^2 (q = current grid point) to w.grad
  freeze_fth > 0        iterative freezing: per-weight EMA flip frequency; weights above f_th are frozen at the level
                        nearest the EMA of their level index (mask applied in levels(); latent pinned by pin_frozen())

Telemetry (all cheap, on GPU):
  flips           running count of weights whose level changed since the previous step
  zero_frac       fraction of zeros in the current quantized weight
  counts          per-weight level visit counts over the counting window (int16, (2*half+1) x numel)
  flipc           per-weight flip counts over the counting window (int16)
  step_abs        running sum of mean |w_t - w_{t-1}| over the counting window (input to a level-crossing estimate)
  traj            q for a fixed subset of weights every step (int8, CPU)
  override        if set, forward uses this tensor instead of the live quantizer
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LSQ(torch.autograd.Function):
    """q = lv * s with STE on w (gradient 1 everywhere, as in the absmean STE) and the LSQ gradient on s."""
    @staticmethod
    def forward(ctx, w, s, lv, inside, gscale):
        ctx.save_for_backward(w, s, lv, inside); ctx.gscale = gscale
        return lv * s

    @staticmethod
    def backward(ctx, g):
        w, s, lv, inside = ctx.saved_tensors
        gs = (g * (lv - inside * (w / s))).sum() * ctx.gscale
        return g, gs.reshape(s.shape), None, None, None


class QuantBase(nn.Module):
    """Quantizer + telemetry over self.weight (any shape). Subclasses define forward()."""
    def _init_quant(self, alphabet="ternary", scale_mult=1.0, deadzone_frac=0.5, stochastic=False,
                    act_bits=None, traj_n=1024, name="", scale_rule="absmean"):
        shape = self.weight.shape
        self.alphabet, self.gamma, self.dz = alphabet, scale_mult, deadzone_frac
        self.stochastic, self.act_bits, self.name, self.scale_rule = stochastic, act_bits, name, scale_rule
        self.telemetry = alphabet != "none"
        self.half = (int(alphabet.split("_")[1]) - 1) // 2 if alphabet.startswith("uniform_") else 1   # level index offset
        if alphabet.startswith("levels:"):
            tab = sorted(float(x) for x in alphabet[len("levels:"):].split(","))
            self.half = len(tab)
            self.register_buffer("level_table", torch.tensor(tab, dtype=torch.float32), persistent=False)
            mids = [(tab[i] + tab[i + 1]) / 2 for i in range(len(tab) - 1)]
            self.register_buffer("level_mids", torch.tensor(mids, dtype=torch.float32), persistent=False)
        else:
            self.level_table = None
        self.register_buffer("q_prev", torch.zeros(shape, dtype=torch.int8), persistent=False)
        self.register_buffer("flips", torch.zeros((), dtype=torch.float64), persistent=False)
        self.register_buffer("flip_steps", torch.zeros((), dtype=torch.float64), persistent=False)
        self.counts = None            # allocated when the final window starts
        self.count_steps = 0
        g = torch.Generator().manual_seed(1234)
        self.traj_idx = torch.randperm(self.weight.numel(), generator=g)[:traj_n]
        self.traj = []                # list of int8 CPU tensors
        self.record_traj = False
        self.override = None          # tensor of weight shape or None
        self.tick = False             # set True by the trainer once per optimizer step
        self.register_buffer("last_scale", torch.zeros(()), persistent=False)
        self.register_buffer("last_zero_frac", torch.zeros(()), persistent=False)
        self.w_prev = None            # latent copy for step_abs (allocated with counts)
        self.register_buffer("step_abs", torch.zeros((), dtype=torch.float64), persistent=False)
        if scale_rule == "lsq":
            self.s_param = nn.Parameter(torch.zeros(()))
            self._lsq_init = False        # set on the first scale() call (eager, before compile); a resumed checkpoint overrides the value
            self.lsq_gscale = 1.0 / math.sqrt(self.weight.numel() * max(1, self.half))
        elif scale_rule == "twn":
            assert alphabet == "ternary", "TWN rule is ternary-only"
        # iterative freezing (off unless freeze_fth > 0)
        self.freeze_fth = 0.0
        self.freeze_m = 0.99
        self.frz_mask = None; self.frz_level = None; self.osc_f = None; self.lv_ema = None

    # ---- scale rules ------------------------------------------------------
    def scale(self, w):
        if self.scale_rule == "lsq":
            if not self._lsq_init:
                with torch.no_grad():
                    self.s_param.fill_(float(self.gamma * w.abs().mean()))
                self._lsq_init = True
            return self.s_param.abs().clamp_min(1e-8)
        if self.scale_rule == "twn":
            m = w.abs().mean()
            thr = 0.7 * m
            big = w.abs() > thr
            return (w.abs() * big).sum().div(big.sum().clamp_min(1)).clamp_min(1e-8).detach()
        return (self.gamma * w.abs().mean()).clamp_min(1e-8)

    def threshold(self, w, s):
        """Ternary zero iff |w| <= threshold."""
        if self.scale_rule == "twn":
            return 0.7 * w.abs().mean().detach()
        return self.dz * s.detach()

    def levels(self, w):
        """Return integer level index in {-1,0,1} (or {-k..k}) and scale s (s carries a gradient only under lsq)."""
        s = self.scale(w)
        sd = s.detach()
        if self.alphabet == "binary":
            lv = torch.sign(w)
            lv = torch.where(lv == 0, torch.ones_like(lv), lv)
        elif self.alphabet == "ternary":
            if self.stochastic:
                p = (w.abs() / sd).clamp(0, 1)
                nz = (torch.rand_like(p) < p).to(w.dtype)
            else:
                nz = (w.abs() > self.threshold(w, s)).to(w.dtype)
            lv = torch.sign(w) * nz
        elif self.alphabet.startswith("uniform_"):
            k = int(self.alphabet.split("_")[1]); half = (k - 1) // 2
            s = (self.gamma * w.abs().amax() / half).clamp_min(1e-8).detach(); sd = s
            lv = torch.clamp(torch.round(w / s), -half, half)
        elif self.alphabet.startswith("levels:"):
            # nearest custom level: index i in {1..half} by counting mid-points below |w|/s; sign from w (zero -> +)
            u = (w.abs() / sd).unsqueeze(-1)
            idx = (u > self.level_mids).sum(-1).to(w.dtype) + 1.0
            sg = torch.sign(w); sg = torch.where(sg == 0, torch.ones_like(sg), sg)
            lv = sg * idx
        else:
            raise ValueError(self.alphabet)
        if self.frz_mask is not None:
            lv = torch.where(self.frz_mask, self.frz_level.to(lv.dtype), lv)
        return lv, (s if self.scale_rule == "lsq" else sd)

    def lv_to_w(self, lv, s):
        """Level index -> weight value. Uniform/binary/ternary: lv*s; custom levels: sign(lv)*table[|lv|-1]*s."""
        if self.level_table is None: return lv * s
        mag = self.level_table[(lv.abs().long() - 1).clamp_min(0)]
        return torch.sign(lv) * mag * s

    def quantize(self, w):
        lv, s = self.levels(w)
        return self.lv_to_w(lv.detach(), s.detach()), lv.detach(), s.detach()

    def ste_weight(self):
        """Forward weight with the straight-through gradient (and the LSQ scale gradient when enabled)."""
        w = self.weight
        lv, s = self.levels(w)
        lv = lv.detach()
        if self.scale_rule == "lsq":
            inside = (w.abs() <= s.detach()).to(w.dtype)
            wq = _LSQ.apply(w, s, lv, inside, self.lsq_gscale)
        else:
            wq = w + (self.lv_to_w(lv, s) - w).detach()
        return wq, lv, s.detach()

    # ---- oscillation baselines -------------------------------------------
    @torch.no_grad()
    def damp_grad(self, lam):
        """Oscillation dampening: gradient of lam * sum_i (w_i - q_i)^2 with q_i the current grid point (detached)."""
        if lam <= 0 or self.weight.grad is None: return
        wq, _, _ = self.quantize(self.weight)
        self.weight.grad.add_(self.weight - wq, alpha=2.0 * lam)

    def start_freezing(self, fth, momentum=0.99):
        n = self.weight.shape; dev = self.weight.device
        self.freeze_fth, self.freeze_m = fth, momentum
        self.frz_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self.frz_level = torch.zeros(n, dtype=torch.int8, device=dev)
        self.osc_f = torch.zeros(n, dtype=torch.float32, device=dev)
        self.lv_ema = self.q_prev.float().clone()

    @torch.no_grad()
    def pin_frozen(self):
        """After the optimizer step: zero the latent drift of frozen weights (their level is fixed by the mask)."""
        if self.frz_mask is None: return
        s = self.scale(self.weight).detach()
        self.weight.masked_scatter_(self.frz_mask, (self.frz_level.to(self.weight.dtype) * s)[self.frz_mask])

    def frozen_frac(self):
        return float(self.frz_mask.float().mean()) if self.frz_mask is not None else 0.0

    # ---- telemetry --------------------------------------------------------
    def start_counting(self):
        n = self.weight.numel()
        self.counts = torch.zeros(2 * self.half + 1, n, dtype=torch.int16, device=self.weight.device)
        self.flipc = torch.zeros(n, dtype=torch.int16, device=self.weight.device)
        self.count_steps = 0
        self.w_prev = self.weight.detach().float().clone(); self.step_abs.zero_()

    @torch.compiler.disable
    @torch.no_grad()
    def _telemetry(self, lv, s):
        li = lv.to(torch.int8)
        changed = li != self.q_prev
        self.flips += changed.sum().double()
        self.flip_steps += 1
        if self.counts is not None and self.count_steps > 0:
            self.flipc += changed.reshape(-1).to(torch.int16)
        if self.frz_mask is not None:
            m = self.freeze_m
            self.osc_f.mul_(m).add_(changed.float(), alpha=1 - m)
            self.lv_ema.mul_(m).add_(li.float(), alpha=1 - m)
            new = (self.osc_f > self.freeze_fth) & ~self.frz_mask
            self.frz_level = torch.where(new, self.lv_ema.round().clamp(-self.half, self.half).to(torch.int8), self.frz_level)
            self.frz_mask |= new
        self.q_prev.copy_(li)
        self.last_scale.copy_(s)
        self.last_zero_frac.copy_((li == 0).float().mean())
        if self.counts is not None:
            idx = (li.reshape(-1).long() + self.half).unsqueeze(0)
            self.counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.int16))
            self.count_steps += 1
            if self.w_prev is not None:
                wf = self.weight.detach().float()
                self.step_abs += (wf - self.w_prev).abs().mean().double(); self.w_prev.copy_(wf)
        if self.record_traj:
            self.traj.append(li.reshape(-1)[self.traj_idx.to(li.device)].cpu())

    def pop_flip_rate(self):
        r = float(self.flips / (self.flip_steps.clamp_min(1) * self.weight.numel()))
        self.flips.zero_(); self.flip_steps.zero_()
        return r

    def mean_abs_step(self):
        return float(self.step_abs) / max(1, self.count_steps - 1)

    @torch.no_grad()
    def duty_cycle_tensors(self):
        """From counts: q_mode (most-visited level), q_bar (time average), both in weight units."""
        assert self.counts is not None and self.count_steps > 0
        s = self.scale(self.weight).detach()
        c = self.counts.float()
        levels = torch.arange(-self.half, self.half + 1, device=c.device, dtype=c.dtype)
        vals = self.lv_to_w(levels, s)                       # value of each level row
        mode = vals[c.argmax(0)]
        qbar = (vals[:, None] * c).sum(0) / self.count_steps
        return mode.view_as(self.weight), qbar.view_as(self.weight)

    def _quant_forward_weight(self, dtype):
        if self.override is not None:
            return self.override.to(dtype)
        if self.alphabet == "none":
            return self.weight
        wq, lv, s = self.ste_weight()
        if self.training and self.telemetry and self.tick:
            self._telemetry(lv, s)
            self.tick = False
        return wq.to(dtype)

    def _act_quant(self, x):
        if self.act_bits == 8:
            sx = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / 127.0
            xq = torch.round(x / sx).clamp(-128, 127) * sx
            x = x + (xq - x).detach()
        return x


class QuantLinear(QuantBase):
    def __init__(self, in_f, out_f, alphabet="ternary", scale_mult=1.0, deadzone_frac=0.5,
                 stochastic=False, act_bits=None, traj_n=1024, name="", scale_rule="absmean"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.weight, std=0.02)
        self._init_quant(alphabet, scale_mult, deadzone_frac, stochastic, act_bits, traj_n, name, scale_rule)

    def forward(self, x):
        w = self._quant_forward_weight(x.dtype)
        x = self._act_quant(x)
        return F.linear(x, w)


class QuantConv2d(QuantBase):
    def __init__(self, in_c, out_c, k, stride=1, padding=0, bias=False, alphabet="ternary", scale_mult=1.0,
                 deadzone_frac=0.5, stochastic=False, act_bits=None, traj_n=1024, name="", scale_rule="absmean"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_c, in_c, k, k))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_c)) if bias else None
        self.stride, self.padding = stride, padding
        self._init_quant(alphabet, scale_mult, deadzone_frac, stochastic, act_bits, traj_n, name, scale_rule)

    def forward(self, x):
        w = self._quant_forward_weight(x.dtype)
        return F.conv2d(x, w, self.bias, self.stride, self.padding)


def quant_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, QuantBase) and m.alphabet != "none"]
