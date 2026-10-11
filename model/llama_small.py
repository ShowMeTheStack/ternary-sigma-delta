"""Small Llama-style decoder with pluggable QuantLinear. RMSNorm, SwiGLU, RoPE, SDPA."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from quant.ste import QuantLinear


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w


def rope_cache(T, hd, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(T, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):  # x: B,H,T,hd
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, nh, qkw, li, arch="llama"):
        super().__init__()
        self.nh, self.hd, self.arch = nh, d // nh, arch
        mk = lambda i, o, nm: QuantLinear(i, o, name=f"blk{li}.{nm}", **qkw)
        Norm = RMSNorm if arch == "llama" else (lambda dd: nn.LayerNorm(dd))
        self.n1, self.n2 = Norm(d), Norm(d)
        self.wq, self.wk, self.wv, self.wo = mk(d, d, "q"), mk(d, d, "k"), mk(d, d, "v"), mk(d, d, "o")
        if arch == "llama":
            hidden = int(8 * d / 3 / 64 + 0.5) * 64
            self.w1, self.w3, self.w2 = mk(d, hidden, "w1"), mk(d, hidden, "w3"), mk(hidden, d, "w2")
        else:                                   # GPT-2 style: LayerNorm, GELU, 4d MLP, learned positions (no RoPE)
            self.w1, self.w2 = mk(d, 4 * d, "w1"), mk(4 * d, d, "w2"); self.w3 = None
        with torch.no_grad():
            for m in (self.wo, self.w2):
                m.weight.mul_(1.0 / math.sqrt(2 * (li + 1)))

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        h = self.n1(x)
        q = self.wq(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.wv(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        if self.arch == "llama":
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(a.transpose(1, 2).reshape(B, T, D))
        h = self.n2(x)
        if self.arch == "llama":
            return x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x + self.w2(F.gelu(self.w1(h)))


class Llama(nn.Module):
    def __init__(self, vocab=50304, d=768, n_layer=12, n_head=12, seq=1024, quant_kwargs=None, arch="llama"):
        super().__init__()
        qkw = quant_kwargs or {}
        self.seq, self.arch = seq, arch
        self.loss_chunk = 8
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pos = nn.Embedding(seq, d) if arch == "gpt2" else None
        if self.pos is not None: nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, n_head, qkw, i, arch) for i in range(n_layer)])
        self.nf = RMSNorm(d) if arch == "llama" else nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)          # kept high precision, untied
        nn.init.normal_(self.head.weight, std=0.02)
        cos, sin = rope_cache(seq, d // n_head, "cpu")
        self.register_buffer("cos", cos, persistent=False); self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.emb(idx)
        if self.pos is not None: x = x + self.pos(torch.arange(T, device=idx.device))
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            x = b(x, cos, sin)
        h = self.nf(x)
        if targets is None:
            return self.head(h)
        # chunked head + cross-entropy so peak memory is one chunk of fp32 logits;
        # checkpointed so backward recomputes the chunk's logits instead of storing them.
        def chunk_loss(hc, tc):
            lg = self.head(hc).float()
            return F.cross_entropy(lg.reshape(-1, lg.size(-1)), tc.reshape(-1), reduction="sum")
        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, B, self.loss_chunk):
            hc, tc = h[i:i + self.loss_chunk], targets[i:i + self.loss_chunk]
            if torch.is_grad_enabled():
                total = total + checkpoint(chunk_loss, hc, tc, use_reentrant=False)
            else:
                total = total + chunk_loss(hc, tc)
        return total / (B * T)

    def n_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.emb.weight.numel() + self.head.weight.numel() + (self.pos.weight.numel() if self.pos is not None else 0)
        return n


SIZES = {
    "XS": dict(d=128,  n_layer=2,  n_head=2),   # smoke tests only
    "S":  dict(d=384,  n_layer=6,  n_head=6),
    "M":  dict(d=512,  n_layer=8,  n_head=8),
    "L":  dict(d=768,  n_layer=12, n_head=12),
    "XL16": dict(d=1024, n_layer=16, n_head=16),   # ~200M non-emb; fits a 12 GB GPU with counting at micro-batch 2
    "XL": dict(d=1024, n_layer=24, n_head=16),
}
