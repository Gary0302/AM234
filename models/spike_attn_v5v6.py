"""v5 / v5_ctrl / v6 / v6_ctrl — hierarchical attention + SNN/Dense readout.

Model map:
  v5      : DoubleAttnEncoder + DoubleLIFReadout  (T=50)
  v5_ctrl : DoubleAttnEncoder + SingleLIFReadout  (T=50)
  v6      : DoubleAttnEncoder + DenseReadout      (T=50)
  v6_ctrl : NormalAttnEncoder + DenseReadout      (T=20)

DoubleAttnEncoder:
  CausalConv1d(4→d, k=5) → [B,d,1000]
  reshape → [B, 50, 20, d]          (50 windows × 20 tokens)
  per-window self-attn              → [B, 50, 20, d]
  max-pool → [B, 50, d]
  Linear(d→919) → [B, 50, 919]

NormalAttnEncoder (v1/v2 style):
  reshape → [B, 20, 50, 4]          (20 chunks × 50 tokens)
  per-chunk self-attn               → [B, 20, 50, d]
  max-pool → [B, 20, d]
  Linear(d→919) → [B, 20, 919]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


# ── Shared building blocks ──────────────────────────────────────────────────

class CausalConvEmbed(nn.Module):
    """One-hot [B,4,L] → [B,d,L], position i sees only nucleotides ≤ i."""
    def __init__(self, d_out: int = 128, kernel_size: int = 5):
        super().__init__()
        self.k = kernel_size
        self.conv = nn.Conv1d(4, d_out, kernel_size=kernel_size, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.k - 1, 0)))   # left-only causal pad


class DoubleAttnEncoder(nn.Module):
    """CausalEmbed + 50-window self-attention → [B, 50, n_tracks]."""
    def __init__(self, n_tracks: int = 919, d: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.embed = CausalConvEmbed(d_out=d, kernel_size=5)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d, n_tracks)
        self._inv_sqrt = 1.0 / math.sqrt(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 4, 1000]
        B = x.shape[0]
        h = self.embed(x).transpose(1, 2)                  # [B, 1000, d]
        h = h.reshape(B, 50, 20, self.d)                   # [B, 50, 20, d]
        Q, K, V = self.W_q(h), self.W_k(h), self.W_v(h)
        A = self.drop(F.softmax(Q @ K.transpose(-1, -2) * self._inv_sqrt, dim=-1))
        out = (A @ V).max(dim=2).values                    # [B, 50, d]
        return self.proj(out)                              # [B, 50, 919]


class NormalAttnEncoder(nn.Module):
    """Raw one-hot + 20-chunk self-attention → [B, 20, n_tracks]."""
    def __init__(self, n_tracks: int = 919, d: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.W_q = nn.Linear(4, d, bias=False)
        self.W_k = nn.Linear(4, d, bias=False)
        self.W_v = nn.Linear(4, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d, n_tracks)
        self._inv_sqrt = 1.0 / math.sqrt(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 4, 1000]
        B = x.shape[0]
        h = x.transpose(1, 2).reshape(B, 20, 50, 4)       # [B, 20, 50, 4]
        Q, K, V = self.W_q(h), self.W_k(h), self.W_v(h)
        A = self.drop(F.softmax(Q @ K.transpose(-1, -2) * self._inv_sqrt, dim=-1))
        out = (A @ V).max(dim=2).values                    # [B, 20, d]
        return self.proj(out)                              # [B, 20, 919]


class DoubleLIFReadout(nn.Module):
    """[B, T, n] → logit [B, n] via two LIF layers."""
    def __init__(self, n: int = 919, beta1: float = 0.9, beta2: float = 0.7,
                 V_th: float = 0.5, slope: float = 25.0):
        super().__init__()
        self.n, self.V_th = n, V_th
        sg = surrogate.fast_sigmoid(slope=slope)
        self.input_scale = nn.Parameter(torch.tensor(1.0))
        self.W_rec1 = nn.Parameter(torch.empty(n, n))
        nn.init.orthogonal_(self.W_rec1, gain=0.05)
        self.lif1 = snn.Leaky(beta=beta1, threshold=V_th, spike_grad=sg,
                               reset_mechanism='subtract', init_hidden=False)
        self.W_rec2 = nn.Parameter(torch.empty(n, n))
        nn.init.orthogonal_(self.W_rec2, gain=0.05)
        self.lif2 = snn.Leaky(beta=beta2, threshold=V_th, spike_grad=sg,
                               reset_mechanism='subtract', init_hidden=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, T, n]
        B, T, n = x.shape
        dev, dt = x.device, x.dtype
        m1 = s1 = m2 = s2 = torch.zeros(B, n, device=dev, dtype=dt)
        for t in range(T):
            s1, m1 = self.lif1(self.input_scale * x[:, t] + s1 @ self.W_rec1.t(), m1)
            s2, m2 = self.lif2(s1 + s2 @ self.W_rec2.t(), m2)
        return m2 - self.V_th


class SingleLIFReadout(nn.Module):
    """[B, T, n] → logit [B, n] via one LIF layer."""
    def __init__(self, n: int = 919, beta: float = 0.9,
                 V_th: float = 0.5, slope: float = 25.0):
        super().__init__()
        self.n, self.V_th = n, V_th
        self.input_scale = nn.Parameter(torch.tensor(1.0))
        self.W_rec = nn.Parameter(torch.empty(n, n))
        nn.init.orthogonal_(self.W_rec, gain=0.05)
        self.lif = snn.Leaky(beta=beta, threshold=V_th,
                              spike_grad=surrogate.fast_sigmoid(slope=slope),
                              reset_mechanism='subtract', init_hidden=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, n = x.shape
        dev, dt = x.device, x.dtype
        m = s = torch.zeros(B, n, device=dev, dtype=dt)
        for t in range(T):
            s, m = self.lif(self.input_scale * x[:, t] + s @ self.W_rec.t(), m)
        return m - self.V_th


class DenseReadout(nn.Module):
    """[B, T, n] → logit [B, n] via mean-pool + 2-layer MLP."""
    def __init__(self, n: int = 919):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n, n),
            nn.BatchNorm1d(n),
            nn.ReLU(),
            nn.Linear(n, n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.mean(dim=1))


# ── 4 public model classes ──────────────────────────────────────────────────

def _count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SpikeAttnV5(nn.Module):
    """DoubleAttn + DoubleLIF"""
    def __init__(self, n_tracks=919, d=128, **kw):
        super().__init__()
        self.encoder = DoubleAttnEncoder(n_tracks, d)
        self.readout = DoubleLIFReadout(n_tracks, **kw)
    def forward(self, x): return self.readout(self.encoder(x))
    def count_params(self):
        e, r = _count(self.encoder), _count(self.readout)
        return {'total': e+r, 'encoder': e, 'readout': r}

class SpikeAttnV5Ctrl(nn.Module):
    """DoubleAttn + SingleLIF"""
    def __init__(self, n_tracks=919, d=128, **kw):
        super().__init__()
        self.encoder = DoubleAttnEncoder(n_tracks, d)
        self.readout = SingleLIFReadout(n_tracks, **kw)
    def forward(self, x): return self.readout(self.encoder(x))
    def count_params(self):
        e, r = _count(self.encoder), _count(self.readout)
        return {'total': e+r, 'encoder': e, 'readout': r}

class DenseAttnV6(nn.Module):
    """DoubleAttn + Dense"""
    def __init__(self, n_tracks=919, d=128, **kw):
        super().__init__()
        self.encoder = DoubleAttnEncoder(n_tracks, d)
        self.readout = DenseReadout(n_tracks)
    def forward(self, x): return self.readout(self.encoder(x))
    def count_params(self):
        e, r = _count(self.encoder), _count(self.readout)
        return {'total': e+r, 'encoder': e, 'readout': r}

class DenseAttnV6Ctrl(nn.Module):
    """NormalAttn + Dense"""
    def __init__(self, n_tracks=919, d=128, **kw):
        super().__init__()
        self.encoder = NormalAttnEncoder(n_tracks, d)
        self.readout = DenseReadout(n_tracks)
    def forward(self, x): return self.readout(self.encoder(x))
    def count_params(self):
        e, r = _count(self.encoder), _count(self.readout)
        return {'total': e+r, 'encoder': e, 'readout': r}



class LinearEmbedAttnEncoder(nn.Module):
    """Linear(4→d) per position + 20-chunk 50-token self-attention → [B, 20, n_tracks].
    Ablation: isolates input dim effect from k=5 causal context in DoubleAttnEncoder."""
    def __init__(self, n_tracks: int = 919, d: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.embed = nn.Linear(4, d, bias=False)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d, n_tracks)
        self._inv_sqrt = 1.0 / math.sqrt(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 4, 1000]
        B = x.shape[0]
        h = self.embed(x.transpose(1, 2))                  # [B, 1000, d]
        h = h.reshape(B, 20, 50, self.d)                   # [B, 20, 50, d]
        Q, K, V = self.W_q(h), self.W_k(h), self.W_v(h)
        A = self.drop(F.softmax(Q @ K.transpose(-1, -2) * self._inv_sqrt, dim=-1))
        out = (A @ V).max(dim=2).values                    # [B, 20, d]
        return self.proj(out)                              # [B, 20, 919]


class DenseAttnV6a(nn.Module):
    """LinearEmbed(4→128) + 50bp windows + Dense — ablation: input dim only"""
    def __init__(self, n_tracks=919, d=128, **kw):
        super().__init__()
        self.encoder = LinearEmbedAttnEncoder(n_tracks, d)
        self.readout = DenseReadout(n_tracks)
    def forward(self, x): return self.readout(self.encoder(x))
    def count_params(self):
        e, r = _count(self.encoder), _count(self.readout)
        return {'total': e+r, 'encoder': e, 'readout': r}


class TwentyBPRawAttnEncoder(nn.Module):
    """Raw 4-dim + 50-window 20bp self-attention → [B, 50, n_tracks].
    Ablation v6_b: isolates window size (20bp) from CausalConv context."""
    def __init__(self, n_tracks: int = 919, d: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.W_q = nn.Linear(4, d, bias=False)
        self.W_k = nn.Linear(4, d, bias=False)
        self.W_v = nn.Linear(4, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d, n_tracks)
        self._inv_sqrt = 1.0 / math.sqrt(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 4, 1000]
        B = x.shape[0]
        h = x.transpose(1, 2).reshape(B, 50, 20, 4)       # [B, 50, 20, 4]
        Q, K, V = self.W_q(h), self.W_k(h), self.W_v(h)
        A = self.drop(F.softmax(Q @ K.transpose(-1, -2) * self._inv_sqrt, dim=-1))
        out = (A @ V).max(dim=2).values                    # [B, 50, d]
        return self.proj(out)                              # [B, 50, 919]


class TrueDoubleAttnEncoder(nn.Module):
    """True hierarchical attention:
    Level 1 (intra-window): CausalConv(k=5) + 20bp attn + max-pool → [B, 50, d]
    Level 2 (inter-window): self-attn across 50 windows               → [B, 50, d]
    Linear(d→n_tracks) → [B, 50, n_tracks] for DenseReadout."""
    def __init__(self, n_tracks: int = 919, d: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.embed = CausalConvEmbed(d_out=d, kernel_size=5)
        # Level 1: intra-window (20 tokens)
        self.W_q1 = nn.Linear(d, d, bias=False)
        self.W_k1 = nn.Linear(d, d, bias=False)
        self.W_v1 = nn.Linear(d, d, bias=False)
        self.drop1 = nn.Dropout(dropout)
        # Level 2: inter-window (50 tokens)
        self.W_q2 = nn.Linear(d, d, bias=False)
        self.W_k2 = nn.Linear(d, d, bias=False)
        self.W_v2 = nn.Linear(d, d, bias=False)
        self.drop2 = nn.Dropout(dropout)
        self.proj = nn.Linear(d, n_tracks)
        self._inv_sqrt = 1.0 / math.sqrt(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 4, 1000]
        B = x.shape[0]
        # Level 1: intra-window attention (20bp)
        h = self.embed(x).transpose(1, 2)                  # [B, 1000, d]
        h = h.reshape(B, 50, 20, self.d)                   # [B, 50, 20, d]
        Q1, K1, V1 = self.W_q1(h), self.W_k1(h), self.W_v1(h)
        A1 = self.drop1(F.softmax(Q1 @ K1.transpose(-1, -2) * self._inv_sqrt, dim=-1))
        h = (A1 @ V1).max(dim=2).values                    # [B, 50, d]
        # Level 2: inter-window attention (50 windows)
        Q2, K2, V2 = self.W_q2(h), self.W_k2(h), self.W_v2(h)
        A2 = self.drop2(F.softmax(Q2 @ K2.transpose(-1, -2) * self._inv_sqrt, dim=-1))
        h = A2 @ V2                                        # [B, 50, d]
        return self.proj(h)                                # [B, 50, 919]


class DenseAttnV6b(nn.Module):
    """20bp windows + raw 4-dim (no context) + Dense — ablation: window size only"""
    def __init__(self, n_tracks=919, d=128, **kw):
        super().__init__()
        self.encoder = TwentyBPRawAttnEncoder(n_tracks, d)
        self.readout = DenseReadout(n_tracks)
    def forward(self, x): return self.readout(self.encoder(x))
    def count_params(self):
        e, r = _count(self.encoder), _count(self.readout)
        return {'total': e+r, 'encoder': e, 'readout': r}



class TwentyBPRawDoubleAttnEncoder(nn.Module):
    """Raw 4-dim + 20bp intra-window attn + inter-window attn.
    2x2 ablation cell A=0 (no CausalConv), B=1 (2-layer attention)."""
    def __init__(self, n_tracks: int = 919, d: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self._inv_sqrt = 1.0 / math.sqrt(d)
        # Level 1: intra-window (20bp, raw 4-dim input)
        self.W_q1 = nn.Linear(4, d, bias=False)
        self.W_k1 = nn.Linear(4, d, bias=False)
        self.W_v1 = nn.Linear(4, d, bias=False)
        self.drop1 = nn.Dropout(dropout)
        # Level 2: inter-window (50 windows)
        self.W_q2 = nn.Linear(d, d, bias=False)
        self.W_k2 = nn.Linear(d, d, bias=False)
        self.W_v2 = nn.Linear(d, d, bias=False)
        self.drop2 = nn.Dropout(dropout)
        self.proj = nn.Linear(d, n_tracks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # [B, 4, 1000]
        B = x.shape[0]
        h = x.transpose(1, 2).reshape(B, 50, 20, 4)       # [B, 50, 20, 4]
        Q1, K1, V1 = self.W_q1(h), self.W_k1(h), self.W_v1(h)
        A1 = self.drop1(F.softmax(Q1 @ K1.transpose(-1, -2) * self._inv_sqrt, dim=-1))
        h = (A1 @ V1).max(dim=2).values                    # [B, 50, d]
        Q2, K2, V2 = self.W_q2(h), self.W_k2(h), self.W_v2(h)
        A2 = self.drop2(F.softmax(Q2 @ K2.transpose(-1, -2) * self._inv_sqrt, dim=-1))
        h = A2 @ V2                                        # [B, 50, d]
        return self.proj(h)                                # [B, 50, 919]


class DenseAttnV6b2(nn.Module):
    """20bp windows + raw 4-dim (no CausalConv) + inter-window attn + Dense.
    Completes the 2x2 factorial: A=0 (no conv), B=1 (2-layer attn)."""
    def __init__(self, n_tracks=919, d=128, **kw):
        super().__init__()
        self.encoder = TwentyBPRawDoubleAttnEncoder(n_tracks, d)
        self.readout = DenseReadout(n_tracks)
    def forward(self, x): return self.readout(self.encoder(x))
    def count_params(self):
        e, r = _count(self.encoder), _count(self.readout)
        return {'total': e+r, 'encoder': e, 'readout': r}


class DenseAttnV7(nn.Module):
    """True double attention: 20bp intra-window + 50-window inter-window + Dense"""
    def __init__(self, n_tracks=919, d=128, **kw):
        super().__init__()
        self.encoder = TrueDoubleAttnEncoder(n_tracks, d)
        self.readout = DenseReadout(n_tracks)
    def forward(self, x): return self.readout(self.encoder(x))
    def count_params(self):
        e, r = _count(self.encoder), _count(self.readout)
        return {'total': e+r, 'encoder': e, 'readout': r}

MODEL_REGISTRY = {
    'v5':       SpikeAttnV5,
    'v5_ctrl':  SpikeAttnV5Ctrl,
    'v6':       DenseAttnV6,
    'v6_ctrl':  DenseAttnV6Ctrl,
    'v6_a':     DenseAttnV6a,
    'v6_b':     DenseAttnV6b,
    'v6_b2':    DenseAttnV6b2,
    'v7':       DenseAttnV7,
}


if __name__ == '__main__':
    torch.manual_seed(0)
    B = 4
    x = F.one_hot(torch.randint(0, 4, (B, 1000)), 4).float().transpose(1, 2)

    for name, Cls in MODEL_REGISTRY.items():
        model = Cls()
        pc = model.count_params()
        logits = model(x)
        assert logits.shape == (B, 919), f'{name}: wrong shape {logits.shape}'
        assert logits.std() > 0.01,       f'{name}: DEAD (std={logits.std():.4f})'
        y = (torch.rand(B, 919) < 0.05).float()
        F.binary_cross_entropy_with_logits(logits, y).backward()
        n_grad = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                     for p in model.parameters() if p.requires_grad)
        n_tot  = sum(1 for _ in model.parameters() if _.requires_grad)
        assert n_grad == n_tot, f'{name}: only {n_grad}/{n_tot} params have grad'
        print(f'{name:10s}  total={pc["total"]/1e6:.3f}M  '
              f'enc={pc["encoder"]/1e3:.0f}K  read={pc["readout"]/1e3:.0f}K  '
              f'logit_std={logits.std():.3f}  grad={n_grad}/{n_tot}  OK')
        # reset grads for next model
        for p in model.parameters():
            p.grad = None
    print('ALL SMOKE TESTS PASS')
