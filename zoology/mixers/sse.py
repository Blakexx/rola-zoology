"""Faithful standalone SSE (Sparse State Expansion), arXiv 2507.16577 (Pan et al., 2025).

COMPLETELY independent of rola.py — clean-room from the paper's Eqs 9-14 + §4.1/§4.2/App. B-C,
revised after an independent adversarial transcription review (2026-06-10):
  * Gate is PER-TOKEN (e_t ∈ R^N shared across heads — App. B algorithms take E ∈ R^{L×N}).
  * The always-selected partition is a SEPARATE (N+1)-th shared partition OUTSIDE the gate:
    ungated full-strength writes/reads ("no masking is applied to the shared portion", §4.2),
    with LoRA-decoupled QK (App. C; V shared, footnote 7).
  * Balance aux loss (partition-level, footnote 6) over the N SPARSE partitions only, delivered
    via get_auxiliary_loss() (the zoology trainer's hook).

    e_t = softmax(x_t W_e) ∈ R^N ;  T_t = top-k(e_t)
    q_t = x_t W_q ∈ R^c (plain) ;  v_t = x_t W_v ;  k_t = softmax(x_t W_k) ∈ R^c
    S^i_t = S^i_{t-1} + e^i_t k_tᵀ v_t  for i ∈ T_t          (sparse partitions; Λ=I — documented
                                                              deviation: paper's LM uses diagonal
                                                              gating; Λ-agnostic per §3)
    S^sh_t = S^sh_{t-1} + k̃_tᵀ v_t                           (shared partition, ungated;
                                                              k̃ = softmax(x(W_k + LoRA)), q̃ likewise)
    o_t = q̃_t S^sh_t + Σ_{i∈T_t} e^i_t q_t S^i_t             (no normalizer — paper Eq 14)

State per head per layer: (N+1)·c·d_head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt


class SSE(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, num_rows: int = 16, num_partitions: int = 4,
                 topk: int = 1, d_head: int = None, shared_partition: bool = True, lora_rank: int = 64,
                 balance_alpha: float = 1e-2, layer_idx: int = None, **kwargs):
        super().__init__()
        self.d_model, self.H = d_model, n_heads
        self.c, self.N, self.k = num_rows, num_partitions, topk
        self.dh = d_head if d_head is not None else d_model // n_heads
        self.shared = shared_partition
        self.balance_alpha = balance_alpha
        H, c, N, dh = self.H, self.c, self.N, self.dh
        self.W_q = nn.Linear(d_model, H * c, bias=False)
        self.W_k = nn.Linear(d_model, H * c, bias=False)
        self.W_v = nn.Linear(d_model, H * dh, bias=False)
        self.W_e = nn.Linear(d_model, N, bias=False)          # PER-TOKEN gate (shared across heads)
        self.W_o = nn.Linear(H * dh, d_model, bias=False)
        if self.shared:
            r = min(lora_rank, d_model, H * c)
            self.lora_qA = nn.Linear(d_model, r, bias=False); self.lora_qB = nn.Linear(r, H * c, bias=False)
            self.lora_kA = nn.Linear(d_model, r, bias=False); self.lora_kB = nn.Linear(r, H * c, bias=False)
            nn.init.zeros_(self.lora_qB.weight); nn.init.zeros_(self.lora_kB.weight)
        self._aux = None

    def _gates(self, x):
        e = F.softmax(self.W_e(x), dim=-1)                                    # [B,L,N] per token
        idx = e.topk(min(self.k, self.N), dim=-1).indices
        m = torch.zeros_like(e).scatter(-1, idx, 1.0)
        return e, m

    @staticmethod
    def _causal_linear(qe, ke, vv, B, H, c, dh):
        """Exact chunked causal linear attention; checkpoint-friendly."""
        L = qe.shape[1]
        C = 64
        pad = (-L) % C
        if pad:
            qe = F.pad(qe, (0, 0, 0, 0, 0, pad)); ke = F.pad(ke, (0, 0, 0, 0, 0, pad))
            vv = F.pad(vv, (0, 0, 0, 0, 0, pad))
        n = (L + pad) // C
        qc = qe.view(B, n, C, H, c); kc = ke.view(B, n, C, H, c); vc = vv.view(B, n, C, H, dh)
        Gc = torch.einsum('bnihf,bnjhf->bnhij', qc, kc)
        causal = torch.tril(torch.ones(C, C, device=qe.device, dtype=qe.dtype))
        o_intra = torch.einsum('bnhij,bnjhd->bnihd', Gc * causal, vc)
        KV = torch.einsum('bnjhf,bnjhd->bnhfd', kc, vc)
        S = torch.cumsum(KV, dim=1) - KV
        o_inter = torch.einsum('bnihf,bnhfd->bnihd', qc, S)
        return (o_intra + o_inter).reshape(B, n * C, H, dh)[:, :L]

    def forward(self, x, **kwargs):
        B, L, _ = x.shape
        H, c, dh = self.H, self.c, self.dh
        q = self.W_q(x).view(B, L, H, c)
        kk = F.softmax(self.W_k(x).view(B, L, H, c), dim=-1)
        v = self.W_v(x).view(B, L, H, dh)
        e, m = self._gates(x)
        E = (e * m).unsqueeze(2)                                              # [B,L,1,N] → broadcast heads
        o = None
        for i in range(self.N):                                               # sparse gated partitions
            Ei = E[..., i].unsqueeze(-1)                                      # [B,L,1,1]
            oi = _ckpt(self._causal_linear, Ei * q, Ei * kk, v, B, H, c, dh, use_reentrant=False)
            o = oi if o is None else o + oi
        if self.shared:                                                       # ungated shared partition
            q_sh = q + self.lora_qB(self.lora_qA(x)).view(B, L, H, c)
            k_sh = F.softmax((self.W_k(x) + self.lora_kB(self.lora_kA(x))).view(B, L, H, c), dim=-1)
            o = o + _ckpt(self._causal_linear, q_sh, k_sh, v, B, H, c, dh, use_reentrant=False)
        # partition balance (footnote 6), SPARSE partitions only; shared is outside the gate
        f = m.float().mean(dim=(0, 1))
        self._aux = self.balance_alpha * (self.N / max(self.k, 1)) * (f * e.float().mean(dim=(0, 1))).sum()
        return self.W_o(o.reshape(B, L, H * dh))

    def get_auxiliary_loss(self):
        a = self._aux
        self._aux = None
        return a if a is not None else 0.0

    def state_size(self, sequence_length: int = None, **kwargs):
        return self.H * (self.N + (1 if self.shared else 0)) * self.c * self.dh


def _recurrent_reference(x, mod):
    """Token-by-token Eqs 9-14 (+ ungated shared partition) for verification."""
    B, L, _ = x.shape
    H, c, N, dh = mod.H, mod.c, mod.N, mod.dh
    q = mod.W_q(x).view(B, L, H, c)
    kk = F.softmax(mod.W_k(x).view(B, L, H, c), dim=-1)
    v = mod.W_v(x).view(B, L, H, dh)
    e, m = mod._gates(x)
    Em = (e * m).unsqueeze(2).expand(B, L, H, N)
    S = x.new_zeros(B, H, N, c, dh)
    if mod.shared:
        q_sh = q + mod.lora_qB(mod.lora_qA(x)).view(B, L, H, c)
        k_sh = F.softmax((mod.W_k(x) + mod.lora_kB(mod.lora_kA(x))).view(B, L, H, c), dim=-1)
        Ssh = x.new_zeros(B, H, c, dh)
    outs = []
    for t in range(L):
        S = S + torch.einsum('bhn,bhc,bhd->bhncd', Em[:, t], kk[:, t], v[:, t])
        ot = torch.einsum('bhn,bhc,bhncd->bhd', Em[:, t], q[:, t], S)
        if mod.shared:
            Ssh = Ssh + torch.einsum('bhc,bhd->bhcd', k_sh[:, t], v[:, t])
            ot = ot + torch.einsum('bhc,bhcd->bhd', q_sh[:, t], Ssh)
        outs.append(ot)
    o = torch.stack(outs, dim=1)
    return mod.W_o(o.reshape(B, L, H * dh))
