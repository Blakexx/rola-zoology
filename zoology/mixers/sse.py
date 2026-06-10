"""Faithful standalone SSE (Sparse State Expansion), arXiv 2507.16577 (Pan et al., 2025).

COMPLETELY independent of rola.py — clean-room from the paper's Eqs 9-14 (see SSE_IMPL_NOTES.md):
    e_t = softmax(x_t W_e) ∈ R^N            (write-read gate over N partitions)
    T_t = top-k(e_t) [+ always-selected partition 0 for training stability]
    q_t = x_t W_q ∈ R^c   (plain; c = state-row/feature dim)
    v_t = x_t W_v ∈ R^d
    k_t = softmax(x_t W_k) ∈ R^c            (the key IS a softmax classifier over state rows)
    S^i_t = S^i_{t-1} + e^i_t k_tᵀ v_t      for i ∈ T_t   (Λ=I here: isolates the classification
                                             mechanism, matching the no-decay RLA comparison)
    o_t  = Σ_{i∈T_t} e^i_t q_t S^i_t        (NO normalizer; e not renormalized over T — paper §4.1)

Parallel form (exact, used for training at MQAR scale): with E = topk-mask ∘ e ∈ R^{L×N},
    o_t = Σ_{j≤t} (q_t·k_j) · (E_t·E_j) v_j   — content gram × hard-masked partition gram.
Aux partition-balance loss (footnote 6): L_bal = α·(N/k)·Σ_i f_i·ē^i, stashed on the module
(self.aux_loss) for trainers that collect it; recorded as a deviation if unused.
State per head per layer: N·c·d_head floats.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSE(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, num_rows: int = 16, num_partitions: int = 4,
                 topk: int = 1, d_head: int = None, always_on: bool = True,
                 balance_alpha: float = 1e-2, layer_idx: int = None, **kwargs):
        super().__init__()
        self.d_model, self.H = d_model, n_heads
        self.c, self.N, self.k = num_rows, num_partitions, topk
        self.dh = d_head if d_head is not None else d_model // n_heads
        self.always_on = always_on
        self.balance_alpha = balance_alpha
        H, c, N, dh = self.H, self.c, self.N, self.dh
        self.W_q = nn.Linear(d_model, H * c, bias=False)
        self.W_k = nn.Linear(d_model, H * c, bias=False)
        self.W_v = nn.Linear(d_model, H * dh, bias=False)
        self.W_e = nn.Linear(d_model, H * N, bias=False)
        self.W_o = nn.Linear(H * dh, d_model, bias=False)
        self.aux_loss = None

    def _gates(self, x):
        B, L, _ = x.shape
        e = F.softmax(self.W_e(x).view(B, L, self.H, self.N), dim=-1)        # [B,L,H,N]
        if self.always_on:
            # partition 0 always selected; top-k chosen among the remaining N-1 (paper Fig.4:
            # green always-selected partition + blue top-k). k counts the sparse picks.
            idx = e[..., 1:].topk(min(self.k, self.N - 1), dim=-1).indices + 1
            m = torch.zeros_like(e).scatter(-1, idx, 1.0)
            m[..., 0] = 1.0
        else:
            idx = e.topk(min(self.k, self.N), dim=-1).indices
            m = torch.zeros_like(e).scatter(-1, idx, 1.0)
        return e, m

    def forward(self, x, **kwargs):
        B, L, _ = x.shape
        H, c, dh = self.H, self.c, self.dh
        q = self.W_q(x).view(B, L, H, c)
        kk = F.softmax(self.W_k(x).view(B, L, H, c), dim=-1)                 # classifier key
        v = self.W_v(x).view(B, L, H, dh)
        e, m = self._gates(x)
        E = e * m                                                            # masked gate [B,L,H,N]
        # Per-partition chunked linear attention (exact). SSE weight (q_t.k_j)(E_t.E_j) =
        # sum_i (E^i_t q_t).(E^i_j k_j) -> N standard causal passes at feature dim c, each
        # checkpointed (recompute in backward) so activations stay O(one partition).
        from torch.utils.checkpoint import checkpoint as _ckpt

        def _part(qe, ke, vv):
            C = 64
            pad = (-qe.shape[1]) % C
            if pad:
                qe = F.pad(qe, (0, 0, 0, 0, 0, pad)); ke = F.pad(ke, (0, 0, 0, 0, 0, pad))
                vv = F.pad(vv, (0, 0, 0, 0, 0, pad))
            n = qe.shape[1] // C
            qc = qe.view(B, n, C, H, c); kc = ke.view(B, n, C, H, c); vc = vv.view(B, n, C, H, dh)
            Gc = torch.einsum('bnihf,bnjhf->bnhij', qc, kc)
            causal = torch.tril(torch.ones(C, C, device=qe.device, dtype=qe.dtype))
            o_intra = torch.einsum('bnhij,bnjhd->bnihd', Gc * causal, vc)
            KV = torch.einsum('bnjhf,bnjhd->bnhfd', kc, vc)
            S = torch.cumsum(KV, dim=1) - KV
            o_inter = torch.einsum('bnihf,bnhfd->bnihd', qc, S)
            return (o_intra + o_inter).reshape(B, n * C, H, dh)[:, :L]

        o = None
        for i in range(self.N):
            Ei = E[..., i].unsqueeze(-1)                                     # [B,L,H,1]
            oi = _ckpt(_part, Ei * q, Ei * kk, v, use_reentrant=False)
            o = oi if o is None else o + oi
        # partition balance aux (paper footnote 6): f_i = hard selection freq, ē_i = mean gate
        f = m.float().mean(dim=(0, 1, 2))
        self.aux_loss = self.balance_alpha * (self.N / max(self.k, 1)) * (f * e.float().mean(dim=(0, 1, 2))).sum()
        return self.W_o(o.reshape(B, L, H * dh))

    def state_size(self, sequence_length: int = None, **kwargs):
        # recurrent state: N partitions × c rows × d_head per head per layer
        return self.H * self.N * self.c * self.dh


def _recurrent_reference(x, mod):
    """Token-by-token Eqs 9-14 for verification."""
    B, L, _ = x.shape
    H, c, N, dh = mod.H, mod.c, mod.N, mod.dh
    q = mod.W_q(x).view(B, L, H, c)
    kk = F.softmax(mod.W_k(x).view(B, L, H, c), dim=-1)
    v = mod.W_v(x).view(B, L, H, dh)
    e, m = mod._gates(x)
    S = x.new_zeros(B, H, N, c, dh)
    outs = []
    for t in range(L):
        Em = (e[:, t] * m[:, t])                                             # [B,H,N]
        S = S + torch.einsum('bhn,bhc,bhd->bhncd', Em, kk[:, t], v[:, t])
        outs.append(torch.einsum('bhn,bhc,bhncd->bhd', Em, q[:, t], S))
    o = torch.stack(outs, dim=1)                                             # [B,L,H,dh]
    return mod.W_o(o.reshape(B, L, H * dh))
