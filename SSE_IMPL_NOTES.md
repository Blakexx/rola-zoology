# Faithful SSE implementation notes (from arXiv 2507.16577, read 2026-06-09)

STANDALONE baseline — completely decoupled from rola.py. New module zoology/zoology/mixers/sse.py.

## Exact method (Eqs 9-14, §3.2 + §4.1)
- Gate:      e_t = softmax(x_t W_e) ∈ R^N            (N partitions)
- Selection: T_t = top-k partitions by e_t, PLUS one always-selected partition (training stability;
             paper adds it with a LoRA to keep params ~constant — at MQAR scale plain shared params
             for the always-on partition are faithful enough; note as deviation if used).
- Projections (shared across partitions, identity feature map — NO elu/phi):
             q_t = x_t W_q;  v_t = x_t W_v
- Key = CLASSIFIER: k_t = softmax(x_t W_k) ∈ R^c     (softmax over the c state rows; W_k shared)
- Update (row-sparse): S^i_t = Λ_t S^i_{t-1} + e^i_t · k_tᵀ v_t   for i∈T; else unchanged
- Readout: o_t = Σ_{i∈T} e^i_t · q_t S^i_t           (NO denominator/normalization; e NOT renormed over T)
- Λ_t agnostic; their LM uses diagonal gating. For the MQAR baseline run Λ=I (isolates classification
  mechanism, matches the RoLA-RLA no-decay comparison); optional scalar-decay variant later.
- Aux balance loss (partition-level, footnote 6): L_bal = α·(N/k)·Σ_i f_i·ē^i  (f_i = selection freq).
  Check zoology trainer for aux-loss support; if absent, document deviation or patch minimally.

## Parallel training form (MQAR scale; exact, no kernels needed)
Per partition i: masked causal linear attention with k̃^i_j = m^i_j e^i_j k_j, q̃^i_t = m^i_t e^i_t q_t
(m = top-k membership incl. always-on), o = Σ_i q̃^i_t Σ_{j≤t} k̃^i_jᵀ v_j. N ≤ 8 ⇒ N chunked/quadratic
passes, einsum at MQAR scale. Straight-through NOT needed: top-k mask is non-differentiable but e_t
flows through the selected values (paper relies on exactly this — gate trainable because e multiplies
both KV and Q).

## Matched-state config
State per head = N·c·d_v rows×dim (k∈R^c per partition). Match the RoLA ladder state
H·nc·d_qk·(d_v+1): SSE has no normalizer column → match N·c·d = nc·d_qk·(d_v+1) per head, or
follow the prior sweep convention (#42). Sweep at the kappa-norm cell: nc=256-equivalent on T4/local.
PRIOR RESULT (memory): faithful row-sparse SSE ~0.44–0.56 vs RoLA-RLA ~0.99 at matched state;
sparser=worse monotone. This re-impl supersedes that — clean-room, standalone.

## Files
- zoology/zoology/mixers/sse.py        (module; pure torch)
- zoology/zoology/experiments/sse_baseline.py  (MQAR config, mirrors rola_kappa_norm protocol)
