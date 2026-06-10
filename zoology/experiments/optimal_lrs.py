"""Calibrated optimal learning rates per RoLA inner kernel, for reuse across
arch/kernel/seed sweeps. Source: rola_lr_probe_a + rola_lr_confirm (RLA/GLA,
seed-robust n=3) + rola_lr_kernels (faithful hedgehog/based/rebased + gdn low-LR).
All at nc=16, ~10k state, MQAR data_ext, 40 epochs.

Monolith (nc=1) LRs ARE now calibrated (rola_lr_mono RLA/GLA/GDN d48; rola_lr_mono2
hedgehog/based/rebased) — see OPTIMAL_LR_MONO below. KEY: the monolith optimum sits one
notch DOWN from the routed optimum for every kernel (≈3e-3 monolith vs 1e-2 routed for
RLA/GLA/Based) — a CONSISTENT downshift, so nc=1↔16 LR transfer is NOT 1:1. Use the
nc-appropriate table when comparing RoLA-vs-monolith (else you risk the mis-tuning that
hid faithful Hedgehog).
"""

# kernel -> (lr, best_max_acc, n_seeds, note)
OPTIMAL_LR_ROUTED = {
    'rola-rla-sym':      (1e-2, 0.939, 3, 'confirmed seed-robust; 3e-3 ~tied (0.937)'),
    'rola-gla-sym':      (1e-2, 0.905, 3, 'confirmed seed-robust'),
    'rola-based-sym':    (1e-2, 0.906, 1, 'provisional (single seed)'),
    'rola-hedgehog-sym': (1e-3, 0.902, 1, 'FAITHFUL impl; old 3e-3 was broken impl; still < RLA'),
    'rola-rebased-sym':  (3e-3, 0.882, 1, 'provisional; < Based (dropped const/linear + LN)'),
    'rola-gdn-sym':      (1e-3, 0.679, 1, 'bracketed (3e-4=.65<1e-3=.68>3e-3=.60); genuinely weak'),
}

# MONOLITH (nc=1, ~10k state: RLA/GLA/GDN at d_qk=48 square; based d8v51, hedgehog
# d48v48, rebased d9v51 to match feat_dim state). Source: rola_lr_mono + rola_lr_mono2,
# single seed, MQAR data_ext, 40ep. kernel -> (lr, best_max_acc, n_seeds, note)
OPTIMAL_LR_MONO = {
    'rola-rla-sym':      (3e-3, 0.910, 1, '1e-2=0.908 ~tied; 1e-3=0.889; 1e-1 collapses (0.43)'),
    'rola-gla-sym':      (3e-3, 0.792, 1, 'sharp peak at 3e-3; 1e-2/1e-3 both 0.628'),
    'rola-gdn-sym':      (3e-3, 0.828, 1, 'MONO >> routed (0.679): routing hurts GDN; 1e-4 collapses'),
    'rola-based-sym':    (3e-3, 0.843, 1, '1e-2=0.805, 3e-2=0.808'),
    'rola-hedgehog-sym': (3e-3, 0.844, 1, '1e-2=0.819; 1e-3=0.769'),
    'rola-rebased-sym':  (1e-2, 0.844, 1, 'flat 0.839-0.844 across 1e-3..1e-2'),
}

# LR×nc-STATE INVARIANCE — CONFIRMED (rola_lr_statecheck RLA + statecheck2 GLA/Hedgehog/
# Based/ReBased, each at nc=3 vs nc=256). For ALL 5 kernels the optimum is the same
# 3e-3–1e-2 plateau (and 3e-2 collapses) at both low and high state → the routed LR is
# invariant along the nc axis; reuse one LR per kernel across the crossover sweep, no
# per-nc retuning. (Accuracy itself climbs ~0.66@nc3 → ~0.95-0.999@nc256 for every kernel
# — the rank/recall-capacity effect, kernel-agnostic.)

def lr_for(name, regime='routed'):
    """Best LR for a RoLA instance. regime='routed' (nc>1) or 'mono' (nc=1).
    Falls back to 1e-2 (routed) / 3e-3 (mono)."""
    if regime == 'mono':
        return OPTIMAL_LR_MONO.get(name, (3e-3,))[0]
    return OPTIMAL_LR_ROUTED.get(name, (1e-2,))[0]
