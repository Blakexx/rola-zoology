"""Clean-room canonical baselines for the MQAR matched-state comparison.

Every baseline is the method's OWN published implementation, sharing ZERO code with
RoLA: RLA and Hedgehog are FLA's `LinearAttention` (feature_map 'elu'/'hedgehog'),
GLA is zoology's `GatedLinearAttention`, GDN is zoology's `GatedDeltaNet`, Based is
zoology's `Based` (the Based paper's own MQAR mixer). No routing, no virtual heads,
no shared feature-map or normalization scaffold.

Conv policy (option A, the literature-standard MQAR form): mixer-internal short conv
is OFF. The harness wraps every model as Hybrid([BaseConv, mixer]) over 2 layers, so
layer 0 is a shared BaseConv and layer 1 is the conv-free mixer, identically for RoLA
and every baseline. This is how these baselines were evaluated on MQAR in the source
papers (zoology's own GLA defaults use_short_conv=False for exactly this reason).
Method-defining non-conv defaults (GLA output gate, GDN gate, Based taylor map, norms)
are kept.

Matched state: reference recurrent floats S(nc) = 624*nc, the RoLA-RLA cell at
d_qk=d_v=12, H=4 with its (d_v+1) normalizer column. Baselines spend the same total
recurrent floats on content (d_k*d_v), three ways, in their own parameterization:
  wide   = widen key/feature dim, d_v held ~12, H=4
  square = key dim = value dim, H=4
  heads  = base d_qk=d_v=12, scale head count
Each method's state formula is its own; the companion verifier reads realized state
back from the built projections, so cells are confirmed, never assumed. The realized
state (not the nominal target) is the matched-state x-coordinate in the table.
"""
import math

DV = 12
DMODEL = 128
BASE_H = 4
REF = lambda nc: 624 * nc


def _gla(d_k_total, d_v_total, num_heads):
    return dict(name="zoology.mixers.gla.GatedLinearAttention",
                kwargs=dict(expand_k=d_k_total / DMODEL, expand_v=d_v_total / DMODEL,
                            num_heads=num_heads, use_short_conv=False))


def _lin(feature_map, d_k_total, d_v_total, num_heads):
    return dict(name="fla.layers.LinearAttention",
                kwargs=dict(hidden_size=DMODEL, expand_k=d_k_total / DMODEL,
                            expand_v=d_v_total / DMODEL, num_heads=num_heads,
                            feature_map=feature_map))


def _gdn(head_dim, num_heads, dv_head):
    # FLA's canonical GatedDeltaNet, explicit head_dim widens KEYS (recall axis); expand_v
    # holds d_v at the task level. v_out = expand_v*num_heads*head_dim, so d_v_head=dv_head
    # needs expand_v = dv_head/head_dim. head_dim kernel-capped at 256.
    return dict(name="fla.layers.GatedDeltaNet",
                kwargs=dict(hidden_size=DMODEL, num_heads=num_heads, head_dim=head_dim,
                            expand_v=dv_head / head_dim, use_short_conv=False))


def _based(feature_dim, num_heads):
    return dict(name="zoology.mixers.based.Based",
                kwargs=dict(feature_dim=feature_dim, num_heads=num_heads,
                            feature_name="taylor_exp", use_short_conv=False))


def baseline_cell(method, shape, nc):
    """(kernel_dict, target_state) for a canonical baseline cell, or None if (method, shape)
    is not well-defined. Dims solved to the nominal target; realized state is verified."""
    S = REF(nc)
    H = BASE_H
    if method in ("rla", "hedgehog"):
        fmap = "elu" if method == "rla" else "hedgehog"
        fm = 2 if method == "hedgehog" else 1            # hedgehog softmax([x,-x]) doubles feat dim
        if shape == "wide":                              # H*fm*dk_head*DV = S
            dk = max(1, round(S / (H * DV * fm)));        return _lin(fmap, dk * H, DV * H, H), S
        if shape == "square":                            # H*fm*d*d = S
            d = max(1, round(math.sqrt(S / (H * fm))));   return _lin(fmap, d * H, d * H, H), S
        if shape == "heads":                             # h*fm*DV*DV = S
            h = max(1, round(S / (DV * DV * fm)));        return _lin(fmap, DV * h, DV * h, h), S
    if method == "gla":
        if shape == "wide":
            dk = max(1, round(S / (H * DV)));            return _gla(dk * H, DV * H, H), S
        if shape == "square":
            d = max(1, round(math.sqrt(S / H)));         return _gla(d * H, d * H, H), S
        if shape == "heads":
            h = max(1, round(S / (DV * DV)));            return _gla(DV * h, DV * h, h), S
    if method == "gdn":
        # FLA GatedDeltaNet, head_dim widens KEYS (recall axis), d_v held via expand_v.
        # head_dim kernel-capped at 256, so wide tops out ~nc<=19; square/heads run full.
        if shape == "wide":                               # H*head_dim*DV = S, d_v=12 held
            hd = max(1, round(S / (H * DV)))
            return (None if hd > 256 else (_gdn(hd, H, DV), S))
        if shape == "square":                             # H*d*d = S, head_dim=d_v=d
            d = max(1, round(math.sqrt(S / H)))
            return (None if d > 256 else (_gdn(d, H, d), S))
        if shape == "heads":                              # h*DV*DV = S, head_dim=DV
            h = max(1, round(S / (DV * DV)));            return _gdn(DV, h, DV), S
    if method == "based":
        # Based's only state axis is the Taylor feature_dim: taylor_exp feat = fd^2+fd+1,
        # state ~= (d_model/H * H) * feat = d_model_v * feat. d_v is tied to d_model/H (~11 at
        # H=11), close to the task d_v=12 but not independently settable. wide == feature-dim
        # scaling; square/heads do not change Based's state, so they are not defined.
        taylor = lambda d: d * d + d + 1
        if shape == "wide":
            HB = 11                                       # d_v = d_model//11 ~= 11.6 (nearest to 12)
            dv_head = DMODEL // HB
            tgt_feat = S / (HB * dv_head)
            d = max(2, round(math.sqrt(tgt_feat)))
            d = min((d - 1, d, d + 1), key=lambda x: abs(taylor(max(1, x)) - tgt_feat))
            return _based(max(2, d), HB), S
        return None                                       # square/heads not defined for Based
    # GDN: delta-rule state = d_k*d_v with d_k bounded by d_model; at fixed d_model and held
    # d_v it cannot scale (state pinned at d_model*d_v=1536). Omitted from the matched-state
    # grid (architectural, not hobbling). See note; decision pending.
    return None


METHODS = ["rla", "hedgehog", "gla", "based", "gdn"]   # competition order
SHAPES = ["wide", "square", "heads"]
NCS = [2, 4, 8, 16, 32, 64, 128, 256]
