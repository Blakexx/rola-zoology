"""Verify clean-room canonical baselines: instantiate every (method, shape, nc) cell and
read the REALIZED recurrent-state size back from the built projections (+ feature-map
probing), comparing to the nominal RoLA budget 624*nc. CPU-only (FLA rolls back), so it
runs without touching the GPU. Prints realized vs target; flags cells off by >12% or that
fail to build. The realized state is what the matched-state table uses as x-coordinate."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch.nn as nn
from zoology.config import ModuleConfig
from zoology.experiments.canonical_baselines import baseline_cell, METHODS, SHAPES, NCS, DV


def realized_state(method, mixer):
    """Recurrent-state floats = sum_heads (key/feature dim) * (value dim), from projections."""
    lins = {n: m for n, m in mixer.named_modules() if isinstance(m, nn.Linear)}
    def out(*names):
        for n in names:
            for k, m in lins.items():
                if k.endswith(n):
                    return m.out_features
        return None
    H = getattr(mixer, "num_heads", None) or getattr(mixer, "n_heads", None)
    dv_total = out("v_proj.weight".replace(".weight", ""), "v_proj", "proj_v", "value")
    dk_total = out("k_proj", "proj_k", "key")
    if method == "based":
        # key feature dim is the taylor expansion of feature_dim, per head; probe the map
        fd = mixer.feature_dim if hasattr(mixer, "feature_dim") else None
        fmap = getattr(mixer, "feature_map", None) or getattr(mixer, "feature_map_q", None)
        with torch.no_grad():
            probe = torch.randn(1, 1, H, fd) if fd else None
            feat = fmap(probe).shape[-1] if (fmap is not None and probe is not None) else None
        dv_head = (dv_total // H) if dv_total else DV
        return (H * feat * dv_head, H, feat, dv_head) if feat else (None, H, None, None)
    if method == "hedgehog":
        fmap = getattr(mixer, "feature_map_q", None) or getattr(mixer, "feature_map", None)
        dk_head = dk_total // H
        with torch.no_grad():
            feat = fmap(torch.randn(1, 1, H, dk_head)).shape[-1] if fmap is not None else dk_head
        dv_head = dv_total // H
        return (H * feat * dv_head, H, feat, dv_head)
    # rla(elu)/gla/gdn: feature dim == key dim per head
    if dk_total and dv_total and H:
        return (dk_total * dv_total // H, H, dk_total // H, dv_total // H)
    return (None, H, None, None)


print(f"{'method':9s} {'shape':6s} {'nc':>4s} {'target':>8s} {'realized':>9s} {'ratio':>6s}  H/dk/dv  status")
n_ok = n_off = n_skip = n_fail = 0
for method in METHODS:
    for shape in SHAPES:
        for nc in NCS:
            cell = baseline_cell(method, shape, nc)
            if cell is None:
                n_skip += 1
                continue
            kdict, target = cell
            try:
                mixer = ModuleConfig(**kdict).instantiate(d_model=128, layer_idx=1)
                st, H, fk, dv = realized_state(method, mixer)
                if st is None:
                    print(f"{method:9s} {shape:6s} {nc:>4d} {target:>8d} {'?':>9s}      ?  H={H} PROBE-FAIL")
                    n_fail += 1; continue
                ratio = st / target
                flag = "OK" if 0.88 <= ratio <= 1.12 else "OFF"
                n_ok += (flag == "OK"); n_off += (flag == "OFF")
                print(f"{method:9s} {shape:6s} {nc:>4d} {target:>8d} {st:>9d} {ratio:>6.2f}  {H}/{fk}/{dv}  {flag}")
            except Exception as e:
                print(f"{method:9s} {shape:6s} {nc:>4d} {target:>8d}  BUILD-FAIL: {str(e).splitlines()[0][:60]}")
                n_fail += 1
print(f"\nOK={n_ok}  OFF(>12%)={n_off}  skipped(N/A)={n_skip}  failed={n_fail}")
