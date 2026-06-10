"""GLA Sweep — kernel-swap follow-up to rla_sweep.

Identical to rla_sweep.py in every dimension EXCEPT the inner kernel:
  RLA writer ('softmax_linear')  →  GLA writer ('softmax_gla_nov1')
                                    ↑↑↑↑↑↑↑↑↑ NO V+1 normalization
  RecurrentLinearAttention base  →  zoology.mixers.gla.GatedLinearAttention
                                    ↑ FLA's canonical GLA (also no V+1)

The V+1 OFF choice is critical: it makes CLA-GLA mathematically comparable to
the canonical FLA GLA baseline, which drops the denominator. Our in-house
RecurrentGLA / SoftmaxGLAWriter (V+1 ON) is a different variant and would
introduce a normalization confound into this kernel-swap comparison.

Reader stays 'softmax_linear' (consistent across all CLA experiments).
n_heads=4, num_chunks per cell, route_on='kq', tie_routers=False.

Tests: does CLA-GLA on the *exact same* canonical multi-task setup as the RLA
sweep give different behavior than RLA? Pure kernel-swap, all other confounds
held.

State formula WITHOUT V+1 (drops the +d_qk term):
  state = 4 · n_chunks · d · d
        = 4 · n_chunks · d²
slightly smaller than RLA's grid (e.g., d=16 nc=8 → 8,192 here vs 8,704 RLA).

Constraint: FLA's `chunk_gla` Triton kernel requires d_qk ≥ 16. Smallest d in
the grid is 16, so all cells are valid.

20 cells × 4 LRs × 5 seeds = 400 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import (
    data, data_small_batch, wrap_hybrid, mha, OOM_CELLS,
    LRS, SEEDS, D_MODEL, VOCAB,
)


def cla_gla(d, num_chunks):
    """CLA-GLA WITHOUT V+1 — softmax_gla_nov1 writer, softmax_linear reader,
    route_on='kq', n_heads=4. Matches base GLA's normalization choice."""
    return dict(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs={
            "d_qk": d, "d_v": d, "num_chunks": num_chunks, "n_heads": 4,
            "writer": "softmax_gla_nov1", "reader": "softmax_linear",
            "tie_routers": False, "use_short_conv": False, "route_on": "kq",
        },
    )


def gla_baseline(d):
    """Canonical single-state GLA — FLA's GatedLinearAttention.

    NOT our cla_bench.RecurrentGLA (that's our V+1-normalized variant and
    would be unfair to compare against). FLA's GLA is V+1 OFF, matching
    the canonical/canonical-state-scan baseline definitions.

    Param mapping for d_qk=d_v=d (n_heads=2 per Zoology canonical):
      head_k_dim = d_model · expand_k / n_heads
      head_v_dim = d_model · expand_v / n_heads
    For d_model=128, n_heads=2 → expand_k = expand_v = 2·d/128 = d/64.
    So d=16 → expand=0.25, d=64 → expand=1.0, d=128 would be 2.0.
    Hmm, d=16, 24, 32, 48 give non-canonical expand values. For state-
    matching, the relevant invariant is state = head_k_dim × head_v_dim ×
    n_heads = (d_model²/n_heads) · expand_k · expand_v.

    To match CLA-GLA state at nc=1 (which is 4·1·d² = 4d²):
      we need 2 · head_k_dim · head_v_dim = 4d², so head_k_dim·head_v_dim = 2d².
    With n_heads=2 and d_model=128, head_dim_total = 64. Setting both to d
    gives state = 2·d·d = 2d² ≠ 4d². So GLA baseline state formula doesn't
    cleanly match CLA-GLA at nc=1 with n_heads=2.

    Workaround: use n_heads=4 for the baseline too (matching the CLA-GLA),
    and expand to set head_dim correctly. With n_heads=4:
      state = 4 · head_k_dim · head_v_dim
    Set head_k = head_v = d:
      state = 4d²  ✓ matches CLA-GLA at nc=1
      head_k_dim = d → expand_k = n_heads·d/d_model = 4d/128 = d/32
    So d=16 → expand_k=expand_v=0.5; d=32 → 1.0; d=64 → 2.0; etc.
    Non-integer for d=24 (0.75) and d=48 (1.5) — let FLA handle.
    """
    expand = d / 32.0  # so head_k_dim = d_model·expand/n_heads = 128·expand/4 = 32·expand = d
    return dict(
        name="zoology.mixers.gla.GatedLinearAttention",
        kwargs={"num_heads": 4, "expand_k": expand, "expand_v": expand,
                "use_short_conv": False},
    )


# Same OOM rule as RLA. Cell names use "gla" instead of "rla" but the d×nc
# shape is identical so the OOM-class is the same set.
GLA_OOM_CELLS = {n.replace("rla", "gla") for n in OOM_CELLS}

CELLS = [
    # ---- MHA ceiling ----
    ("mha-ceiling",         mha,                {}),
    # ---- Decoupling grid: nc=1 column = FLA GLA baseline ----
    ("gla-baseline-d16",    gla_baseline(16),   {}),
    ("gla-baseline-d24",    gla_baseline(24),   {}),
    ("gla-baseline-d32",    gla_baseline(32),   {}),
    ("gla-baseline-d48",    gla_baseline(48),   {}),
    ("gla-baseline-d64",    gla_baseline(64),   {}),
    # ---- nc≥2: CLA-GLA (no V+1) ----
    ("cla-gla-d16-nc2",     cla_gla(16,  2),    {}),
    ("cla-gla-d16-nc4",     cla_gla(16,  4),    {}),
    ("cla-gla-d24-nc2",     cla_gla(24,  2),    {}),
    ("cla-gla-d24-nc4",     cla_gla(24,  4),    {}),
    ("cla-gla-d32-nc2",     cla_gla(32,  2),    {}),
    ("cla-gla-d32-nc4",     cla_gla(32,  4),    {}),
    ("cla-gla-d48-nc2",     cla_gla(48,  2),    {}),
    ("cla-gla-d48-nc4",     cla_gla(48,  4),    {}),
    ("cla-gla-d64-nc2",     cla_gla(64,  2),    {}),
    ("cla-gla-d64-nc4",     cla_gla(64,  4),    {}),
    # ---- Interpolation column at d=16 ----
    ("cla-gla-d16-nc8",     cla_gla(16,  8),    {}),
    ("cla-gla-d16-nc16",    cla_gla(16, 16),    {}),
    ("cla-gla-d16-nc32",    cla_gla(16, 32),    {}),
    ("cla-gla-d16-nc64",    cla_gla(16, 64),    {}),
]

assert len(CELLS) == 20, f"expected 20 cells, got {len(CELLS)}"

configs = []
configs_envs = []
for name, kernel, env in CELLS:
    mixer = wrap_hybrid(kernel)
    cell_data = data_small_batch if name in GLA_OOM_CELLS else data
    for lr in LRS:
        for seed in SEEDS:
            configs.append(
                TrainConfig(
                    data=cell_data,
                    model=ModelConfig(
                        block_type="TransformerBlock",
                        sequence_mixer=mixer,
                        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                        d_model=D_MODEL, n_layers=2,
                        max_position_embeddings=0,
                        vocab_size=VOCAB,
                    ),
                    logger=LoggerConfig(project_name="gla-sweep", entity=""),
                    max_epochs=32, learning_rate=lr, weight_decay=0.0, seed=seed,
                    run_id=f"{name}_lr{lr:.2e}_s{seed}",
                    early_stopping_threshold=0.99,
                    early_stopping_metric="valid/accuracy",
                    slice_keys=["num_kv_pairs"],
                )
            )
            configs_envs.append(env)

assert len(configs) == 400, f"expected 400 configs, got {len(configs)}"


def load_configs_and_envs():
    return configs, configs_envs
