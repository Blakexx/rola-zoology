"""RLA Extreme — sharp test of P2 (decoupling) at LARGE matched state.

Hypothesis: CLA's value should appear at the diminishing-returns regime of
single-state RLA. We've measured up to state ≈ 16k (d=64 baseline) and CLA
hasn't beaten matched-state RLA. The question: does the curve flatten further
out at state ~150k?

Three architectures at matched state ≈ 153,600 floats per layer (V+1 norm):
  cla-rla-d24-nc64 : 4·64·24·25 = 153,600  (chunks of narrow d)
  rla-d1536-dv24   : 4· 1·1536·25 = 153,600  (wide asymmetric — wide d_qk, narrow d_v)
  rla-d196-dv196   : 4· 1·196·197 = 154,448  (wide symmetric, 0.5% over)

Smoke verified all three instantiate + forward cleanly at d_model=128, n_heads=4.

PARAM-COUNT NOTE: matched state ≠ matched params. Wider d_qk grows the Q/K
projection significantly. RLA-d1536-dv24 has ~2.4× the params of CLA-d24-nc64.
This is an inherent architectural property; reported in results table.

3 cells × 4 LRs × 5 seeds = 60 runs.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, LoggerConfig
from zoology.experiments.rla_sweep import (
    data, data_small_batch, wrap_hybrid, cla_rla,
    LRS, SEEDS, D_MODEL, VOCAB,
)


def rla_asymmetric(d_qk: int, d_v: int):
    """RLA with asymmetric d_qk != d_v (e.g., wide d_qk, narrow d_v)."""
    return dict(
        name="zoology.mixers.cla.RecurrentLinearAttention",
        kwargs={"d_qk": d_qk, "d_v": d_v, "n_heads": 4},
    )


def cla_rla_asym(d_qk: int, d_v: int, num_chunks: int):
    """CLA-RLA with asymmetric d_qk != d_v."""
    return dict(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs={
            "d_qk": d_qk, "d_v": d_v, "num_chunks": num_chunks, "n_heads": 4,
            "writer": "softmax_linear", "reader": "softmax_linear",
            "tie_routers": False, "use_short_conv": False, "route_on": "kq",
        },
    )


# Cells. The CLA cell already exists in rla_sweep.py CELLS but we re-list it
# here for self-contained run.
CELLS = [
    # === d24-nc16 matched group (state ≈ 38,400) ===
    ("cla-rla-d24-nc16",   cla_rla(24, 16),               {}),  # 38,400
    ("rla-d384-dv24",      rla_asymmetric(384, 24),       {}),  # 38,400 asymmetric
    ("rla-d98-dv98",       rla_asymmetric(98, 98),        {}),  # 38,808 symmetric
    # === d24-nc64 matched group (state ≈ 153,600) ===
    ("cla-rla-d24-nc64",   cla_rla(24, 64),               {}),  # 153,600
    ("rla-d1536-dv24",     rla_asymmetric(1536, 24),      {}),  # 153,600 asymmetric
    ("rla-d196-dv196",     rla_asymmetric(196, 196),      {}),  # 154,448 symmetric
    # === d16-nc16 matched group (state ≈ 17,408) — added without rebuild! ===
    ("cla-rla-d16-nc16",   cla_rla(16, 16),               {}),  # 17,408
    ("rla-d256-dv16",      rla_asymmetric(256, 16),       {}),  # 17,408 asymmetric
    ("rla-d66-dv66",       rla_asymmetric(66, 66),        {}),  # 17,688 symmetric
    # === asymmetric-CLA group (state ≈ 4,608) — d_qk=16, d_v=8, nc=8 ===
    ("cla-rla-d16-dv8-nc8", cla_rla_asym(16, 8, 8),       {}),  # 4·8·16·9 = 4,608
    ("rla-d34-dv34",       rla_asymmetric(34, 34),        {}),  # 4·34·35 = 4,760 (closest sym ≥)
    ("rla-d128-dv8",       rla_asymmetric(128, 8),        {}),  # 4·128·9 = 4,608 (matched asym)
    # === asym-CLA variants (state varies) — (nc, d_qk, d_v) ===
    ("cla-rla-nc16-d8-dv8",  cla_rla_asym(8, 8, 16),      {}),  # 4·16·8·9  = 4,608
    ("cla-rla-nc12-d11-dv8", cla_rla_asym(11, 8, 12),     {}),  # 4·12·11·9 = 4,752
    ("cla-rla-nc16-d16-dv8", cla_rla_asym(16, 8, 16),     {}),  # 4·16·16·9 = 9,216
    # === state ≈ 9,216 matched group ===
    ("cla-rla-nc32-d8-dv8",  cla_rla_asym(8, 8, 32),      {}),  # 4·32·8·9  = 9,216
    ("rla-d256-dv8",         rla_asymmetric(256, 8),      {}),  # 4·256·9   = 9,216 asym
    ("rla-d48-dv48",         rla_asymmetric(48, 48),      {}),  # 4·48·49   = 9,408 sym
]

# Wide-d_qk cells have very wide Q/K activations.
# At B=256, T=256, fp32: Q intermediate = 256·256·4·d_qk·4 bytes.
#   d=1536 → 1.6 GB each for Q/K → tight on 12 GB local, fine on A100 40 GB
#   d=384 → 400 MB each → fits even locally
# Keep batch=256 on cloud A100; drop to 128 only for the extreme d=1536.
OOM_CELLS = {"rla-d1536-dv24"}

configs = []
configs_envs = []
for name, kernel, env in CELLS:
    mixer = wrap_hybrid(kernel)
    cell_data = data_small_batch if name in OOM_CELLS else data
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
                    logger=LoggerConfig(project_name="rla-extreme", entity=""),
                    max_epochs=32, learning_rate=lr, weight_decay=0.0, seed=seed,
                    run_id=f"{name}_lr{lr:.2e}_s{seed}",
                    early_stopping_threshold=0.99,
                    early_stopping_metric="valid/accuracy",
                    slice_keys=["num_kv_pairs"],
                )
            )
            configs_envs.append(env)

assert len(configs) == 360, f"expected 360 configs (18 cells × 20), got {len(configs)}"


def load_configs_and_envs():
    return configs, configs_envs
