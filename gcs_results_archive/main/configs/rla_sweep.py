"""RLA Sweep — Experiment A from resources/rla_sweep.md.

Tests two hypotheses on CLA-RLA (ChunkedLinearAttention with softmax_linear
writer/reader, V+1 on):
  P1 (Interpolation): at fixed per-chunk d, accuracy rises monotonically toward
      MHA as n_chunks grows.
  P2 (Decoupling): at matched total state, accuracy is invariant to allocation.

Sweep:
  3a. Decoupling grid: d ∈ {16, 24, 32, 48, 64} × n_chunks ∈ {1, 2, 4}  → 15 cells
  3b. Interpolation column at d=16: n_chunks ∈ {8, 16, 32, 64}            →  4 cells
  3c. MHA ceiling                                                          →  1 cell
  Total: 20 cells × 4 LRs × 5 seeds = 400 runs.

Task: Zoology canonical multi-task MQAR (matches original_mqar_configs.py).
LR sweep: np.logspace(-4, -2, 4) (extended below repo default to reach 1e-4).
"""
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB = 8_192
D_MODEL = 128
SEEDS = [1337, 42, 7, 0, 1]
# Spec §2: extends below the repo's logspace(-3, -1.5, 4) to capture the
# low-LR optimum the pilot data shows.
LRS = np.logspace(-4, -2, 4).tolist()  # ~{1e-4, 4.64e-4, 2.15e-3, 1e-2}
BATCH = 256

# Multi-task data matching original_mqar_configs.py exactly.
train_configs = [
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,  num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB, input_seq_len=128, num_examples=20_000,  num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=20_000,  num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=20_000,  num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=20_000,  num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,  num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,  num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB, input_seq_len=64,  num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB, input_seq_len=128, num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB, input_seq_len=512, num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB, input_seq_len=1024,num_examples=1_000, num_kv_pairs=256),
]
INPUT_SEQ_LEN = max(c.input_seq_len for c in train_configs + test_configs)

data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(BATCH, BATCH // 8),
    cache_dir="/tmp/zoology_cache_rla_sweep",
)

# Lower-batch DataConfig for cells projected to OOM on the local 3080 Ti at
# batch=256: cla-rla-d16-{nc16,nc32,nc64} and cla-rla-d64-nc4. Same cache_dir
# because the underlying examples are identical; only batching changes.
data_small_batch = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(128, 32),
    cache_dir="/tmp/zoology_cache_rla_sweep",
)
# Cells that OOM on local 3080 Ti at batch=256 (12 GB VRAM with BaseConv FFT
# overhead). The boundary is ~250 MB per virt-tensor (activation B·L·H·C·d_qk).
# Empirically validated: d=32 nc=8 (268 MB per virt) OOM'd in pilot.
# d=48 nc=4 (200 MB) is a safety-margin inclusion.
OOM_CELLS = {
    "cla-rla-d16-nc16",   # 134 MB per virt × 2 (C=16 vs C=8) = 268 MB
    "cla-rla-d16-nc32",   # 536 MB
    "cla-rla-d16-nc64",   # 1.07 GB
    "cla-rla-d48-nc4",    # 200 MB — safety margin
    "cla-rla-d64-nc4",    # 268 MB — matches d=32 nc=8 (empirical OOM)
}

base_conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={"l_max": INPUT_SEQ_LEN, "kernel_size": 3, "implicit_long_conv": True},
)


def wrap_hybrid(kernel_kwargs):
    return ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [base_conv_mixer, kernel_kwargs]},
    )


# MHA ceiling, canonical kwargs (matches add_attention in models_repo.py).
mha = dict(
    name="zoology.mixers.attention.MHA",
    kwargs={"num_heads": 2, "dropout": 0.1},
)


def cla_rla(d, num_chunks):
    """CLA-RLA: softmax_linear writer + reader (V+1 on), route_on='kq', n_heads=4."""
    return dict(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs={
            "d_qk": d, "d_v": d, "num_chunks": num_chunks, "n_heads": 4,
            "writer": "softmax_linear", "reader": "softmax_linear",
            "tie_routers": False, "use_short_conv": False, "route_on": "kq",
        },
    )


def rla_baseline(d):
    """Canonical single-state RLA (V+1 normalized, ELU+1 q/k). Same FLA kernel
    CLA's softmax_linear writer uses, so nc=1 CLA degenerates to this in theory
    — but using the dedicated module sidesteps any 'is the degeneracy correct?'
    reviewer concern. State formula matches: 4·1·(d²+d)."""
    return dict(
        name="zoology.mixers.cla.RecurrentLinearAttention",
        kwargs={"d_qk": d, "d_v": d, "n_heads": 4},
    )


# Recipe (asym router init + curriculum LR) was ABLATED OUT.
# Phase-1 ablation (4 base_lrs × 3 writer-curriculum scales = 12 cells at d=16
# nc=8) showed recipe-on max=0.916 vs recipe-off=0.940 across the tested grid.
# Driver of the regression is the asymmetric writer init (peaky std=1.0 on
# W_route parameter under route_on='kq'); curriculum is a 2nd-order effect.
# Hypothesis: asymmetric init works for gated kernels (GDN/GLA — internal
# normalization absorbs noisy peaky-route signal) but hurts plain LA (V+1
# denominator destabilizes). See resources/SWEEP_RESULTS.md for details.
ASYM_RECIPE_ENV = {  # kept for reference; not used in CELLS below
    "MQAR_ROUTER_STD_WRITE": "1.0",
    "MQAR_ROUTER_STD_READ":  "0.05",
    "MQAR_CURR_MODE": "linear",
    "MQAR_CURR_W_LR_PHASE1": "3.0", "MQAR_CURR_W_LR_PHASE2": "0.3",
    "MQAR_CURR_R_LR_PHASE1": "0.3", "MQAR_CURR_R_LR_PHASE2": "3.0",
}


# State (n_heads=4, V+1): 4 * num_chunks * (d*d + d) = 4·nc·d·(d+1)
# 3a Decoupling grid: 5 d × 3 nc = 15 cells
#     The nc=1 column is the canonical RLA baseline (RecurrentLinearAttention).
#     The nc=2, nc=4 columns are CLA-RLA recipe-OFF.
# 3b Interpolation column at d=16: 4 extra cells (nc=8,16,32,64) — CLA-RLA recipe-OFF
# 3c MHA: 1 cell
CELLS = [
    # ---- 3c MHA ceiling ----
    ("mha-ceiling",         mha,                {}),
    # ---- 3a Decoupling grid ----
    # nc=1 column: canonical RLA baseline (NOT CLA-at-nc=1).
    ("rla-baseline-d16",    rla_baseline(16),   {}),                  # 1,088
    ("rla-baseline-d24",    rla_baseline(24),   {}),                  # 2,400
    ("rla-baseline-d32",    rla_baseline(32),   {}),                  # 4,224
    ("rla-baseline-d48",    rla_baseline(48),   {}),                  # 9,408
    ("rla-baseline-d64",    rla_baseline(64),   {}),                  # 16,640
    # nc≥2 cells: CLA-RLA + recipe.
    ("cla-rla-d16-nc2",     cla_rla(16,  2),    {}),     # 2,176
    ("cla-rla-d16-nc4",     cla_rla(16,  4),    {}),     # 4,352
    ("cla-rla-d24-nc2",     cla_rla(24,  2),    {}),     # 4,800
    ("cla-rla-d24-nc4",     cla_rla(24,  4),    {}),     # 9,600
    ("cla-rla-d32-nc2",     cla_rla(32,  2),    {}),     # 8,448
    ("cla-rla-d32-nc4",     cla_rla(32,  4),    {}),     # 16,896
    ("cla-rla-d48-nc2",     cla_rla(48,  2),    {}),     # 18,816
    ("cla-rla-d48-nc4",     cla_rla(48,  4),    {}),     # 37,632
    ("cla-rla-d64-nc2",     cla_rla(64,  2),    {}),     # 33,280
    ("cla-rla-d64-nc4",     cla_rla(64,  4),    {}),     # 66,560
    # ---- 3b Interpolation column at d=16 ----
    ("cla-rla-d16-nc8",     cla_rla(16,  8),    {}),     # 8,704
    ("cla-rla-d16-nc16",    cla_rla(16, 16),    {}),     # 17,408
    ("cla-rla-d16-nc32",    cla_rla(16, 32),    {}),     # 34,816
    ("cla-rla-d16-nc64",    cla_rla(16, 64),    {}),     # 69,632
]

assert len(CELLS) == 20, f"expected 20 cells, got {len(CELLS)}"

configs = []
configs_envs = []
for name, kernel, env in CELLS:
    mixer = wrap_hybrid(kernel)
    # OOM-projected cells on local 3080 Ti use batch=128 instead of 256.
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
                    logger=LoggerConfig(project_name="rla-sweep", entity=""),
                    max_epochs=32, learning_rate=lr, weight_decay=0.0, seed=seed,
                    run_id=f"{name}_lr{lr:.2e}_s{seed}",
                    early_stopping_threshold=0.99,
                    early_stopping_metric="valid/accuracy",
                    slice_keys=["num_kv_pairs"],
                )
            )
            configs_envs.append(env)

assert len(configs) == 400, f"expected 400 configs, got {len(configs)}"
