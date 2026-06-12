"""TRUE Zoology canonical state-scan sweep.

Matches `mqar_example_configs/original_mqar_configs.py` exactly except:
  - Adds CLA-GLA and CLA-GDN state-scan curves at matched state points
  - Adds recipe ablation pairs on CLA cells
  - All FLA + CLA models wrapped in Hybrid([BaseConv, kernel]) per Zoology canonical
  - MHA included as ceiling reference (gets positional info via BaseConv prefix)

Task config (canonical):
  vocab=8192, n_layers=2, d_model=128, state_mixer=Identity, max_pos_emb=0,
  batch=256, max_epochs=32,
  Multi-task TRAIN (5 configs from kv=4/seq=64 up to kv=64/seq=256),
  Multi-test EVAL (7 configs from kv=4/seq=64 up to kv=256/seq=1024),
  Hybrid([BaseConv(k=3, implicit_long_conv=True), kernel]) on every model,
  slice_keys=["num_kv_pairs"] so per-difficulty accuracy is reported.

LR sweep: 4 LRs per cell (np.logspace(-3, -1.5, 4)) = {1e-3, 3.16e-3, 1e-2, 3.16e-2}.
Report max-over-LRs per (cell, seed) as Zoology does.

NOTE: this is much heavier than canonical_state_scan.py:
  - 5x runs (4 LRs × 5 seeds vs 1 LR × 5 seeds)
  - longer per-run (seq up to 256, multi-task ~180k examples)

Cells (mirrors canonical_state_scan.py but with broken cells fixed):
  Baselines:
    1.  MHA (wrapped in Hybrid([BaseConv, MHA]))
    2.  baseline-GLA at FLA defaults (~2,048 state)
    3.  baseline-GLA scaled up: matches CLA state points
    4.  baseline-GDN at FLA defaults (~524k state)
    5.  baseline-GDN scaled down: matches CLA state points

  CLA-GLA curve (n_heads=4, n_chunks=8, V+1, conv+kq routing):
    State points: 2,016 / 8,704 / 33,792 / 264,192
    Each × {recipe, no recipe}

  CLA-GDN curve (n_heads=4, n_chunks=8, no V+1, conv+kq routing):
    State points: 2,048 / 8,192 / 32,768 / 262,144
    Each × {recipe, no recipe}

NOTE on small-state CLA-GLA: dropped d_qk=4 cell — Triton chunk_gla kernel
requires d_qk ≥ 16. The 2,016 state cell (d_qk=7) is the smallest that works.
"""
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB = 8_192
D_MODEL = 128
SEEDS = [1337, 42, 7, 0, 1]
LRS = np.logspace(-3, -1.5, 4).tolist()  # ~{1e-3, 3.16e-3, 1e-2, 3.16e-2}
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
    cache_dir="/tmp/zoology_cache_zcanonical",
)

# Zoology-canonical BaseConv pre-mixer (applied to every kernel via Hybrid).
base_conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={"l_max": INPUT_SEQ_LEN, "kernel_size": 3, "implicit_long_conv": True},
)


def wrap_hybrid(kernel_kwargs):
    return ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [base_conv_mixer, kernel_kwargs]},
    )


# ------- Baselines -------

# Baselines use Zoology canonical kwargs (models_repo.py: add_attention, add_gla, add_gated_delta_net):
#   MHA:        num_heads=2, dropout=0.1
#   FLA-GLA:    num_heads=2, use_short_conv=False
#   FLA-GDN:    num_heads=2, use_gate=False, use_short_conv=True, conv_size=4
# State varies via expand_k/expand_v (GLA) and head_dim/expand_v (GDN).
mha = dict(
    name="zoology.mixers.attention.MHA",
    kwargs={"num_heads": 2, "dropout": 0.1},
)

# baseline-GLA: state = num_heads * head_k_dim * head_v_dim where head_k_dim = d_model*ek/num_heads.
# With num_heads=2, d_model=128: state = 2 * (64*ek) * (64*ev) = 8192 * ek * ev.
def fla_gla(expand_k, expand_v):
    return dict(
        name="zoology.mixers.gla.GatedLinearAttention",
        kwargs={"num_heads": 2, "expand_k": expand_k, "expand_v": expand_v, "use_short_conv": False},
    )

# baseline-GDN: state = num_heads * head_dim * (head_dim * expand_v).
# With num_heads=2: state = 2 * head_dim^2 * expand_v.
def fla_gdn(head_dim, expand_v):
    return dict(
        name="zoology.mixers.gated_delta_net.GatedDeltaNet",
        kwargs={"num_heads": 2, "head_dim": head_dim, "expand_v": expand_v,
                "use_gate": False, "use_short_conv": True, "conv_size": 4},
    )

# ------- CLA-GLA curve (n_heads=4, c=8, V+1, conv + kq routing) -------

def cla_gla(d_qk, d_v):
    return dict(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs={"d_qk": d_qk, "d_v": d_v, "num_chunks": 8, "n_heads": 4,
                "writer": "softmax_gla", "reader": "softmax_linear",
                "tie_routers": False, "use_short_conv": True, "route_on": "kq"},
    )

# ------- CLA-GDN curve (n_heads=4, c=8, no V+1, conv + kq routing) -------

def cla_gdn(d_qk, d_v):
    return dict(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs={"d_qk": d_qk, "d_v": d_v, "num_chunks": 8, "n_heads": 4,
                "writer": "softmax_gdn", "reader": "softmax_linear",
                "tie_routers": False, "use_short_conv": True, "route_on": "kq"},
    )

ASYM_RECIPE_ENV = {
    "MQAR_ROUTER_STD_WRITE": "1.0",
    "MQAR_ROUTER_STD_READ":  "0.05",
    "MQAR_CURR_MODE": "linear",
    "MQAR_CURR_W_LR_PHASE1": "3.0", "MQAR_CURR_W_LR_PHASE2": "0.3",
    "MQAR_CURR_R_LR_PHASE1": "0.3", "MQAR_CURR_R_LR_PHASE2": "3.0",
}

# (name, kernel_dict, env_overrides). State sizes annotated in comments.
CELLS = [
    # ---- Baselines (7 state points each + MHA) at Zoology canonical num_heads=2 ----
    ("mha",                        mha,                {}),
    # baseline-GLA (num_heads=2): state = 8192 * ek * ev.
    ("baseline-gla-s2k",           fla_gla(0.5, 0.5),  {}),  # head_k=32,  head_v=32   → 2,048
    ("baseline-gla-s8k",           fla_gla(1.0, 1.0),  {}),  # head_k=64,  head_v=64   → 8,192
    ("baseline-gla-s33k",          fla_gla(2.0, 2.0),  {}),  # head_k=128, head_v=128  → 32,768
    ("baseline-gla-s64k",          fla_gla(2.0, 4.0),  {}),  # head_k=128, head_v=256  → 65,536
    ("baseline-gla-s128k",         fla_gla(4.0, 4.0),  {}),  # head_k=256, head_v=256  → 131,072
    ("baseline-gla-s256k",         fla_gla(4.0, 8.0),  {}),  # head_k=256, head_v=512  → 262,144
    ("baseline-gla-s524k",         fla_gla(8.0, 8.0),  {}),  # head_k=512, head_v=512  → 524,288
    # baseline-GDN (num_heads=2): state = 2 * head_dim^2 * expand_v.
    ("baseline-gdn-s2k",           fla_gdn(32, 1),     {}),  # 2*32*32   → 2,048
    ("baseline-gdn-s8k",           fla_gdn(64, 1),     {}),  # 2*64*64   → 8,192
    ("baseline-gdn-s33k",          fla_gdn(128, 1),    {}),  # 2*128*128 → 32,768
    ("baseline-gdn-s64k",          fla_gdn(128, 2),    {}),  # 2*128*256 → 65,536
    ("baseline-gdn-s128k",         fla_gdn(256, 1),    {}),  # 2*256*256 → 131,072
    ("baseline-gdn-s256k",         fla_gdn(256, 2),    {}),  # 2*256*512 → 262,144 (FLA default at num_heads=2)
    ("baseline-gdn-s524k",         fla_gdn(256, 4),    {}),  # 2*256*1024 → 524,288

    # ---- CLA-GLA state-scan (n_heads=4, c=8, V+1) × {recipe, norecipe} ----
    #  state = 32 × d_qk × (d_v + 1)
    ("cla-gla-s2k-recipe",         cla_gla(7, 8),       ASYM_RECIPE_ENV),   # 2,016
    ("cla-gla-s2k-norecipe",       cla_gla(7, 8),       {}),
    ("cla-gla-s9k-recipe",         cla_gla(16, 16),     ASYM_RECIPE_ENV),   # 8,704
    ("cla-gla-s9k-norecipe",       cla_gla(16, 16),     {}),
    ("cla-gla-s34k-recipe",        cla_gla(32, 32),     ASYM_RECIPE_ENV),   # 33,792
    ("cla-gla-s34k-norecipe",      cla_gla(32, 32),     {}),
    ("cla-gla-s66k-recipe",        cla_gla(32, 64),     ASYM_RECIPE_ENV),   # 66,560
    ("cla-gla-s66k-norecipe",      cla_gla(32, 64),     {}),
    ("cla-gla-s133k-recipe",       cla_gla(64, 64),     ASYM_RECIPE_ENV),   # 133,120
    ("cla-gla-s133k-norecipe",     cla_gla(64, 64),     {}),
    ("cla-gla-s264k-recipe",       cla_gla(64, 128),    ASYM_RECIPE_ENV),   # 264,192
    ("cla-gla-s264k-norecipe",     cla_gla(64, 128),    {}),
    ("cla-gla-s528k-recipe",       cla_gla(128, 128),   ASYM_RECIPE_ENV),   # 528,384
    ("cla-gla-s528k-norecipe",     cla_gla(128, 128),   {}),

    # ---- CLA-GDN state-scan (n_heads=4, c=8, no V+1) × {recipe, norecipe} ----
    #  state = 32 × d_qk × d_v
    ("cla-gdn-s2k-recipe",         cla_gdn(8, 8),       ASYM_RECIPE_ENV),   # 2,048
    ("cla-gdn-s2k-norecipe",       cla_gdn(8, 8),       {}),
    ("cla-gdn-s8k-recipe",         cla_gdn(16, 16),     ASYM_RECIPE_ENV),   # 8,192
    ("cla-gdn-s8k-norecipe",       cla_gdn(16, 16),     {}),
    ("cla-gdn-s33k-recipe",        cla_gdn(32, 32),     ASYM_RECIPE_ENV),   # 32,768
    ("cla-gdn-s33k-norecipe",      cla_gdn(32, 32),     {}),
    ("cla-gdn-s65k-recipe",        cla_gdn(32, 64),     ASYM_RECIPE_ENV),   # 65,536
    ("cla-gdn-s65k-norecipe",      cla_gdn(32, 64),     {}),
    ("cla-gdn-s131k-recipe",       cla_gdn(64, 64),     ASYM_RECIPE_ENV),   # 131,072
    ("cla-gdn-s131k-norecipe",     cla_gdn(64, 64),     {}),
    ("cla-gdn-s262k-recipe",       cla_gdn(64, 128),    ASYM_RECIPE_ENV),   # 262,144
    ("cla-gdn-s262k-norecipe",     cla_gdn(64, 128),    {}),
    ("cla-gdn-s524k-recipe",       cla_gdn(128, 128),   ASYM_RECIPE_ENV),   # 524,288
    ("cla-gdn-s524k-norecipe",     cla_gdn(128, 128),   {}),
]

configs = []
configs_envs = []
for name, kernel, env in CELLS:
    mixer = wrap_hybrid(kernel)
    for lr in LRS:
        for seed in SEEDS:
            configs.append(
                TrainConfig(
                    data=data,
                    model=ModelConfig(
                        block_type="TransformerBlock",
                        sequence_mixer=mixer,
                        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                        d_model=D_MODEL, n_layers=2,
                        max_position_embeddings=0,
                        vocab_size=VOCAB,
                    ),
                    logger=LoggerConfig(project_name="zoology-canonical-scan", entity=""),
                    max_epochs=32, learning_rate=lr, weight_decay=0.0, seed=seed,
                    run_id=f"{name}_lr{lr:.2e}_s{seed}",
                    early_stopping_threshold=0.99,
                    early_stopping_metric="valid/accuracy",
                    slice_keys=["num_kv_pairs"],
                )
            )
            configs_envs.append(env)
