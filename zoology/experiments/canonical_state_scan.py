"""Canonical Zoology MQAR — state-scan sweep.

Task (fixed):
  vocab_size      = 8,192
  num_kv_pairs    = 16
  input_seq_len   = 64
  d_model         = 128
  n_layers        = 2
  state_mixer     = Identity
  max_pos_emb     = 0 (Zoology default)
  conv            = True on all conv-capable models (FLA short_conv for baselines,
                    internal short_conv + route_on='kq' for CLA variants)

Training: batch=256, max_epochs=32, lr=3.2e-3, early-stop@0.99 valid/acc.
Train: 20,000 examples (matches Zoology canonical kv=16 spec).

Cells (17 × 5 seeds = 85 runs):
  1. MHA                                                                 (ceiling)
  2. baseline-GLA  : FLA defaults + use_short_conv=True                  (~2,048 state)
  3. baseline-GDN  : FLA defaults + use_short_conv=True                  (~524,288 state)

  CLA-GLA curve (n_heads=4, n_chunks=8, V+1 trick, conv+kq routing):
  4. d_qk=4,  d_v=4   →   640 state, recipe
  5. d_qk=4,  d_v=4   →   640 state, no recipe
  6. d_qk=7,  d_v=8   → 2,016 state, recipe
  7. d_qk=7,  d_v=8   → 2,016 state, no recipe
  8. d_qk=16, d_v=16  → 8,704 state, recipe
  9. d_qk=16, d_v=16  → 8,704 state, no recipe
 10. d_qk=32, d_v=32  → 33,792 state, recipe
 11. d_qk=32, d_v=32  → 33,792 state, no recipe

  CLA-GDN curve (n_heads=4, n_chunks=8, no V+1, conv+kq routing):
 12. d_qk=8,  d_v=8   →  2,048 state, recipe
 13. d_qk=8,  d_v=8   →  2,048 state, no recipe
 14. d_qk=16, d_v=16  →  8,192 state, recipe
 15. d_qk=16, d_v=16  →  8,192 state, no recipe
 16. d_qk=32, d_v=32  → 32,768 state, recipe
 17. d_qk=32, d_v=32  → 32,768 state, no recipe   (added for ablation symmetry)
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB, SEQ_LEN, NUM_KV = 8_192, 64, 16
TRAIN_EXAMPLES = 20_000
D_MODEL, BATCH = 128, 256
LR = 3.2e-3
SEEDS = [1337, 42, 7, 0, 1]


# ------- Baselines -------

mha = dict(
    name="zoology.mixers.attention.MHA",
    kwargs={"num_heads": 4, "dropout": 0.0},
)

# FLA-GLA defaults at num_heads=4: head_k=16, head_v=32 → state = 4*16*32 = 2,048.
fla_gla_conv = dict(
    name="zoology.mixers.gla.GatedLinearAttention",
    kwargs={"num_heads": 4, "use_short_conv": True},
)

# FLA-GDN at FLA defaults: head_dim=256, expand_v=2, num_heads=4 → state = 4*256*512 = 524,288.
fla_gdn_conv = dict(
    name="zoology.mixers.gated_delta_net.GatedDeltaNet",
    kwargs={"num_heads": 4, "use_short_conv": True, "conv_size": 4},
)


# ------- CLA-GLA curve -------

def cla_gla(d_qk, d_v):
    return dict(
        name="zoology.mixers.cla.ChunkedLinearAttention",
        kwargs={"d_qk": d_qk, "d_v": d_v, "num_chunks": 8, "n_heads": 4,
                "writer": "softmax_gla", "reader": "softmax_linear",
                "tie_routers": False, "use_short_conv": True, "route_on": "kq"},
    )


# ------- CLA-GDN curve -------

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


# (name, kernel, env_overrides)
RUNS = [
    # Baselines
    ("mha",                   mha,          {}),                  # 1 ceiling
    ("baseline-gla",          fla_gla_conv, {}),                  # 2
    ("baseline-gdn",          fla_gdn_conv, {}),                  # 3

    # CLA-GLA curve
    ("cla-gla-s640-recipe",   cla_gla(4, 4),    ASYM_RECIPE_ENV), # 4
    ("cla-gla-s640-norecipe", cla_gla(4, 4),    {}),              # 5
    ("cla-gla-s2k-recipe",    cla_gla(7, 8),    ASYM_RECIPE_ENV), # 6
    ("cla-gla-s2k-norecipe",  cla_gla(7, 8),    {}),              # 7
    ("cla-gla-s9k-recipe",    cla_gla(16, 16),  ASYM_RECIPE_ENV), # 8
    ("cla-gla-s9k-norecipe",  cla_gla(16, 16),  {}),              # 9
    ("cla-gla-s34k-recipe",   cla_gla(32, 32),  ASYM_RECIPE_ENV), # 10
    ("cla-gla-s34k-norecipe", cla_gla(32, 32),  {}),              # 11

    # CLA-GDN curve
    ("cla-gdn-s2k-recipe",    cla_gdn(8, 8),    ASYM_RECIPE_ENV), # 12
    ("cla-gdn-s2k-norecipe",  cla_gdn(8, 8),    {}),              # 13
    ("cla-gdn-s8k-recipe",    cla_gdn(16, 16),  ASYM_RECIPE_ENV), # 14
    ("cla-gdn-s8k-norecipe",  cla_gdn(16, 16),  {}),              # 15
    ("cla-gdn-s33k-recipe",   cla_gdn(32, 32),  ASYM_RECIPE_ENV), # 16
    ("cla-gdn-s33k-norecipe", cla_gdn(32, 32),  {}),              # 17 (symmetry)
]


data = DataConfig(
    train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                              num_examples=TRAIN_EXAMPLES, num_kv_pairs=NUM_KV,
                              random_non_queries=False)],
    test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                             num_examples=1_000, num_kv_pairs=NUM_KV,
                             random_non_queries=False)],
    batch_size=(BATCH, BATCH // 8),
    cache_dir="/tmp/zoology_cache_canonical",
)

configs = []
configs_envs = []
for name, kernel, env in RUNS:
    for seed in SEEDS:
        configs.append(
            TrainConfig(
                data=data,
                model=ModelConfig(
                    block_type="TransformerBlock",
                    sequence_mixer=ModuleConfig(name=kernel["name"], kwargs=kernel["kwargs"]),
                    state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                    d_model=D_MODEL, n_layers=2,
                    max_position_embeddings=0,
                    vocab_size=VOCAB,
                ),
                logger=LoggerConfig(project_name="canonical-state-scan", entity=""),
                max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"{name}_lr{LR:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
            )
        )
        configs_envs.append(env)
