"""Axis 1 — final scale-up at Zoology canonical task AND canonical state size.

All cells share state ≈ 262,144 floats per layer (FLA-GDN canonical default).
Task = Zoology canonical multi-task curriculum, evaluated at 7 difficulty levels
(kv=4/seq=64 up to kv=256/seq=1024). The figure is accuracy-vs-difficulty per
cell, all at fixed 262k state — i.e. which architecture handles the most before
breaking under a fixed memory budget.

Task config matches `mqar_example_configs/original_mqar_configs.py`:
  vocab=8192, n_layers=2, d_model=128, state_mixer=Identity, max_pos_emb=0,
  batch=256, max_epochs=32, multi-task MQAR, lr=3.2e-3,
  every kernel wrapped in Hybrid([BaseConv(k=3, implicit_long_conv=True), kernel]).

7 cells × 5 seeds = 35 runs.
  1. cla-gla-conv-recipe   : CLA-GLA, d_qk=128 d_v=128 c=8 n_heads=2 + conv + asym+curr  (264,192*)  ← hero
  2. cla-gla-conv-norecipe : same arch, no recipe                                          (264,192*)
  3. cla-gdn-conv-recipe   : CLA-GDN, d_qk=128 d_v=128 c=8 n_heads=2 + conv + asym+curr   (262,144)   ← crossover
  4. cla-gdn-conv-norecipe : same arch, no recipe                                          (262,144)
  5. baseline-gla-conv     : FLA-GLA, expand_k=4 expand_v=8 num_heads=2 + short_conv      (262,144)
  6. baseline-gdn-conv     : FLA-GDN canonical default (head_dim=256, expand_v=2, n_h=2)  (262,144)
  7. mha                   : zoology.mixers.attention.MHA, num_heads=2                    (quadratic) ← ceiling

* CLA-GLA's V+1 trick adds d_qk per chunk per head → 2×8×(128×128+128) = 264,192,
  ≈ 0.8% over canonical 262,144. Close enough for fixed-state comparison.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB = 8_192
D_MODEL = 128
LR = 3.2e-3
SEEDS = [1337, 42, 7, 0, 1]

# Multi-task train + multi-test eval (Zoology canonical).
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
BATCH = 256

data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    batch_size=(BATCH, BATCH // 8),
    cache_dir="/tmp/zoology_cache_axis1",
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


# ------- Kernel mixer configs (all at ~262,144 state, wrapped via Hybrid) -------

# 5. baseline-gla-conv: FLA-GLA scaled to 262,144 state.
#    state = (hidden^2 × expand_k × expand_v) / num_heads
#          = (128² × 4 × 8) / 2 = 262,144 ✓
#    head_k_dim = 128*4/2 = 256, head_v_dim = 128*8/2 = 512 (matches FLA-GDN canonical shape).
fla_gla_conv = dict(
    name="zoology.mixers.gla.GatedLinearAttention",
    kwargs={"num_heads": 2, "expand_k": 4.0, "expand_v": 8.0, "use_short_conv": True},
)

# 6. baseline-gdn-conv: FLA-GDN canonical defaults.
#    head_dim=256, expand_v=2, num_heads=2 → 2 × 256 × 512 = 262,144 state.
zoology_gdn_conv = dict(
    name="zoology.mixers.gated_delta_net.GatedDeltaNet",
    kwargs={"num_heads": 2, "use_gate": False, "use_short_conv": True, "conv_size": 4},
)

# 1/2. CLA-GLA at canonical state. V+1 makes per-entry = d_qk×d_v + d_qk.
#      state = 2 × 8 × (128×128 + 128) = 264,192 (+0.8% over canonical 262,144).
cla_gla_conv = dict(
    name="zoology.mixers.cla.ChunkedLinearAttention",
    kwargs={"d_qk": 128, "d_v": 128, "num_chunks": 8, "n_heads": 2,
            "writer": "softmax_gla", "reader": "softmax_linear",
            "tie_routers": False, "use_short_conv": True},
)

# 3/4. CLA-GDN at canonical state (no V+1).
#      state = 2 × 8 × 128 × 128 = 262,144 ✓
cla_gdn_conv = dict(
    name="zoology.mixers.cla.ChunkedLinearAttention",
    kwargs={"d_qk": 128, "d_v": 128, "num_chunks": 8, "n_heads": 2,
            "writer": "softmax_gdn", "reader": "softmax_linear",
            "tie_routers": False, "use_short_conv": True},
)

# 7. MHA: standard quadratic attention (no recurrent state — ceiling).
mha = dict(
    name="zoology.mixers.attention.MHA",
    kwargs={"num_heads": 2, "dropout": 0.0},
)


ASYM_RECIPE_ENV = {
    "MQAR_ROUTER_STD_WRITE": "1.0",
    "MQAR_ROUTER_STD_READ":  "0.05",
    "MQAR_CURR_MODE": "linear",
    "MQAR_CURR_W_LR_PHASE1": "3.0", "MQAR_CURR_W_LR_PHASE2": "0.3",
    "MQAR_CURR_R_LR_PHASE1": "0.3", "MQAR_CURR_R_LR_PHASE2": "3.0",
}


# (name, kernel_dict, env_overrides)
RUNS = [
    ("cla-gla-conv-recipe",     cla_gla_conv,     ASYM_RECIPE_ENV),   # 1 — hero
    ("cla-gla-conv-norecipe",   cla_gla_conv,     {}),                # 2 — recipe ablation
    ("cla-gdn-conv-recipe",     cla_gdn_conv,     ASYM_RECIPE_ENV),   # 3 — crossover
    ("cla-gdn-conv-norecipe",   cla_gdn_conv,     {}),                # 4 — recipe ablation (GDN)
    ("baseline-gla-conv",       fla_gla_conv,     {}),                # 5 — CLA-vs-GLA isolation
    ("baseline-gdn-conv",       zoology_gdn_conv, {}),                # 6 — the bar to beat
    ("mha",                     mha,              {}),                # 7 — ceiling
]


configs = []
configs_envs = []
for name, kernel, env in RUNS:
    mixer = wrap_hybrid(kernel)
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
                logger=LoggerConfig(project_name="axis1-canonical", entity=""),
                max_epochs=32, learning_rate=LR, weight_decay=0.0, seed=seed,
                run_id=f"{name}_lr{LR:.1e}_s{seed}",
                early_stopping_threshold=0.99,
                early_stopping_metric="valid/accuracy",
                slice_keys=["num_kv_pairs"],
            )
        )
        configs_envs.append(env)
