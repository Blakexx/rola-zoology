"""Fast-benchmark baseline before Axis 1 canonical run.

Same 7 models as axis1_canonical.py, but at small state (~2,048 floats) on our
fast MQAR task (vocab=512, seq=128, kv=8, 30k train, 24 epochs). Both
max_position_embeddings = 0 and 128. 7 cells × 2 pos × 5 seeds = 70 runs.

Goals:
  - Confirm fixed-state methodology works as expected (all cells comparable)
  - Get a quick recipe-ablation signal on both pos=0 and pos=128
  - Smoke-test before expensive Axis 1 canonical run (262k state, canonical task)

All cells use internal short_conv (the "+ conv" feature), matching the Axis 1
convention. No BaseConv prefix here — kept consistent with our prior fast-
benchmark runs (cla-gla-asym, baseline-gdn-noconv, etc.) for direct comparison.
"""
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB, SEQ_LEN, NUM_KV = 512, 128, 8
TRAIN_EXAMPLES, D_MODEL, BATCH = 30_000, 128, 128
LR = 2e-3
SEEDS = [1337, 42, 7, 0, 1]
POSITIONS = [0, SEQ_LEN]   # 0 and 128

# ------- Kernel mixer configs (all at ~2,048 state) -------

# baseline-gla-conv: FLA default at num_heads=4 → 4 × 16 × 32 = 2,048 state.
fla_gla_conv = dict(
    name="zoology.mixers.gla.GatedLinearAttention",
    kwargs={"num_heads": 4, "use_short_conv": True},
)

# baseline-gdn-conv: stmatched at num_heads=2, head_dim=32, expand_v=1 → 2 × 32 × 32 = 2,048 state.
fla_gdn_conv_stmatched = dict(
    name="zoology.mixers.gated_delta_net.GatedDeltaNet",
    kwargs={"num_heads": 2, "head_dim": 32, "expand_v": 1,
            "use_gate": False, "use_short_conv": True, "conv_size": 4},
)

# CLA-GLA: n_heads=4, c=8, d_qk=7, d_v=8 + V+1 → 4 × 8 × (56+7) = 2,016 state.
# route_on='kq': writer routes on k, reader routes on q — aligns routing with the
# conv-output the kernel sees (avoids router/kernel mismatch when use_short_conv=True).
cla_gla_conv = dict(
    name="zoology.mixers.cla.ChunkedLinearAttention",
    kwargs={"d_qk": 7, "d_v": 8, "num_chunks": 8, "n_heads": 4,
            "writer": "softmax_gla", "reader": "softmax_linear",
            "tie_routers": False, "use_short_conv": True, "route_on": "kq"},
)

# CLA-GDN: n_heads=4, c=8, d_qk=8, d_v=8 → 4 × 8 × 64 = 2,048 state.
cla_gdn_conv = dict(
    name="zoology.mixers.cla.ChunkedLinearAttention",
    kwargs={"d_qk": 8, "d_v": 8, "num_chunks": 8, "n_heads": 4,
            "writer": "softmax_gdn", "reader": "softmax_linear",
            "tie_routers": False, "use_short_conv": True, "route_on": "kq"},
)

# MHA: ceiling reference.
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


# (name, kernel_dict, env_overrides) — 7 cells per position
CELLS = [
    ("cla-gla-conv-recipe",     cla_gla_conv,            ASYM_RECIPE_ENV),   # hero
    ("cla-gla-conv-norecipe",   cla_gla_conv,            {}),                # recipe ablation
    ("cla-gdn-conv-recipe",     cla_gdn_conv,            ASYM_RECIPE_ENV),   # crossover
    ("cla-gdn-conv-norecipe",   cla_gdn_conv,            {}),                # recipe ablation (GDN)
    ("baseline-gla-conv",       fla_gla_conv,            {}),                # CLA-vs-GLA isolation
    ("baseline-gdn-conv",       fla_gdn_conv_stmatched,  {}),                # the bar to beat
    ("mha",                     mha,                     {}),                # ceiling
]


def make_mixer(kernel):
    # No BaseConv prefix — matches our prior fast-benchmark runs for direct comparison.
    return ModuleConfig(name=kernel["name"], kwargs=kernel["kwargs"])


data = DataConfig(
    train_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                              num_examples=TRAIN_EXAMPLES, num_kv_pairs=NUM_KV,
                              random_non_queries=False)],
    test_configs=[MQARConfig(vocab_size=VOCAB, input_seq_len=SEQ_LEN,
                             num_examples=2_000, num_kv_pairs=NUM_KV,
                             random_non_queries=False)],
    batch_size=(BATCH, BATCH // 4),
    cache_dir="/tmp/zoology_cache_h2h",
)

configs = []
configs_envs = []
for pos in POSITIONS:
    for name, kernel, env in CELLS:
        for seed in SEEDS:
            run_id = f"{name}-pos{pos}_lr{LR:.1e}_s{seed}"
            configs.append(
                TrainConfig(
                    data=data,
                    model=ModelConfig(
                        sequence_mixer=make_mixer(kernel),
                        state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                        d_model=D_MODEL, n_layers=2,
                        max_position_embeddings=pos,
                        vocab_size=VOCAB,
                    ),
                    logger=LoggerConfig(project_name="fast-fixed-state", entity=""),
                    max_epochs=24, learning_rate=LR, weight_decay=0.0, seed=seed,
                    run_id=run_id,
                    early_stopping_threshold=0.99,
                    early_stopping_metric="valid/accuracy",
                )
            )
            configs_envs.append(env)
